"""Real, output-supervised Q/K bridge; no p-adic differentiation."""
import numpy as np
import torch


class DualAffine(torch.nn.Module):
    def __init__(self, queries, keys, seed):
        super().__init__()
        if queries.ndim != 4 or queries.shape != keys.shape or queries.shape[-1] < 2:
            raise ValueError("matching nonempty [windows,heads,length,width] Q/K required")
        if not bool(torch.isfinite(queries).all()) or not bool(torch.isfinite(keys).all()):
            raise ValueError("finite calibration states required")
        self.seed = int(seed)
        for role, values in (("query", queries), ("key", keys)):
            values = values.float()
            self.register_buffer(f"{role}_mean", values.mean(dim=(0, 2)))
            self.register_buffer(f"{role}_std", values.std(dim=(0, 2), unbiased=False).clamp_min(1e-4))
        heads, width = queries.shape[1], queries.shape[-1]
        matrices = []
        for head in range(heads):
            rng = np.random.default_rng(np.random.SeedSequence([seed, head, 7603]))
            q, r = np.linalg.qr(rng.normal(size=(width, 2)))
            matrices.append(q * np.where(np.diag(r) < 0, -1, 1))
        matrix = torch.as_tensor(np.stack(matrices), dtype=torch.float32, device=queries.device)
        self.query_weight = torch.nn.Parameter(matrix.clone())
        self.key_weight = torch.nn.Parameter(matrix.clone())
        self.query_bias = torch.nn.Parameter(torch.zeros(heads, 2, device=queries.device))
        self.key_bias = torch.nn.Parameter(torch.zeros(heads, 2, device=queries.device))

    def encode(self, values, role):
        if role not in ("query", "key"):
            raise ValueError("role must be query or key")
        normalized = (values.float() - getattr(self, f"{role}_mean")[None, :, None]) / getattr(self, f"{role}_std")[None, :, None]
        return torch.einsum("bhld,hdr->bhlr", normalized, getattr(self, f"{role}_weight")) + getattr(self, f"{role}_bias")[None, :, None]

    def folded_arrays(self):
        arrays = {}
        with torch.no_grad():
            for role in ("query", "key"):
                weight = getattr(self, f"{role}_weight") / getattr(self, f"{role}_std")[:, :, None]
                bias = getattr(self, f"{role}_bias") - torch.einsum("hd,hdr->hr", getattr(self, f"{role}_mean"), weight)
                arrays[f"{role}_weight"] = weight.cpu().numpy().copy()
                arrays[f"{role}_bias"] = bias.cpu().numpy().copy()
        return arrays


def folded_latents(values, arrays, role):
    """Fixed block/head GEMM recipe shared by calibration and deployment."""
    if role not in ("query", "key"):
        raise ValueError("role must be query or key")
    weight = torch.as_tensor(arrays[f"{role}_weight"], device=values.device)
    bias = torch.as_tensor(arrays[f"{role}_bias"], device=values.device)
    blocks = []
    for block in values:
        blocks.append(torch.stack([torch.addmm(bias[h], block[h].float(), weight[h])
                                  for h in range(block.shape[0])]))
    return torch.stack(blocks)


def gaussian_values(query_latents, key_latents, values, positions):
    """Causal FP32 Gaussian surrogate at a declared subset of query positions."""
    if (query_latents.ndim != 4 or key_latents.ndim != 4 or query_latents.shape[-1] != 2
            or key_latents.shape[-1] != 2 or values.shape[:3] != key_latents.shape[:3]
            or query_latents.shape[:2] != key_latents.shape[:2]
            or positions.shape != (query_latents.shape[0], query_latents.shape[2])
            or positions.dtype != torch.int64 or bool((positions < 0).any())
            or bool((positions >= key_latents.shape[2]).any())):
        raise ValueError("Gaussian states/causal query positions disagree")
    delta = query_latents[:, :, :, None] - key_latents[:, :, None]
    logits = -delta.square().sum(dim=-1) / 2
    allowed = torch.arange(key_latents.shape[2], device=positions.device)[None, None, None] <= positions[:, None, :, None]
    probabilities = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1, dtype=torch.float32)
    return probabilities @ values.float()


def output_loss(predicted, target):
    if predicted.shape != target.shape or not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("matching finite predicted/target values required")
    return (predicted - target.float()).square().sum() / target.float().square().sum().clamp_min(1e-12)


def fit_encoder(queries, keys, values, target, seed, check_time):
    model = DualAffine(queries, keys, seed)
    initial = {name: a.detach().cpu().numpy().copy() for name, a in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 19001]))
    losses, window_ids, query_ids = [], [], []
    for step in range(200):
        check_time()
        blocks = rng.choice(len(queries), 2, replace=False)
        positions = np.stack([rng.choice(np.arange(128, queries.shape[2]), 32, replace=False) for _ in blocks])
        indices = torch.as_tensor(blocks, device=queries.device)
        pos = torch.as_tensor(positions, device=queries.device, dtype=torch.int64)
        q = queries[indices].gather(2, pos[:, None, :, None].expand(2, queries.shape[1], 32, queries.shape[-1]))
        truth = target[indices].gather(2, pos[:, None, :, None].expand(2, target.shape[1], 32, target.shape[-1]))
        predicted = gaussian_values(model.encode(q, "query"), model.encode(keys[indices], "key"), values[indices], pos)
        loss = output_loss(predicted, truth)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite output-supervised loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("nonfinite/missing bridge gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        window_ids.append(blocks)
        query_ids.append(positions)
    return model, initial, {"loss": np.asarray(losses), "windows": np.stack(window_ids), "queries": np.stack(query_ids)}

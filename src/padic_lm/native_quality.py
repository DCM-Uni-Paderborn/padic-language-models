"""Sequence quality of the fixed single-query native readout on original QKV."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .native_decode import FiniteBridge, depth_lookup, selected_keys, fixed_keys, attention_component

SEEDS = (17, 29, 43)
KINDS = ("p2", "p3", "coarsened_p2")
METHODS = ("native_full", "recency", "uniform") + tuple(f"s{s}_{k}" for s in SEEDS for k in KINDS)


def encode_states(q, k, frozen, guard=lambda: None):
    """Preserve actual t=1 bridge contraction rather than batch encoding."""
    states, saved = {}, {}
    for seed in SEEDS:
        encoder, book = frozen[seed]
        for family, prime, digits in (("p2", 2, (4, 4)), ("p3", 3, (3, 2))):
            bridge = FiniteBridge(encoder, {n: book[family + "_" + n]
                                  for n in ("center", "projection", "thresholds")}, prime, digits, q.device)
            pair = {}
            for role, values in (("query", q), ("key", k.repeat_interleave(3, dim=0))):
                codes, latents = [], []
                for pos in range(q.shape[1]):
                    if pos % 128 == 0: guard()
                    c, z = bridge.encode(values[:, pos:pos + 1], role)
                    codes.append(c); latents.append(z)
                pair[role] = torch.cat(codes, 1)
                saved[f"s{seed}_{family}_{role}_codes"] = pair[role].cpu().numpy()
                saved[f"s{seed}_{family}_{role}_latents"] = torch.cat(latents, 1).cpu().numpy()
            states[f"s{seed}_{family}"] = (pair["query"], pair["key"], depth_lookup(prime, digits, 1, q.device))
            if family == "p2":
                states[f"s{seed}_coarsened_p2"] = (pair["query"], pair["key"], depth_lookup(2, digits, 2, q.device))
    return states, saved


def sequence_component(q, k, v, weight, method, states, guard=lambda: None):
    """Actual chronological one-query Flash attention/projection calls."""
    if method not in METHODS or q.ndim != 3 or q.shape[0] != 15 or k.shape[0] != 5 or k.shape != v.shape:
        raise ValueError("fixed physical-GQA sequence required")
    projected, heads, ids = [], [], []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for pos in range(q.shape[1]):
            if pos % 128 == 0: guard()
            prefix = pos + 1
            if method == "native_full": chosen = None
            elif method in ("recency", "uniform"): chosen = fixed_keys(prefix, method, q.device)
            else:
                qc, kc, lut = states[method]
                chosen = selected_keys(qc[:, pos], kc[:, :prefix], lut)
            out, head = attention_component(q[:, pos:pos + 1], k[:, :prefix], v[:, :prefix], weight, chosen)
            projected.append(out); heads.append(head)
            if chosen is not None:
                padded = torch.full((5, 128), 65535, dtype=torch.int64, device=q.device)
                padded[:, :chosen.shape[1]] = chosen
                ids.append(padded)
    return torch.cat(projected, 1), torch.cat(heads, 2), (torch.stack(ids, 1).cpu().numpy().astype(np.uint16) if ids else None)


class InjectedAttention(torch.nn.Module):
    """First-layer replacement only; require original sequence projections."""
    def __init__(self, original, projected, reference):
        super().__init__()
        self.original, self.projected, self.reference = original, projected, reference
        self.config, self.layer_idx = original.config, original.layer_idx

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                cache_position=None, **kwargs):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        if self.training or past_key_values is not None or kwargs.get("past_key_value") is not None:
            raise ValueError("uncached frozen sequence intervention required")
        shape = (*hidden_states.shape[:-1], -1, self.original.head_dim)
        q = self.original.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.original.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.original.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        for name, value in (("q", q), ("k", k), ("v", v)):
            if not torch.equal(value, self.reference[name]):
                raise AssertionError("native intervention original QKV differs: " + name)
        if self.projected.shape != hidden_states.shape:
            raise ValueError("injected first-layer output shape differs")
        return self.projected, None


def replay_projected(model, original, batch, projected, reference):
    replacement = InjectedAttention(original, projected, reference).eval()
    model.model.layers[0].self_attn = replacement
    try:
        logits = model(batch, use_cache=False).logits
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                              batch[:, 1:].reshape(-1), reduction="none")
        predictions = logits[:, :-1].argmax(-1)[0]
        return nll.cpu().numpy(), predictions.cpu().numpy()
    finally:
        model.model.layers[0].self_attn = original

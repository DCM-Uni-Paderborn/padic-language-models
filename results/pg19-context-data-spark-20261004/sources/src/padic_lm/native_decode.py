"""GPU-resident exact finite-rank selection and physical-GQA decode readout.

This scans compact codes. It is not the threshold posting index, a KV eviction
scheme, or a fused end-to-end model. Backend selection belongs to the caller.
"""
import numpy as np
import torch
import torch.nn.functional as F


class FiniteBridge:
    def __init__(self, encoder, book, prime, digits, device):
        self.prime, self.digits = prime, tuple(digits)
        if (prime, self.digits) not in ((2, (4, 4)), (3, (3, 2))):
            raise ValueError("fixed binary/ternary recipes required")
        self.encoder = {k: torch.as_tensor(encoder[k], device=device)
                        for k in ("query_weight", "query_bias", "key_weight", "key_bias")}
        self.center = torch.as_tensor(book["center"], device=device)
        self.projection = torch.as_tensor(book["projection"], device=device)
        self.thresholds = torch.as_tensor(book["thresholds"], device=device)
        self.cut_counts = torch.tensor([prime ** d - 1 for d in digits], device=device)
        self.cut_valid = torch.arange(self.thresholds.shape[-1], device=device)[None] < self.cut_counts[:, None]
        reverse = []
        width = max(prime ** d for d in digits)
        for d in digits:
            row = np.zeros(width, dtype=np.int64)
            for value in range(prime ** d):
                row[value] = sum((value // prime ** j % prime) * prime ** (d - j - 1) for j in range(d))
            reverse.append(row)
        self.reverse = torch.as_tensor(np.array(reverse), device=device)
        self.coordinate = torch.arange(2, device=device)[None, None]

    def encode(self, values, role):
        """[15,tokens,64] BF16 -> [15,tokens] uint8; same actual path at t=1."""
        if role not in ("query", "key") or values.ndim != 3 or values.shape[0] != 15 or values.shape[-1] != 64:
            raise ValueError("fixed fifteen-head real bridge input required")
        latent = torch.baddbmm(self.encoder[role + "_bias"][:, None],
                              values.float(), self.encoder[role + "_weight"])
        projected = torch.bmm(latent.double() - self.center[:, None], self.projection)
        bins = ((projected[..., None] >= self.thresholds[:, None]) & self.cut_valid[None, None]).sum(-1)
        residues = self.reverse[self.coordinate, bins]
        packed = residues[..., 0] + self.prime ** self.digits[0] * residues[..., 1]
        return packed.to(torch.uint8), latent

    def tensors(self):
        return [*self.encoder.values(), self.center, self.projection,
                self.thresholds, self.cut_counts, self.cut_valid, self.reverse, self.coordinate]


def depth_lookup(prime, digits, coarsen, device):
    if (prime, tuple(digits), coarsen) not in ((2, (4, 4), 1), (3, (3, 2), 1), (2, (4, 4), 2)):
        raise ValueError("fixed finite recipes required")
    packed = np.arange(256, dtype=np.int64)
    codes = np.stack((packed % prime ** digits[0], packed // prime ** digits[0]), axis=-1)
    delta = codes[:, None] - codes[None]
    depth = np.zeros((256, 256), dtype=np.int64)
    for level in range(1, max(digits) + 1):
        depth += np.all(delta % prime ** level == 0, axis=-1)
    return torch.as_tensor((depth // coarsen).astype(np.uint8), device=device)


def selected_keys(query_codes, key_codes, lookup, *, budget=128, recent=8):
    """Exact shell histograms/midranks and unique integer score/time top-k.

    key_codes contains only the causal prefix including the current token.
    Returned positions are chronological [5,min(prefix,budget)].
    """
    heads, length = key_codes.shape
    if (heads != 15 or query_codes.shape != (15,) or not 1 <= length <= 1_000_000
            or not 0 <= recent <= budget or budget < 1):
        raise ValueError("bounded causal prefix and fixed GQA dimensions required")
    count = min(length, budget)
    if count == length:
        return torch.arange(length, device=key_codes.device).expand(5, length)
    local = min(recent, count)
    stop = length - local
    optional = count - local
    tail = torch.arange(stop, length, device=key_codes.device).expand(5, local)
    if not optional:
        return tail
    scores = lookup[query_codes.long()[:, None], key_codes[:, :stop].long()].long()
    # All fixed recipes have at most five score levels; unused bins are zero.
    histogram = torch.zeros((heads, 5), dtype=torch.int64, device=key_codes.device)
    histogram.scatter_add_(1, scores, torch.ones_like(scores))
    ranks_by_score = 2 * histogram.cumsum(-1) - histogram - 1
    ranks = ranks_by_score.gather(1, scores).reshape(5, 3, stop).sum(1)
    # Timestamp separates every tied aggregate rank; int64 has ample headroom
    # under the explicit length limit, so topk has no equal input values.
    lexicographic = ranks * stop + torch.arange(stop, device=key_codes.device)
    chosen = torch.topk(lexicographic, optional, dim=-1, sorted=False).indices
    return torch.cat((chosen, tail), dim=-1).sort(dim=-1).values


def fixed_keys(length, kind, device, budget=128, recent=8):
    count = min(length, budget)
    if kind == "recency" or count == length:
        ids = torch.arange(length - count, length, device=device)
    elif kind == "uniform":
        stop, optional = length - recent, count - recent
        ids = torch.cat((((2 * torch.arange(optional, device=device) + 1) * stop) // (2 * optional),
                         torch.arange(stop, length, device=device)))
    else:
        raise ValueError("fixed recency/uniform control required")
    return ids.expand(5, count)


def attention_component(query, keys, values, output_weight, selected=None):
    """Only five physical KV heads are gathered; optimized GQA is explicit.

    The supplied prefix is already causal: is_causal=False is essential for
    one-query decode. Backend enforcement is external to this function.
    """
    if selected is not None:
        index = selected[..., None].expand(-1, -1, keys.shape[-1])
        keys = keys.gather(1, index)
        values = values.gather(1, index)
    output = F.scaled_dot_product_attention(query[None], keys[None], values[None],
                                           dropout_p=0., is_causal=False, enable_gqa=True)
    return F.linear(output.transpose(1, 2).reshape(1, 1, -1), output_weight), output

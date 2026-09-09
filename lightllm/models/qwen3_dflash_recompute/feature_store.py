import torch
import triton
import triton.language as tl


@triton.jit
def _gather_features(POOL, NEW, REQS, STARTS, INDICES, POS, OUT, C: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    batch, slot = row // C, row % C
    req = tl.load(REQS + batch)
    start = tl.load(STARTS + batch)
    index = tl.load(INDICES + row)
    valid = tl.load(POS + row) >= 0
    old = tl.load(POOL + (req * C + index) * H + dim, valid & (index < C) & (dim < H), other=0)
    new = tl.load(NEW + (start + index - C) * H + dim, valid & (index >= C) & (dim < H), other=0)
    tl.store(OUT + row * H + dim, old + new, dim < H)


class FeatureStore:
    """One fused target feature per retained position; no persistent layer K/V.

    Request slots reset on the first prefill chunk. Updates contain only accepted
    rows. Bottom-k priorities make streaming history selection equal to selection
    from the full prefix, without retaining discarded features.
    """

    def __init__(self, requests, capacity, hidden_size, dtype, device, window, sinks, history):
        self.capacity, self.window, self.sinks, self.history = capacity, window, sinks, history
        self.features = torch.zeros((requests, capacity, hidden_size), dtype=dtype, device=device)
        self.positions = torch.full((requests, capacity), -1, dtype=torch.int64, device=device)
        self.counts = torch.zeros(requests, dtype=torch.int32, device=device)

    @torch.no_grad()
    def update(self, reqs, new_features, starts, first_positions, lengths, max_new, reset):
        capacity = self.capacity
        old = self.positions.index_select(0, reqs).masked_fill(reset[:, None], -1)
        offsets = torch.arange(max_new, device=reqs.device)
        new = (first_positions[:, None] + offsets).masked_fill(offsets[None, :] >= lengths[:, None], -1)
        positions = torch.cat((old, new), dim=1)
        invalid = positions < 0
        end = first_positions + lengths
        infinity = torch.iinfo(torch.int64).max
        if self.window:
            recent_start = end[:, None] - self.window
            priority = ((positions + 1) * 2654435761) % 2147483647
            scores = self.sinks + self.window + priority
            scores = torch.where(positions >= recent_start, self.sinks + positions - recent_start, scores)
            scores = torch.where(positions < self.sinks, positions, scores)
        else:
            scores = positions
        indices = scores.masked_fill(invalid, infinity).topk(capacity, dim=1, largest=False).indices
        selected = positions.gather(1, indices)
        order = selected.masked_fill(selected < 0, infinity).argsort(dim=1)
        selected = selected.gather(1, order).contiguous()
        indices = indices.gather(1, order).contiguous()
        gathered = torch.empty(
            (reqs.numel(), capacity, self.features.shape[-1]), device=new_features.device, dtype=new_features.dtype
        )
        _gather_features[(reqs.numel() * capacity, triton.cdiv(self.features.shape[-1], 512))](
            self.features,
            new_features,
            reqs,
            starts,
            indices,
            selected,
            gathered,
            capacity,
            self.features.shape[-1],
            512,
        )
        # Separate gather/scatter prevents readers racing an in-place slot update.
        self.features.index_copy_(0, reqs, gathered)
        self.positions.index_copy_(0, reqs, selected)
        self.counts.index_copy_(0, reqs, (selected >= 0).sum(dim=1).to(torch.int32))


@triton.jit
def _pack_kv(CONTEXT, NOISE, COUNTS, OUT, C: tl.constexpr, B: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    req, slot = row // (C + B), row % (C + B)
    count = tl.load(COUNTS + req)
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    context = tl.load(CONTEXT + (req * C + slot) * WIDTH + dim, (slot < count) & (dim < WIDTH), other=0)
    noise = tl.load(
        NOISE + (req * B + slot - count) * WIDTH + dim, (slot >= count) & (slot < count + B) & (dim < WIDTH), other=0
    )
    tl.store(OUT + row * WIDTH + dim, context + noise, dim < WIDTH)


def pack_kv(context, noise, counts, capacity, block_size):
    kv_heads, head_dim = context.shape[-2:]
    result = context.new_empty((counts.numel(), capacity + block_size, kv_heads, head_dim))
    width = kv_heads * head_dim
    _pack_kv[(counts.numel() * (capacity + block_size), triton.cdiv(width, 512))](
        context,
        noise,
        counts,
        result,
        capacity,
        block_size,
        width,
        512,
    )
    return result

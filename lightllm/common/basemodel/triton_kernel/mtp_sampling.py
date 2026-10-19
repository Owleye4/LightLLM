"""Draft sampling and chain rejection sampling for the shared MTP layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sample_mtp_token(
    target_probs,
    target_probs_stride,
    draft_probs,
    draft_probs_stride,
    uniform_sample,
    vocab_size,
    IS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Only a rejected row needs (p - q)+; bonus tokens use p directly.
    prob_sum = 0.0
    if IS_RESIDUAL:
        for start_loc in range(0, vocab_size, BLOCK_SIZE):
            offset = start_loc + tl.arange(0, BLOCK_SIZE)
            target_prob = tl.load(target_probs + offset * target_probs_stride, offset < vocab_size, other=0.0)
            draft_prob = tl.load(draft_probs + offset * draft_probs_stride, offset < vocab_size, other=0.0)
            prob_sum += tl.sum(tl.maximum(target_prob - draft_prob, 0.0))
    use_residual = IS_RESIDUAL & (prob_sum > 0.0)
    # A zero residual can occur through rounding. Falling back to p keeps the
    # sampled token in support instead of selecting an arbitrary vocabulary id.
    sample_value = uniform_sample * tl.where(use_residual, prob_sum, 1.0)
    cumulative_prob = 0.0
    sampled_token_id = vocab_size
    last_token_id = 0
    start_loc = 0
    while (start_loc < vocab_size) & (sampled_token_id == vocab_size):
        offset = start_loc + tl.arange(0, BLOCK_SIZE)
        target_prob = tl.load(target_probs + offset * target_probs_stride, offset < vocab_size, other=0.0)
        if IS_RESIDUAL:
            draft_prob = tl.load(draft_probs + offset * draft_probs_stride, offset < vocab_size, other=0.0)
            target_prob = tl.where(use_residual, tl.maximum(target_prob - draft_prob, 0.0), target_prob)
        cumulative_probs = cumulative_prob + tl.cumsum(target_prob)
        sampled_token_id = tl.min(
            tl.where((offset < vocab_size) & (cumulative_probs > sample_value), offset, vocab_size)
        )
        last_token_id = tl.maximum(last_token_id, tl.max(tl.where(target_prob > 0.0, offset, 0)))
        cumulative_prob += tl.sum(target_prob)
        start_loc += BLOCK_SIZE
    return tl.where(sampled_token_id < vocab_size, sampled_token_id, last_token_id)


@triton.jit
def _fwd_kernel_sample_mtp_draft(
    draft_probs,
    draft_probs_stride_0,
    draft_probs_stride_1,
    b_req_idx,
    req_to_mtp_seed,
    req_to_mtp_sample_offset,
    draft_token_ids,
    block_size: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    row_start = cur_index * block_size
    cur_req_idx = tl.load(b_req_idx + row_start)
    seed = tl.load(req_to_mtp_seed + cur_req_idx)
    sample_offset = tl.load(req_to_mtp_sample_offset + cur_req_idx)
    for step in range(block_size):
        row = row_start + step
        uniform_sample = tl.rand(seed, sample_offset + step)
        token_id = _sample_mtp_token(
            target_probs=draft_probs + row * draft_probs_stride_0,
            target_probs_stride=draft_probs_stride_1,
            draft_probs=draft_probs,
            draft_probs_stride=draft_probs_stride_1,
            uniform_sample=uniform_sample,
            vocab_size=vocab_size,
            IS_RESIDUAL=False,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        tl.store(draft_token_ids + row, token_id)
    tl.store(req_to_mtp_sample_offset + cur_req_idx, sample_offset + block_size)


def sample_mtp_draft(model_output, b_req_idx, sampling_params_manager, block_size=1):
    """Sample from q and return that exact distribution in the model vocabulary.

    q uses request temperature; other target filters are unnecessary for
    correctness. Greedy requests propose a point mass at the draft argmax.
    Request-local RNG counters also work inside DSpark's captured Markov head.
    """
    logits = model_output.logits.float()
    b_req_idx_long = b_req_idx.long()
    temperatures = sampling_params_manager.req_to_mtp_temperature[b_req_idx_long]
    draft_probs = torch.softmax(logits / temperatures[:, None], dim=-1)
    greedy_rows = sampling_params_manager.req_to_mtp_top_k[b_req_idx_long] == 1
    draft_probs.masked_fill_(greedy_rows[:, None], 0.0)
    draft_probs.scatter_add_(1, logits.argmax(dim=-1, keepdim=True), greedy_rows[:, None].float())
    draft_token_ids = torch.empty((logits.shape[0],), dtype=torch.int64, device=logits.device)
    if draft_token_ids.numel():
        _fwd_kernel_sample_mtp_draft[(logits.shape[0] // block_size,)](
            draft_probs=draft_probs,
            draft_probs_stride_0=draft_probs.stride(0),
            draft_probs_stride_1=draft_probs.stride(1),
            b_req_idx=b_req_idx,
            req_to_mtp_seed=sampling_params_manager.req_to_mtp_seed,
            req_to_mtp_sample_offset=sampling_params_manager.req_to_mtp_sample_offset,
            draft_token_ids=draft_token_ids,
            block_size=block_size,
            vocab_size=logits.shape[1],
            BLOCK_SIZE=1024,
            num_warps=4,
        )
    if model_output.logits_token_ids is not None:
        draft_token_ids = model_output.logits_token_ids.gather(1, draft_token_ids[:, None]).view(-1)
        full_probs = draft_probs.new_zeros((logits.shape[0], sampling_params_manager.vocab_size))
        full_probs.scatter_add_(1, model_output.logits_token_ids, draft_probs)
        draft_probs = full_probs
    return draft_token_ids, draft_probs


@triton.jit
def _fwd_kernel_mtp_reject_sampling(
    req_to_next_token_ids,
    req_to_next_token_ids_stride_0,
    req_to_next_token_ids_stride_1,
    req_to_draft_probs,
    req_to_draft_probs_stride_0,
    req_to_draft_probs_stride_1,
    req_to_draft_probs_stride_2,
    target_probs,
    target_probs_stride_0,
    target_probs_stride_1,
    b_verify_uniform,
    b_verify_uniform_stride,
    b_sample_uniform,
    b_sample_uniform_stride,
    new_next_token_ids,
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    accepted_index,
    verify_batch_size,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    req_nums = tl.num_programs(0)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    req_end_loc = tl.load(b_req_mtp_start_loc + cur_index + 1, cur_index + 1 < req_nums, other=verify_batch_size)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    req_draft_probs = req_to_draft_probs + cur_req_idx * req_to_draft_probs_stride_0
    step = 0
    continue_verify = True
    while (req_start_loc + step + 1 < req_end_loc) & continue_verify:
        cur_row = req_start_loc + step
        draft_token_id = tl.load(
            req_to_next_token_ids
            + cur_req_idx * req_to_next_token_ids_stride_0
            + (step + 1) * req_to_next_token_ids_stride_1
        )
        target_prob = tl.load(target_probs + cur_row * target_probs_stride_0 + draft_token_id * target_probs_stride_1)
        draft_prob = tl.load(
            req_draft_probs + step * req_to_draft_probs_stride_1 + draft_token_id * req_to_draft_probs_stride_2
        )
        uniform_sample = tl.load(b_verify_uniform + cur_row * b_verify_uniform_stride)
        # q == 0 marks the first decode, before a real proposal exists.
        if (draft_prob > 0.0) & (uniform_sample < tl.minimum(1.0, target_prob / draft_prob)):
            tl.store(new_next_token_ids + cur_row, draft_token_id)
            step += 1
        else:
            continue_verify = False

    cur_row = req_start_loc + step
    uniform_sample = tl.load(b_sample_uniform + cur_row * b_sample_uniform_stride)
    target_row_probs = target_probs + cur_row * target_probs_stride_0
    if continue_verify:
        token_id = _sample_mtp_token(
            target_probs=target_row_probs,
            target_probs_stride=target_probs_stride_1,
            draft_probs=req_draft_probs,
            draft_probs_stride=req_to_draft_probs_stride_2,
            uniform_sample=uniform_sample,
            vocab_size=vocab_size,
            IS_RESIDUAL=False,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        token_id = _sample_mtp_token(
            target_probs=target_row_probs,
            target_probs_stride=target_probs_stride_1,
            draft_probs=req_draft_probs + step * req_to_draft_probs_stride_1,
            draft_probs_stride=req_to_draft_probs_stride_2,
            uniform_sample=uniform_sample,
            vocab_size=vocab_size,
            IS_RESIDUAL=True,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    tl.store(new_next_token_ids + cur_row, token_id)
    tl.store(mtp_accept_len + cur_index, step + 1)
    for accepted_row in range(req_start_loc, req_end_loc):
        tl.store(accepted_index + accepted_row, accepted_row <= cur_row)


def mtp_reject_sampling(
    req_to_next_token_ids,
    req_to_draft_probs,
    target_probs,
    b_verify_uniform,
    b_sample_uniform,
    b_req_mtp_start_loc,
    b_req_idx,
):
    """Return tokens, prefix lengths including the terminal token, and row masks.

    Both random inputs share b_req_idx's flat [verify_batch_size] layout.
    """
    next_token_ids = torch.zeros_like(b_req_idx, dtype=torch.int64)
    mtp_accept_len = torch.empty_like(b_req_mtp_start_loc)
    accepted_index = torch.empty_like(b_req_idx)
    if b_req_mtp_start_loc.numel():
        _fwd_kernel_mtp_reject_sampling[(b_req_mtp_start_loc.numel(),)](
            req_to_next_token_ids=req_to_next_token_ids,
            req_to_next_token_ids_stride_0=req_to_next_token_ids.stride(0),
            req_to_next_token_ids_stride_1=req_to_next_token_ids.stride(1),
            req_to_draft_probs=req_to_draft_probs,
            req_to_draft_probs_stride_0=req_to_draft_probs.stride(0),
            req_to_draft_probs_stride_1=req_to_draft_probs.stride(1),
            req_to_draft_probs_stride_2=req_to_draft_probs.stride(2),
            target_probs=target_probs,
            target_probs_stride_0=target_probs.stride(0),
            target_probs_stride_1=target_probs.stride(1),
            b_verify_uniform=b_verify_uniform,
            b_verify_uniform_stride=b_verify_uniform.stride(0),
            b_sample_uniform=b_sample_uniform,
            b_sample_uniform_stride=b_sample_uniform.stride(0),
            new_next_token_ids=next_token_ids,
            mtp_accept_len=mtp_accept_len,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            accepted_index=accepted_index,
            verify_batch_size=b_req_idx.numel(),
            vocab_size=target_probs.shape[1],
            BLOCK_SIZE=1024,
            num_warps=4,
        )
    return next_token_ids, mtp_accept_len, accepted_index


@triton.jit
def _fwd_kernel_scatter_mtp_draft_probs(
    req_to_draft_probs,
    req_to_draft_probs_stride_0,
    req_to_draft_probs_stride_1,
    req_to_draft_probs_stride_2,
    draft_probs,
    draft_probs_stride_0,
    draft_probs_stride_1,
    draft_probs_stride_2,
    b_req_mtp_start_loc,
    b_req_idx,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    step = tl.program_id(1)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    probs = tl.load(
        draft_probs + cur_index * draft_probs_stride_0 + step * draft_probs_stride_1 + offset * draft_probs_stride_2,
        offset < vocab_size,
        other=0.0,
    )
    tl.store(
        req_to_draft_probs
        + cur_req_idx * req_to_draft_probs_stride_0
        + step * req_to_draft_probs_stride_1
        + offset * req_to_draft_probs_stride_2,
        probs,
        offset < vocab_size,
    )


def scatter_mtp_draft_probs(req_to_draft_probs, draft_probs, b_req_mtp_start_loc, b_req_idx):
    if draft_probs is None or draft_probs.numel() == 0:
        return
    _fwd_kernel_scatter_mtp_draft_probs[
        (draft_probs.shape[0], draft_probs.shape[1], triton.cdiv(draft_probs.shape[2], 1024))
    ](
        req_to_draft_probs=req_to_draft_probs,
        req_to_draft_probs_stride_0=req_to_draft_probs.stride(0),
        req_to_draft_probs_stride_1=req_to_draft_probs.stride(1),
        req_to_draft_probs_stride_2=req_to_draft_probs.stride(2),
        draft_probs=draft_probs,
        draft_probs_stride_0=draft_probs.stride(0),
        draft_probs_stride_1=draft_probs.stride(1),
        draft_probs_stride_2=draft_probs.stride(2),
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        vocab_size=draft_probs.shape[2],
        BLOCK_SIZE=1024,
        num_warps=4,
    )

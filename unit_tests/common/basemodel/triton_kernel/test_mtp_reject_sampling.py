from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.common.basemodel.triton_kernel.dynamic_mtp_utils import sample_dynamic_mtp_row_mask
from lightllm.common.basemodel.triton_kernel.mtp_sampling import (
    mtp_reject_sampling,
    sample_mtp_draft,
    scatter_mtp_draft_probs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("vocab_size", [4, 2051])
@pytest.mark.parametrize("coin, expected_len", [(0.1, 2), (0.9, 1)])
def test_ratio_residual_bonus_and_variable_width(vocab_size, coin, expected_len):
    # Noncontiguous buffers exercise every probability/token stride.
    tokens = torch.zeros((5, 8), dtype=torch.int64, device="cuda")[:, ::2]
    tokens[2, 1:] = torch.tensor([1, 2, 3], device="cuda")
    tokens[4, 1:] = torch.tensor([3, 3, 3], device="cuda")
    draft = torch.zeros((5, 3, vocab_size * 2), device="cuda")[:, :, ::2]
    draft[2, 0, 1], draft[2, 0, 2] = 0.8, 0.2
    draft[2, 1, 2] = 1.0
    draft[4, :, 3] = 1.0
    target = torch.zeros((8, vocab_size * 2), device="cuda")[:, ::2]
    target[0, 1], target[0, 2] = 0.4, 0.6
    target[1:3, vocab_size - 1] = 1.0
    target[3:6, 3] = 1.0
    target[6, vocab_size - 1] = 1.0
    target[7, 0] = 1.0
    starts = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    req_idx = torch.tensor([2, 2, 2, 4, 4, 4, 4, 1], dtype=torch.int32, device="cuda")
    coins = torch.full((16,), coin, device="cuda")[::2]
    output, lengths, mask = mtp_reject_sampling(tokens, draft, target, coins, coins, starts, req_idx)
    assert lengths.tolist() == [expected_len, 4, 1]
    assert output[:expected_len].tolist() == ([1, vocab_size - 1] if expected_len == 2 else [2])
    assert output[3:7].tolist() == [3, 3, 3, vocab_size - 1]
    assert output[7].item() == 0
    assert mask.tolist() == [int(i < expected_len) for i in range(3)] + [1, 1, 1, 1, 1]


def test_first_decode_and_request_slot_scatter():
    tokens = torch.ones((3, 3), dtype=torch.int64, device="cuda")
    cache = torch.zeros((3, 2, 4), device="cuda")
    target = torch.tensor([[0, 0, 1, 0]] * 3, dtype=torch.float32, device="cuda")
    starts = torch.tensor([0], dtype=torch.int32, device="cuda")
    req_idx = torch.full((3,), 2, dtype=torch.int32, device="cuda")
    coins = torch.full_like(req_idx, 0.5, dtype=torch.float32)
    output, lengths, mask = mtp_reject_sampling(tokens, cache, target, coins, coins, starts, req_idx)
    assert (output[0].item(), lengths.item(), mask.tolist()) == (2, 1, [1, 0, 0])
    q = target[:2].reshape(1, 2, 4)
    scatter_mtp_draft_probs(cache, q, starts, req_idx)
    torch.testing.assert_close(cache[2], q[0])
    assert cache[:2].count_nonzero() == 0
    empty = mtp_reject_sampling(tokens, cache, target[:0], coins[:0], coins[:0], starts[:0], req_idx[:0])
    assert all(t.numel() == 0 for t in empty)


def test_distribution_and_compact_draft_rng():
    n = 20000
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda")
    manager = SimpleNamespace(
        vocab_size=4,
        req_to_mtp_temperature=torch.ones(n, device="cuda"),
        req_to_mtp_top_k=torch.full((n,), 4, dtype=torch.int32, device="cuda"),
        req_to_mtp_seed=torch.arange(n, dtype=torch.int64, device="cuda"),
        req_to_mtp_sample_offset=torch.zeros(n, dtype=torch.int64, device="cuda"),
    )
    req_idx = torch.arange(n, dtype=torch.int32, device="cuda")
    proposals, q = sample_mtp_draft(ModelOutput(logits=probs.log().expand(n, -1)), req_idx, manager)
    p = torch.tensor([0.4, 0.3, 0.2, 0.1], device="cuda")
    tokens = torch.stack([torch.zeros_like(proposals), proposals], dim=1)
    target = p.expand(n * 2, -1).contiguous()
    rows = req_idx.repeat_interleave(2)
    for dynamic in (False, True):
        starts = req_idx * 2
        selected = torch.ones(n * 2, dtype=torch.bool, device="cuda")
        if dynamic:
            # Confidence is known before drawing the current candidate. Ranking
            # by q(sampled_token) would instead bias which candidates get tested.
            scores = torch.stack([torch.ones(n, device="cuda"), q.amax(-1)], dim=1)
            selected = sample_dynamic_mtp_row_mask(3 * n // 2, rows, scores, max_draft_step=1).bool()
            widths = selected.reshape(n, 2).sum(-1)
            starts = (widths.cumsum(0) - widths).int()
        output, _, _ = mtp_reject_sampling(
            tokens,
            q[:, None, :],
            target[selected],
            torch.rand_like(rows[selected], dtype=torch.float32),
            torch.rand_like(rows[selected], dtype=torch.float32),
            starts,
            rows[selected],
        )
        observed = torch.bincount(output[starts.long()], minlength=4).float() / n
        torch.testing.assert_close(observed, p, atol=0.015, rtol=0)
    manager.req_to_mtp_sample_offset.zero_()
    replay, _ = sample_mtp_draft(ModelOutput(logits=probs.log().expand(n, -1)), req_idx, manager)
    torch.testing.assert_close(replay, proposals)
    assert manager.req_to_mtp_sample_offset.unique().tolist() == [1]
    manager.req_to_mtp_top_k[:2] = 1
    compact = ModelOutput(
        logits=torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="cuda"),
        logits_token_ids=torch.tensor([[2, 3], [0, 2]], device="cuda"),
    )
    ids, q = sample_mtp_draft(compact, req_idx[:2], manager)
    assert ids.tolist() == [3, 0]
    torch.testing.assert_close(q, torch.nn.functional.one_hot(ids, 4).float())


@pytest.mark.parametrize("compact", [False, True])
def test_eagle3_draft_sampling_maps_probabilities_to_target_vocab(compact):
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend

    manager = SimpleNamespace(
        vocab_size=7,
        req_to_mtp_temperature=torch.ones(2, device="cuda"),
        req_to_mtp_top_k=torch.tensor([1, 4], dtype=torch.int32, device="cuda"),
        req_to_mtp_seed=torch.tensor([11, 22], dtype=torch.int64, device="cuda"),
        req_to_mtp_sample_offset=torch.zeros(2, dtype=torch.int64, device="cuda"),
    )
    # Two draft IDs share one target ID; q must accumulate both probabilities.
    target_ids = torch.tensor([5, 1, 6, 1], device="cuda")
    backend = SimpleNamespace(
        args=SimpleNamespace(mtp_mode="eagle3"),
        draft_models=[SimpleNamespace(map_draft_vocab_to_main_vocab=lambda ids: target_ids[ids])],
        model=SimpleNamespace(vocab_size=7, req_manager=SimpleNamespace(req_sampling_params_manager=manager)),
    )
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], device="cuda").log()
    candidate_ids = torch.tensor([[0, 3], [2, 1]], device="cuda") if compact else None
    output = ModelOutput(logits=logits.gather(1, candidate_ids) if compact else logits, logits_token_ids=candidate_ids)
    req_idx = torch.arange(2, dtype=torch.int32, device="cuda")
    token_ids, probs = ModeBackend._sample_mtp_draft(backend, output, req_idx)
    expected = torch.tensor(
        [[0, 1, 0, 0, 0, 0, 0], [0, 0.6, 0, 0, 0, 0, 0.4] if compact else [0, 0.4, 0, 0, 0, 0.4, 0.2]],
        device="cuda",
    )
    torch.testing.assert_close(probs, expected)
    assert token_ids[0].item() == 1 and torch.all(probs.gather(1, token_ids[:, None]) > 0)
    assert output.logits_token_ids is candidate_ids
    manager.req_to_mtp_sample_offset.zero_()
    replay, _ = ModeBackend._sample_mtp_draft(backend, output, req_idx)
    torch.testing.assert_close(replay, token_ids)


@pytest.mark.parametrize(
    "min_new_tokens, expected_mask",
    [(0, [False] * 4), (5, [True, True, False, False]), (10, [True] * 4)],
)
def test_mtp_eos_mask_after_prefill(monkeypatch, min_new_tokens, expected_mask):
    from lightllm.server.router.model_infer.mode_backend import generic_post_process
    from lightllm.server.router.model_infer.pin_mem_manager import PinMemTensorManager

    monkeypatch.setattr(generic_post_process, "g_pin_mem_manager", PinMemTensorManager())
    shm_param = SimpleNamespace(
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.0)),
        min_new_tokens=min_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=4,
    )
    req = SimpleNamespace(
        sampling_param=SimpleNamespace(shm_param=shm_param, invalid_token_ids=[]),
        shm_req=SimpleNamespace(input_len=1),
        get_cur_total_len=lambda: 3,
        generator=None,
        vocab_size=4,
        req_idx=0,
    )
    reqs = [req] * 4
    sampling_params_manager = SimpleNamespace(req_to_bitmask_enabled=None)
    prefill_mask = generic_post_process._get_post_sample_tensors(reqs, sampling_params_manager)[5]
    assert prefill_mask.tolist() == [expected_mask[0]] * 4
    b_mtp_index = torch.arange(4, dtype=torch.int32, device="cuda")
    mtp_mask = generic_post_process._get_post_sample_tensors(reqs, sampling_params_manager, b_mtp_index=b_mtp_index)[5]
    assert mtp_mask.dtype == torch.bool
    assert mtp_mask.tolist() == expected_mask
    next_prefill_mask = generic_post_process._get_post_sample_tensors(reqs, sampling_params_manager)[5]
    torch.testing.assert_close(next_prefill_mask, prefill_mask)

from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen3_dflash_recompute.feature_store import FeatureStore, pack_kv
from lightllm.utils.dflash_recompute import feature_pool_bytes
from lightllm.utils.sgl_utils import flash_attn_with_kvcache


def reference_positions(end, window, sinks, history):
    if window == 0:
        return list(range(end))
    mandatory = set(range(min(sinks, end))) | set(range(max(0, end - window), end))
    candidates = sorted(set(range(end)) - mandatory, key=lambda p: ((p + 1) * 2654435761) % 2147483647)
    return sorted(mandatory | set(candidates[:history]))


@pytest.mark.parametrize("window,sinks,history", [(8, 2, 5), (8, 0, 0), (8, 12, 5), (0, 0, 0)])
def test_streaming_selection_acceptance_and_reused_slot(window, sinks, history):
    capacity = window + sinks + history if window else 256
    store = FeatureStore(4, capacity, 33, torch.float32, "cuda", window, sinks, history)
    reqs = torch.tensor([2, 0], device="cuda")
    ends = [0, 0]
    for step, lengths in enumerate([(3, 7), (17, 1), (8, 8), (1, 6), (63, 9), (4, 2)]):
        if step == 4:
            ends[1] = 0  # slot 0 now belongs to an unrelated request
        max_new = max(lengths) + 7  # rejected verification rows must not survive
        first = torch.tensor(ends, device="cuda")
        positions = first[:, None] + torch.arange(max_new, device="cuda")
        features = (positions[:, :, None] * 100 + torch.arange(33, device="cuda")).float().flatten(0, 1)
        store.update(
            reqs,
            features,
            torch.tensor([0, max_new], device="cuda"),
            first,
            torch.tensor(lengths, device="cuda"),
            max_new,
            first == 0,
        )
        ends = [end + length for end, length in zip(ends, lengths)]
        for req, end in zip([2, 0], ends):
            expected = reference_positions(end, window, sinks, history)
            count = store.counts[req].item()
            assert count == len(expected)
            assert store.positions[req, :count].tolist() == expected
            want = torch.tensor(expected, device="cuda")[:, None] * 100 + torch.arange(33, device="cuda")
            torch.testing.assert_close(store.features[req, :count], want.float(), rtol=0, atol=0)
    assert store.counts[3].item() == 0  # graph padding/HOLD remains empty


def test_packed_fa3_matches_dense_attention_and_graph_reads_new_state():
    torch.manual_seed(17)
    batch, capacity, block, kv_heads, q_heads, dim = 3, 19, 7, 2, 8, 128
    context = torch.randn(batch * capacity, 2 * kv_heads, dim, dtype=torch.bfloat16, device="cuda")
    noise = torch.randn(batch * block, 2 * kv_heads, dim, dtype=torch.bfloat16, device="cuda")
    counts = torch.tensor([0, 9, 19], dtype=torch.int32, device="cuda")
    q = torch.randn(batch, block, q_heads, dim, dtype=torch.bfloat16, device="cuda")

    def run():
        scratch = pack_kv(context, noise, counts, capacity, block)
        return flash_attn_with_kvcache(
            q, scratch[:, :, :kv_heads], scratch[:, :, kv_heads:], cache_seqlens=counts + block, causal=False
        )

    def reference():
        outputs = []
        for i, count in enumerate(counts.tolist()):
            kv = torch.cat(
                (
                    context.view(batch, capacity, 2 * kv_heads, dim)[i, :count],
                    noise.view(batch, block, 2 * kv_heads, dim)[i],
                )
            )
            k, v = kv[:, :kv_heads].repeat_interleave(q_heads // kv_heads, dim=1), kv[:, kv_heads:].repeat_interleave(
                q_heads // kv_heads, dim=1
            )
            scores = torch.einsum("bhd,thd->hbt", q[i].float(), k.float()) / dim**0.5
            outputs.append(torch.einsum("hbt,thd->bhd", scores.softmax(-1), v.float()))
        return torch.stack(outputs).to(q.dtype)

    torch.testing.assert_close(run(), reference(), atol=0.016, rtol=0.016)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        actual = run()
    counts.copy_(torch.tensor([19, 1, 0], dtype=torch.int32, device="cuda"))
    context.mul_(0.37)
    graph.replay()
    torch.testing.assert_close(actual, reference(), atol=0.016, rtol=0.016)


def test_exact_feature_memory_accounting():
    args = SimpleNamespace(
        dflash_recompute_window=512, dflash_recompute_sinks=4, dflash_recompute_history=128, running_max_req_size=64
    )
    assert feature_pool_bytes(args, 2560, 2) == 65 * (644 * (5120 + 8) + 4)


def test_zero_draft_kv_still_reserves_weights(monkeypatch, tmp_path):
    import json
    from lightllm.utils import envs_utils

    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 5}))
    args = SimpleNamespace(mtp_mode="dflash_recompute", mtp_draft_model_dir=[str(tmp_path)])
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.get_mtp_weight_layer_num.cache_clear()
    try:
        assert envs_utils.get_added_mtp_kv_layer_num() == 0
        assert envs_utils.get_mtp_weight_layer_num() == 5
    finally:
        envs_utils.get_added_mtp_kv_layer_num.cache_clear()
        envs_utils.get_mtp_weight_layer_num.cache_clear()


def test_auto_profile_reserves_feature_pool(monkeypatch):
    from lightllm.utils import profile_max_tokens

    monkeypatch.setattr(profile_max_tokens, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=10000))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    actual = profile_max_tokens.get_mtp_adjusted_mem_fraction(0.9, 3600, 36, 5, extra_reserved_bytes=1000)
    assert actual == pytest.approx(0.75)


def test_decode_commit_retains_only_accepted_rows():
    from lightllm.models.qwen3_dflash_recompute.model import Qwen3DFlashRecomputeModel

    model = Qwen3DFlashRecomputeModel.__new__(Qwen3DFlashRecomputeModel)
    model.args = SimpleNamespace(mtp_step=7)
    model.pre_infer = SimpleNamespace(eps_=1e-6)
    model.pre_post_weight = SimpleNamespace(
        fc_weight_=SimpleNamespace(mm=lambda x, **_: x), hidden_norm_weight_=lambda x, **_: x
    )
    model.feature_store = FeatureStore(4, 16, 33, torch.float32, "cuda", 8, 2, 6)
    seq = torch.cat((torch.arange(11, 19), torch.arange(41, 49))).cuda().int()
    hidden = (seq[:, None] - 1 + torch.arange(33, device="cuda")).float()
    model.commit_features(
        SimpleNamespace(is_prefill=False, b_req_idx=torch.tensor([2] * 8 + [0] * 8, device="cuda"), b_seq_len=seq),
        hidden,
        torch.tensor([0, 8], device="cuda", dtype=torch.int32),
        torch.tensor([1, 8], device="cuda", dtype=torch.int32),
    )
    assert model.feature_store.positions[2, :1].tolist() == [10]
    assert model.feature_store.positions[0, :8].tolist() == list(range(40, 48))
    assert model.feature_store.counts.tolist() == [8, 0, 1, 0]
    torch.testing.assert_close(model.feature_store.features[0, :8], hidden[8:])

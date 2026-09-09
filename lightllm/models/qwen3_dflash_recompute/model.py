import torch

from lightllm.common.basemodel.attention.base_att import BaseAttBackend, BaseDecodeAttState
from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dflash_recompute.feature_store import FeatureStore
from lightllm.models.qwen3_dflash_recompute.layer_infer import Qwen3DFlashRecomputeLayerInfer
from lightllm.utils.dflash_recompute import feature_capacity, validate_feature_recompute
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class RecomputeDecodeState(BaseDecodeAttState):
    def init_state(self):
        pass

    def decode_att(self, *args, **kwargs):
        raise RuntimeError("Feature recompute uses bounded contiguous FA3 attention directly")


class RecomputeAttBackend(BaseAttBackend):
    def create_att_decode_state(self, infer_state):
        return RecomputeDecodeState(backend=self, infer_state=infer_state)


@DraftModelRegistry(model_type="qwen3", spec_modes="dflash_recompute")
class Qwen3DFlashRecomputeModel(Qwen3DFlashModel):
    transformer_layer_infer_class = Qwen3DFlashRecomputeLayerInfer

    def _verify_params(self):
        super()._verify_params()
        validate_feature_recompute(self.args)

    def _init_custom(self):
        super()._init_custom()
        self.feature_store = FeatureStore(
            self.args.running_max_req_size + 1,
            feature_capacity(self.args),
            self.config["hidden_size"],
            self.data_type,
            self._cos_cached.device,
            self.args.dflash_recompute_window,
            self.args.dflash_recompute_sinks,
            self.args.dflash_recompute_history,
        )
        store = self.feature_store
        pool_bytes = sum(t.numel() * t.element_size() for t in (store.features, store.positions, store.counts))
        logger.info(
            f"DFlash recompute: persistent draft KV layers=0, feature capacity={store.capacity}, "
            f"feature pool bytes={pool_bytes}, target KV layers={len(self.mem_manager.kv_buffer)}"
        )

    def _init_att_backend(self):
        self.prefill_att_backend = self.decode_att_backend = RecomputeAttBackend(model=self)

    @torch.no_grad()
    def commit_features(self, model_input, hidden, starts=None, accept_len=None):
        projected = self.pre_post_weight.fc_weight_.mm(hidden, use_custom_tensor_mananger=False)
        projected = self.pre_post_weight.hidden_norm_weight_(
            projected,
            eps=self.pre_infer.eps_,
            alloc_func=torch.empty,
        )
        if model_input.is_prefill:
            reqs = model_input.b_req_idx.long()
            starts = model_input.b_prefill_start_loc
            first = model_input.b_ready_cache_len
            lengths = model_input.b_seq_len - first
            max_new = model_input.max_q_seq_len
            reset = first == 0
        else:
            reqs = model_input.b_req_idx.index_select(0, starts.long()).long()
            first = model_input.b_seq_len.index_select(0, starts.long()) - 1
            lengths = accept_len
            max_new = self.args.mtp_step + 1
            reset = torch.zeros_like(reqs, dtype=torch.bool)
        self.feature_store.update(reqs, projected, starts, first, lengths, max_new, reset)

    def _decode(self, model_input):
        # Feature state is committed by the proposer. Draft block rows never own
        # target cache slots and must not overwrite its request-to-token table.
        origin_batch_size = model_input.batch_size
        assert origin_batch_size > 0 and origin_batch_size % self.block_size == 0
        infer_batch_size = origin_batch_size
        use_cuda_graph = self.graph is not None and self.graph.can_run(
            batch_size=infer_batch_size,
            max_len_in_batch=max(2, model_input.max_kv_seq_len),
        )
        need_capture = False
        if use_cuda_graph:
            infer_batch_size = self.graph.find_closest_graph_batch_size(batch_size=infer_batch_size)
            need_capture = self.graph.need_capture(infer_batch_size)
        model_input = self._create_padded_decode_model_input(model_input, infer_batch_size)
        infer_state = self._create_inferstate(model_input)
        infer_state.is_cuda_graph = need_capture
        infer_state.init_some_extra_state(self)
        infer_state.init_att_state()
        if use_cuda_graph:
            if need_capture:
                output = self.graph.capture_decode(self._token_forward, infer_state)
            else:
                output = self.graph.replay(infer_state)
        else:
            output = self._token_forward(infer_state)
        return self._create_unpad_decode_model_output(output, origin_batch_size=origin_batch_size)

    def _token_forward(self, infer_state):
        # Gather inside the graph: request IDs and persistent state change between
        # replays. Capturing a pre-gathered history would silently reuse stale data.
        reqs = infer_state.b_req_idx[:: self.block_size].long()
        store = self.feature_store
        features = store.features.index_select(0, reqs).flatten(0, 1)
        positions = store.positions.index_select(0, reqs).flatten().clamp_min(0)
        infer_state.recompute_context = (
            features,
            store.counts.index_select(0, reqs),
            self._cos_cached.index_select(0, positions),
            self._sin_cached.index_select(0, positions),
            store.capacity,
            self.block_size,
        )
        try:
            return super()._token_forward(infer_state)
        finally:
            del infer_state.recompute_context

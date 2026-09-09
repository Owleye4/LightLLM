from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
from lightllm.models.qwen3_dflash_recompute.feature_store import pack_kv
from lightllm.utils.sgl_utils import flash_attn_with_kvcache


class Qwen3DFlashRecomputeLayerInfer(Qwen3DFlashTransformerLayerInfer):
    def token_attention_forward(self, input_embdings, infer_state, layer_weight):
        q, noise_kv = self._get_qkv(input_embdings, infer_state, layer_weight)
        features, counts, cos, sin, capacity, block = infer_state.recompute_context
        context_kv = layer_weight.kv_proj.mm(features, use_custom_tensor_mananger=False)
        qk_rmsnorm_forward(
            context_kv[:, : self.tp_k_head_num_ * self.head_dim_], layer_weight.qk_norm_weight_.k_weight, self.eps_
        )
        context_kv = context_kv.view(-1, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        rotary_emb_fwd(
            context_kv[:, : self.tp_k_head_num_], None, cos, sin, partial_rotary_factor=self.partial_rotary_factor
        )
        scratch = pack_kv(context_kv, noise_kv, counts, capacity, block)
        output = flash_attn_with_kvcache(
            q.view(-1, block, self.tp_q_head_num_, self.head_dim_),
            scratch[:, :, : self.tp_k_head_num_],
            scratch[:, :, self.tp_k_head_num_ :],
            cache_seqlens=counts + block,
            causal=False,
            softmax_scale=self.head_dim_**-0.5,
        )
        return self._get_o(output.reshape(-1, self.tp_q_head_num_ * self.head_dim_), infer_state, layer_weight)

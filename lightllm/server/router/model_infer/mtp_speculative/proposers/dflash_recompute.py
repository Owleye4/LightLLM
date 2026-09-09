import torch

from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer


class DFlashRecomputeProposer(DFlashProposer):
    def fill_draft_model_kv_state(self, target_model_input, target_model_output, target_next_token_ids):
        hidden = target_model_output.mtp_collector.spec_hidden
        if hidden.numel():
            self.backend.draft_models[0].commit_features(target_model_input, hidden)

    def _commit_verify(self, model_input, model_output, starts, accept_len):
        self.backend.draft_models[0].commit_features(
            model_input,
            model_output.mtp_collector.spec_hidden,
            starts,
            accept_len,
        )

    def _allocate_scratch(self, token_num):
        # These indices only satisfy the shared ModelInput shape contract. The
        # recompute model neither writes KV nor maps them into the target table.
        model = self.backend.draft_models[0]
        return torch.full((token_num,), model.mem_manager.HOLD_TOKEN_MEMINDEX, dtype=torch.int32, device="cuda"), None

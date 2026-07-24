from __future__ import annotations

import collections
import json
import os
import threading
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.speculative.config import SpeculativeConfig
from lightllm.server.router.model_infer.speculative.planner import FixedMTPPlanner, SpecDecodePlan
from lightllm.server.router.model_infer.speculative.proposers import build_spec_proposer
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.speculative.runner import (
    SpecDecodeForwardState,
    SpecDecodePostState,
    SpecDecodeRunner,
)
from lightllm.server.router.model_infer.speculative.state import SpecForwardContext, SpecHiddenStore
from lightllm.server.router.model_infer.speculative.verifier import SpecVerifier, SpecVerifyResult


class SpecRuntime:
    """Facade between LightLLM backend code and speculative algorithms.

    The runtime keeps speculative decoding out of BaseModel and
    chunked_prefill:
    - BaseModel only calls hidden-capture methods.
    - proposer implementations own draft-model state/proposal generation.
    - SpecVerifier owns service-specific verify/scatter kernels.

    The main target->draft data path is:
    1. target model forward captures hidden features into SpecHiddenStore
    2. runtime injects those features into ModelInput.mtp_draft_input_hiddens
    3. proposer forwards the draft model and returns SpecProposal.token_ids
    4. verifier checks target acceptance and scatters candidates for the next
       iteration
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        self._target_layer_ids: Optional[List[int]] = None
        self.hidden_store = SpecHiddenStore(self)
        self.verifier = SpecVerifier(backend)
        self.proposer = build_spec_proposer(self)
        self.decode_runner = SpecDecodeRunner(self)
        self.planner = self._build_decode_planner()

        self._dflash_commit_cost_events = collections.deque()
        self._dflash_commit_cost_pending = set()
        self._dflash_block_cost_events = collections.deque()
        self._dflash_block_cost_pending = False
        self._dflash_one_step_ahead = (
            os.getenv("LIGHTLLM_DFLASH_ONE_STEP_AHEAD", "1").lower()
            in {"1", "true", "yes", "on"}
        )
        self._dflash_direct_prepare = (
            os.getenv("LIGHTLLM_DFLASH_DIRECT_PREPARE", "1").lower()
            in {"1", "true", "yes", "on"}
        )
        self._dflash_next_plans = {}
        self._dflash_next_plan_lock = threading.Lock()
        self._dflash_next_plan_hits = 0
        self._dflash_next_plan_misses = 0

    @property
    def spec_config(self) -> SpeculativeConfig:
        return self.backend.spec_config

    @property
    def enable_dynamic_mtp(self) -> bool:
        return self.spec_config.dynamic_verify

    @property
    def needs_intermediate_target_hidden(self) -> bool:
        return self.spec_config.needs_target_layer_hidden

    def is_draft_model(self, model) -> bool:
        return any(model is draft_model for draft_model in self.backend.draft_models)

    def is_block_draft_model(self, model) -> bool:
        return bool(getattr(self.spec_config, "uses_block_draft_model", False)) and self.is_draft_model(model)

    def is_draft_forward(self, infer_state, model=None) -> bool:
        return (
            infer_state.mtp_draft_input_hiddens is not None
            or getattr(infer_state, "is_draft_model", False)
            or (model is not None and self.is_draft_model(model))
        )

    def get_capture_layer_ids(self, model, infer_state) -> List[int]:
        if self.is_draft_forward(infer_state, model=model):
            if self.spec_config.uses_chained_draft_models or self.spec_config.uses_recurrent_draft_model:
                return [model.config["n_layer"] - 1]
            return []
        if self.needs_intermediate_target_hidden:
            return self._get_target_layer_ids(model)
        return []

    def create_forward_context(self, model, infer_state) -> SpecForwardContext:
        return SpecForwardContext(runtime=self, model=model, infer_state=infer_state)

    def capture_hidden(
        self,
        *,
        infer_state,
        hidden: torch.Tensor,
        final_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self.hidden_store.capture_hidden(
            infer_state=infer_state,
            hidden=hidden,
            final_hidden=final_hidden,
        )

    def get_hidden(self, microbatch_index: int = 0) -> torch.Tensor:
        return self.hidden_store.get_hidden(microbatch_index)

    def unpad_hidden(self, *, token_num: int, microbatch_index: int = 0) -> None:
        self.hidden_store.unpad_hidden(token_num=token_num, microbatch_index=microbatch_index)
        return

    def alloc_extra_mem_indexes(self, token_count: int) -> torch.Tensor:
        """Allocate speculative draft-owned temporary KV slots."""

        token_count = int(token_count)
        assert token_count >= 0
        if token_count == 0:
            return torch.empty((0,), dtype=torch.int32, device="cpu")

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(token_count)
        return g_infer_context.req_manager.mem_manager.alloc(token_count)

    def build_padded_next_token_ids(
        self,
        *,
        token_ids: Optional[torch.Tensor],
        batch_size: int,
        copy_len: int = None,
        source_start: int = 0,
        device=None,
    ) -> torch.Tensor:
        """Build a padded draft-token input buffer for padded DP batches."""

        batch_size = int(batch_size)
        source_start = int(source_start)
        assert batch_size >= 0
        assert source_start >= 0
        if token_ids is None:
            assert copy_len is None or int(copy_len) == 0
            assert device is not None
            copy_len = 0
        else:
            copy_len = token_ids.shape[0] - source_start if copy_len is None else int(copy_len)
            assert copy_len >= 0
            assert source_start + copy_len <= token_ids.shape[0]
            if device is None:
                device = token_ids.device
        assert copy_len <= batch_size

        padded_token_ids = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        if copy_len > 0:
            padded_token_ids[:copy_len].copy_(
                token_ids[source_start : source_start + copy_len],
                non_blocking=True,
            )
        return padded_token_ids

    def build_padded_eagle_step_mem_indexes(
        self,
        *,
        eagle_mem_indexes: torch.Tensor,
        step: int,
        real_req_num: int,
        padded_req_num: int,
    ) -> torch.Tensor:
        """Build one padded Eagle scratch-index column for DP decode."""

        step = int(step)
        real_req_num = int(real_req_num)
        padded_req_num = int(padded_req_num)
        assert step >= 0
        assert real_req_num >= 0
        assert padded_req_num >= 0

        start = step * real_req_num
        end = start + real_req_num
        assert end <= eagle_mem_indexes.shape[0]
        step_mem_indexes = eagle_mem_indexes[start:end]
        if padded_req_num == 0:
            return step_mem_indexes

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        hold_token_memindex = g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX
        hold_mem_indexes = torch.full(
            (padded_req_num,),
            int(hold_token_memindex),
            dtype=eagle_mem_indexes.dtype,
            device=eagle_mem_indexes.device,
        )
        return torch.cat([step_mem_indexes, hold_mem_indexes], dim=0)

    def append_padded_eagle_step_mem_indexes(
        self,
        *,
        model_input: ModelInput,
        eagle_mem_indexes: torch.Tensor,
        step: int,
        real_req_num: int,
        padded_req_num: int,
        mtp_step: int,
    ) -> torch.Tensor:
        """Roll one padded Eagle scratch-index column into ModelInput."""

        mtp_step = int(mtp_step)
        assert mtp_step >= 0
        step_mem_indexes = self.build_padded_eagle_step_mem_indexes(
            eagle_mem_indexes=eagle_mem_indexes,
            step=step,
            real_req_num=real_req_num,
            padded_req_num=padded_req_num,
        )
        grouped_mem_indexes = model_input.mem_indexes.view(-1, mtp_step + 1)
        assert grouped_mem_indexes.shape[0] == step_mem_indexes.shape[0]
        model_input.mem_indexes = torch.cat(
            [grouped_mem_indexes[:, 1:], step_mem_indexes.view(-1, 1)],
            dim=1,
        ).view(-1)
        return model_input.mem_indexes

    def prepare_draft_prefill_input(
        self,
        *,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: Optional[torch.Tensor] = None,
        microbatch_index: int = 0,
    ) -> ModelInput:
        """Build draft prefill input from target prefill input.

        `next_token_ids`: [run_req_num]
        `mtp_draft_input_hiddens`: captured target feature.  The first
        dimension matches the target prefill token layout after padding/unpad
        handling; the second dimension is either hidden_size or
        hidden_size * len(target_layer_ids).
        """

        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        return prepare_mtp_prefill_inputs(
            model_input=model_input,
            b_next_token_ids=next_token_ids,
            mtp_draft_input_hiddens=(
                self.get_hidden(microbatch_index) if mtp_draft_input_hiddens is None else mtp_draft_input_hiddens
            ),
        )

    def prepare_draft_decode_input(
        self,
        *,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: Optional[torch.Tensor] = None,
        microbatch_index: int = 0,
    ) -> ModelInput:
        """Mutate a decode ModelInput for one draft forward.

        `next_token_ids`: [verify_batch]
        `mtp_draft_input_hiddens`: [verify_batch, hidden_dim_for_draft]
        """

        model_input.input_ids = next_token_ids
        if mtp_draft_input_hiddens is None:
            mtp_draft_input_hiddens = self.get_hidden(microbatch_index)
        model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
        return model_input

    def graph_cache_key(self, model_context, model=None):
        model = getattr(model_context, "model", None) or model
        infer_state = getattr(model_context, "infer_state", model_context)
        role = "draft" if self.is_draft_forward(infer_state, model=model) else "main"
        disable_mtp_decode_att = bool(getattr(infer_state, "disable_mtp_decode_att", False))
        return ("spec", self.spec_config.mode, role, disable_mtp_decode_att)

    def get_decode_graph_mtp_step(self, model) -> int:
        is_draft_model = any(model is draft_model for draft_model in self.backend.draft_models)
        if self.spec_config.uses_block_draft_model and is_draft_model:
            runtime_block_size = int(getattr(model, "block_size", 0))
            assert runtime_block_size > 0
            return runtime_block_size - 1
        return self.spec_config.get_decode_graph_mtp_step(
            model_config=model.config,
            is_draft_model=is_draft_model,
        )

    def export_graph_capture(self):
        return self.hidden_store.export_graph_capture()

    def restore_graph_capture(self, captured_hiddens) -> None:
        self.hidden_store.restore_graph_capture(captured_hiddens)
        return

    def build_initial_draft_state(
        self,
        *,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
    ) -> None:
        self.proposer.build_initial_draft_state(model_input=model_input, next_token_ids=next_token_ids)
        return

    def build_initial_draft_state_overlap(
        self,
        *,
        model_input0: ModelInput,
        next_token_ids0: torch.Tensor,
        model_input1: ModelInput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        self.proposer.build_initial_draft_state_overlap(
            model_input0=model_input0,
            next_token_ids0=next_token_ids0,
            model_input1=model_input1,
            next_token_ids1=next_token_ids1,
        )
        return

    def start_dflash_commit_cost(self, *, batch_size: int):
        if not self.enable_dynamic_mtp or not self.spec_config.is_dflash:
            return None
        has_commit_cost = getattr(self.planner, "has_commit_cost", None)
        if has_commit_cost is None or has_commit_cost(batch_size):
            return None
        batch_size = int(batch_size)
        if batch_size in self._dflash_commit_cost_pending:
            return None
        self._dflash_commit_cost_pending.add(batch_size)
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return batch_size, start_event

    def finish_dflash_commit_cost(self, timer) -> None:
        if timer is None:
            return
        batch_size, start_event = timer
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._dflash_commit_cost_events.append((batch_size, start_event, end_event))

    def start_dflash_block_cost(self, *, block_rows: int):
        if not self.enable_dynamic_mtp or not self.spec_config.is_dflash:
            return None
        should_sample = getattr(self.planner, "should_sample_block_cost", None)
        if should_sample is None or not should_sample(block_rows) or self._dflash_block_cost_pending:
            return None
        self._dflash_block_cost_pending = True
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return int(block_rows), start_event

    def finish_dflash_block_cost(self, timer) -> None:
        if timer is None:
            return
        block_rows, start_event = timer
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._dflash_block_cost_events.append((block_rows, start_event, end_event))

    def _drain_dflash_costs(self, *, synchronize: bool) -> None:
        update_commit_cost = getattr(self.planner, "update_commit_cost", None)
        while update_commit_cost is not None and self._dflash_commit_cost_events:
            batch_size, start_event, end_event = self._dflash_commit_cost_events.popleft()
            if not synchronize and not end_event.query():
                self._dflash_commit_cost_events.appendleft((batch_size, start_event, end_event))
                break
            if synchronize:
                end_event.synchronize()
            infer_cost_ms = float(start_event.elapsed_time(end_event))
            if dist.is_available() and dist.is_initialized():
                cost = torch.tensor([infer_cost_ms], dtype=torch.float32, device="cuda")
                dist.all_reduce(cost, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
                infer_cost_ms = float(cost.item())
            update_commit_cost(batch_size=batch_size, infer_cost_ms=infer_cost_ms)
            self._dflash_commit_cost_pending.discard(batch_size)

        update_block_cost = getattr(self.planner, "update_block_cost", None)
        while update_block_cost is not None and self._dflash_block_cost_events:
            block_rows, start_event, end_event = self._dflash_block_cost_events.popleft()
            if not synchronize and not end_event.query():
                self._dflash_block_cost_events.appendleft((block_rows, start_event, end_event))
                break
            if synchronize:
                end_event.synchronize()
            infer_cost_ms = float(start_event.elapsed_time(end_event))
            if dist.is_available() and dist.is_initialized():
                cost = torch.tensor([infer_cost_ms], dtype=torch.float32, device="cuda")
                dist.all_reduce(cost, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
                infer_cost_ms = float(cost.item())
            update_block_cost(block_rows=block_rows, infer_cost_ms=infer_cost_ms)
            self._dflash_block_cost_pending = False

    @property
    def uses_dflash_direct_prepare(self) -> bool:
        return self._dflash_direct_prepare and self.enable_dynamic_mtp and self.spec_config.is_dflash

    @property
    def uses_dflash_one_step_ahead(self) -> bool:
        return (
            self._dflash_one_step_ahead
            and self.enable_dynamic_mtp
            and self.spec_config.is_dflash
            and callable(getattr(self.planner, "preview_plan", None))
            and callable(getattr(self.planner, "commit_scheduled_plan", None))
        )

    def plan_decode_shape(self, *, req_num: int, original_batch_size: int) -> SpecDecodePlan:
        """Plan from the logical full-width shape before allocating target KV."""

        self._drain_dflash_costs(synchronize=not self.uses_dflash_one_step_ahead)
        if self.uses_dflash_one_step_ahead:
            key = (int(req_num), int(original_batch_size))
            with self._dflash_next_plan_lock:
                plan = self._dflash_next_plans.pop(key, None)
            if plan is not None:
                self._dflash_next_plan_hits += 1
                commit_scheduled_plan = getattr(self.planner, "commit_scheduled_plan")
                return commit_scheduled_plan(
                    req_num=req_num,
                    original_batch_size=original_batch_size,
                    scheduled_plan=plan,
                )
            self._dflash_next_plan_misses += 1
        return self.planner.plan(req_num=req_num, original_batch_size=original_batch_size)

    def plan_decode(self, *, model_input: ModelInput, req_num: int) -> SpecDecodePlan:
        """Return the static or dynamic MTP plan for one decode iteration."""

        return self.plan_decode_shape(
            req_num=req_num,
            original_batch_size=model_input.batch_size,
        )

    def schedule_next_dflash_plan(self, *, req_num: int, original_batch_size: int) -> None:
        """Compute a future DFlash budget on the overlapped post thread."""

        if not self.uses_dflash_one_step_ahead or req_num <= 0:
            return
        self._drain_dflash_costs(synchronize=False)
        key = (int(req_num), int(original_batch_size))
        preview_plan = getattr(self.planner, "preview_plan")(
            req_num=req_num,
            original_batch_size=original_batch_size,
        )
        with self._dflash_next_plan_lock:
            # A shape can recur after newer verify feedback has arrived. Keep
            # the newest preview instead of letting an unconsumed stale entry
            # pin that shape indefinitely.
            self._dflash_next_plans[key] = preview_plan

    def run_decode_speculative_forward(
        self,
        *,
        model_input: ModelInput,
        model_output: ModelOutput,
        run_reqs: List,
        req_num: int,
        plan: SpecDecodePlan,
        selected_run_reqs_cpu: Optional[torch.Tensor],
        next_token_ids: torch.Tensor,
        next_token_logprobs: torch.Tensor,
        copy_next_token_infos: Callable[
            [torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> SpecDecodeForwardState:
        return self.decode_runner.run_speculative_forward(
            model_input=model_input,
            model_output=model_output,
            run_reqs=run_reqs,
            req_num=req_num,
            plan=plan,
            selected_run_reqs_cpu=selected_run_reqs_cpu,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            copy_next_token_infos=copy_next_token_infos,
        )

    def resolve_decode_pre_post_reqs(self, *, state: SpecDecodeForwardState, decode_reqs: List):
        return self.decode_runner.resolve_pre_post_reqs(state=state, decode_reqs=decode_reqs)

    def finish_decode_post(
        self,
        *,
        state: SpecDecodeForwardState,
        req_num: int,
        run_reqs: List,
    ) -> SpecDecodePostState:
        post_state = self.decode_runner.finish_post(state=state, req_num=req_num, run_reqs=run_reqs)
        self.schedule_next_dflash_plan(
            req_num=req_num,
            original_batch_size=req_num * (self.backend.mtp_step + 1),
        )
        return post_state

    def prepare_decode_model_input(
        self,
        *,
        model_input: ModelInput,
        req_num: int,
        plan: SpecDecodePlan,
    ):
        """Apply dynamic MTP row compaction when the planner selects it."""

        if not plan.is_dynamic:
            return model_input, None

        self._clear_stale_dynamic_token_probs(pre_draft_step=plan.pre_draft_step)

        from lightllm.common.basemodel.triton_kernel.mtp_utils import prepare_dynamic_mtp_model_input

        compact_mem_indexes_cpu = None
        if self.uses_dflash_direct_prepare and model_input.mem_indexes_cpu is None:
            compact_mem_indexes_cpu = self.alloc_extra_mem_indexes(plan.dynamic_batch_size)

        model_input, selected_run_reqs = prepare_dynamic_mtp_model_input(
            model_input=model_input,
            req_num=req_num,
            dynamic_batch_size=plan.dynamic_batch_size,
            req_to_next_token_ids=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            req_to_next_token_probs=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_probs,
            compact_mem_indexes_cpu=compact_mem_indexes_cpu,
        )
        return model_input, selected_run_reqs

    def async_copy_selected_run_reqs(self, selected_run_reqs: Optional[torch.Tensor]):
        if selected_run_reqs is None:
            return None
        from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

        return g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="selected_run_reqs",
            gpu_tensor=selected_run_reqs,
        )

    def build_decode_req_lists(
        self,
        *,
        original_run_reqs,
        selected_run_reqs_cpu: Optional[torch.Tensor],
        accepted_index_cpu: torch.Tensor,
    ):
        """Build post-handle request lists after optional dynamic MTP compaction."""

        if self.enable_dynamic_mtp:
            assert selected_run_reqs_cpu is not None
            selected_run_reqs_cpu_numpy = selected_run_reqs_cpu.numpy()
            run_reqs = [
                original_run_reqs[i] for i in range(len(original_run_reqs)) if selected_run_reqs_cpu_numpy[i] == 1
            ]
        else:
            run_reqs = original_run_reqs

        accepted_index_cpu_numpy = accepted_index_cpu.numpy()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu_numpy[i] == 1]
        return run_reqs, verify_ok_reqs

    def build_decode_free_mem_indexes_cpu(
        self,
        *,
        model_input: ModelInput,
        selected_run_reqs_cpu: Optional[torch.Tensor],
        accepted_index_cpu: torch.Tensor,
    ) -> torch.Tensor:
        mem_indexes_cpu = model_input.mem_indexes_cpu
        if not self.enable_dynamic_mtp:
            return mem_indexes_cpu[accepted_index_cpu == 0]

        assert selected_run_reqs_cpu is not None
        selected_mask = selected_run_reqs_cpu.to(dtype=torch.bool)
        accepted_mask = accepted_index_cpu.to(dtype=torch.bool)
        # Direct DFlash preparation allocates only the K selected target rows.
        # Their order already matches accepted_index_cpu, so there are no
        # unselected target slots to release.
        if mem_indexes_cpu.shape[0] == accepted_mask.shape[0]:
            return mem_indexes_cpu[~accepted_mask]

        selected_mem_indexes_cpu = mem_indexes_cpu[selected_mask]
        assert selected_mem_indexes_cpu.shape[0] == accepted_mask.shape[0]

        unselected_mem_indexes_cpu = mem_indexes_cpu[~selected_mask]
        rejected_selected_mem_indexes_cpu = selected_mem_indexes_cpu[~accepted_mask]
        if len(unselected_mem_indexes_cpu) == 0:
            return rejected_selected_mem_indexes_cpu
        if len(rejected_selected_mem_indexes_cpu) == 0:
            return unselected_mem_indexes_cpu
        return torch.cat([unselected_mem_indexes_cpu, rejected_selected_mem_indexes_cpu], dim=0)

    def update_dynamic_accept_stats(
        self,
        *,
        req_num: int,
        run_reqs,
        accepted_index_cpu: torch.Tensor,
        dynamic_batch_size: Optional[int],
        verify_step: Optional[int] = None,
    ) -> None:
        if not self.enable_dynamic_mtp:
            return

        assert dynamic_batch_size is not None
        id_to_verify_len = collections.defaultdict(int)
        id_to_accept_len = collections.defaultdict(int)
        assert len(run_reqs) == accepted_index_cpu.shape[0]
        total_count = 0
        accept_count = 0
        for req, accepted in zip(run_reqs, accepted_index_cpu.numpy()):
            if int(accepted) == 1:
                id_to_verify_len[req.req_idx] += 1
                id_to_accept_len[req.req_idx] += 1
                accept_count += 1
                total_count += 1
            else:
                id_to_verify_len[req.req_idx] += 1
                total_count += 1

        self.planner.update_req_num_to_dynamic_batch_size_to_accept_ratio(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            accept_ratio=accept_count / total_count,
            **({"verify_step": verify_step} if self.planner.planner_mode in ("eagle3", "dflash") else {}),
        )

        update_full_verify_tokens_per_req = getattr(
            self.planner,
            "update_full_verify_tokens_per_req",
            None,
        )
        is_full_verify = dynamic_batch_size == req_num * (self.backend.mtp_step + 1)
        if update_full_verify_tokens_per_req is not None and is_full_verify:
            update_full_verify_tokens_per_req(
                accept_count / req_num,
                req_num=req_num,
            )

        update_observed_iteration_stats = getattr(
            self.planner,
            "update_observed_iteration_stats",
            None,
        )
        if update_observed_iteration_stats is not None:
            update_observed_iteration_stats(
                tokens_per_req=accept_count / req_num,
                verify_rows_per_req=dynamic_batch_size / req_num,
                is_full_verify=is_full_verify,
                req_num=req_num,
            )

        # Feedback-controlled planners use these values as an unbiased survival curve. Updating it
        # from confidence-selected dynamic rows would bias every depth upward;
        # full-width warmup/probe iterations are the valid samples.
        update_verified_batch_prefix_stats = getattr(
            self.planner,
            "update_verified_batch_prefix_stats",
            None,
        )
        if is_full_verify and update_verified_batch_prefix_stats is not None:
            update_verified_batch_prefix_stats(
                verify_and_accept_lengths=[
                    (verify_len, id_to_accept_len[req_idx]) for req_idx, verify_len in id_to_verify_len.items()
                ],
            )
        elif self.planner.planner_mode not in ("eagle3", "dflash"):
            for req_idx, verify_len in id_to_verify_len.items():
                accept_len = id_to_accept_len[req_idx]
                self.planner.update_verified_prefix_stats(
                    verify_len=verify_len,
                    accept_len=accept_len,
                )
        return

    def needs_schedule_probs_cpu(self) -> bool:
        """Whether this planner consumes proposal confidence on the CPU."""

        planner_needs_schedule_probs_cpu = getattr(self.planner, "needs_schedule_probs_cpu", None)
        if planner_needs_schedule_probs_cpu is not None:
            return bool(planner_needs_schedule_probs_cpu())
        return callable(getattr(self.planner, "update_predicted_schedule_probs", None))

    def update_dynamic_schedule_stats(
        self,
        *,
        req_num: int,
        schedule_probs_cpu: Optional[torch.Tensor],
    ) -> None:
        if not self.enable_dynamic_mtp or schedule_probs_cpu is None:
            return

        update_predicted_schedule_probs = getattr(self.planner, "update_predicted_schedule_probs", None)
        if update_predicted_schedule_probs is None:
            return

        update_predicted_schedule_probs(
            schedule_probs=schedule_probs_cpu,
            req_num=req_num,
        )
        return

    def propose_next(
        self,
        *,
        main_model_input: ModelInput,
        main_model_output: Optional[ModelOutput] = None,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        verify_result: Optional[SpecVerifyResult] = None,
    ) -> SpecProposal:
        return self.proposer.propose_next(
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=draft_step,
            verify_result=verify_result,
        )

    def verify_target_tokens(
        self,
        *,
        new_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ) -> SpecVerifyResult:
        return self.verifier.verify_target_tokens(
            new_next_token_ids=new_next_token_ids,
            b_req_idx=b_req_idx,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
        )

    def build_all_next_token_probs(
        self,
        *,
        next_token_logprobs: torch.Tensor,
        proposal: SpecProposal,
        draft_step: int,
    ) -> Optional[torch.Tensor]:
        """Build selected-token probability matrix for dynamic MTP scatter.

        Output shape is [verify_batch, mtp_step + 1].  Column 0 is the target
        token probability, fixed to 1 because the target sample is always the
        base accepted position.  Draft columns store selected-token
        probabilities from each proposer step.
        """

        if not self.enable_dynamic_mtp:
            return None

        schedule_probs = proposal.schedule_probs if proposal.schedule_probs is not None else proposal.draft_probs
        assert schedule_probs is not None

        all_next_token_probs = torch.zeros(
            size=(next_token_logprobs.shape[0], self.backend.mtp_step + 1),
            dtype=torch.float32,
            device=next_token_logprobs.device,
        )
        all_next_token_probs[:, 0] = 1.0

        if isinstance(schedule_probs, torch.Tensor):
            assert schedule_probs.shape == (next_token_logprobs.shape[0], draft_step)
            if draft_step > 0:
                all_next_token_probs[:, 1 : draft_step + 1] = schedule_probs
            return all_next_token_probs

        assert len(schedule_probs) == draft_step
        for step_idx, step_probs in enumerate(schedule_probs):
            all_next_token_probs[:, step_idx + 1] = step_probs
        return all_next_token_probs

    def pad_all_next_token_ids(self, *, token_ids: torch.Tensor, draft_step: int) -> torch.Tensor:
        """Pad dynamic proposal ids to the static MTP width before scatter."""

        if not self.enable_dynamic_mtp or draft_step >= self.backend.mtp_step:
            return token_ids

        append_next_token_ids = torch.ones(
            size=(token_ids.shape[0], self.backend.mtp_step - draft_step),
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        return torch.cat([token_ids, append_next_token_ids], dim=-1)

    def scatter_next_tokens(
        self,
        *,
        b_req_mtp_start_loc: torch.Tensor,
        all_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        all_next_token_probs: Optional[torch.Tensor] = None,
    ) -> None:
        self.verifier.scatter_next_tokens(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
            all_next_token_probs=all_next_token_probs,
        )
        return

    def scatter_token_id_steps(
        self,
        *,
        token_id_steps: List[torch.Tensor],
        b_req_mtp_start_loc: torch.Tensor,
        b_req_idx: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        row_count: int = None,
    ) -> torch.Tensor:
        """Stack proposal token columns and scatter them for the next verify."""

        all_next_token_ids = torch.stack(token_id_steps, dim=1)
        if row_count is not None:
            all_next_token_ids = all_next_token_ids[: int(row_count), :]
        self.scatter_next_tokens(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return all_next_token_ids

    def _get_target_layer_ids(self, model) -> List[int]:
        if self._target_layer_ids is not None:
            return self._target_layer_ids

        draft_model_dir = self.backend.args.mtp_draft_model_dir
        if isinstance(draft_model_dir, list):
            draft_model_dir = draft_model_dir[0]

        if draft_model_dir:
            with open(os.path.join(draft_model_dir, "config.json"), "r") as json_file:
                draft_config = json.load(json_file)
            target_layer_ids = draft_config.get("target_layer_ids")
            if target_layer_ids is not None:
                self._target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
                return self._target_layer_ids

        self._target_layer_ids = [1, model.config["n_layer"] // 2 - 1, model.config["n_layer"] - 4]
        return self._target_layer_ids

    def _build_decode_planner(self):
        if self.enable_dynamic_mtp:
            from lightllm.server.router.model_infer.infer_batch import g_infer_context

            assert g_infer_context.dynamic_mtp_planner is not None
            return g_infer_context.dynamic_mtp_planner
        return FixedMTPPlanner(self.backend.mtp_step)

    def _clear_stale_dynamic_token_probs(self, *, pre_draft_step: int) -> None:
        # Columns after the previous draft length are stale and must not be
        # sampled by dynamic row compaction in the current target forward.
        self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_probs[
            :, (pre_draft_step + 1) :
        ].fill_(0.0)
        return


def build_spec_runtime(backend) -> SpecRuntime:
    return SpecRuntime(backend)

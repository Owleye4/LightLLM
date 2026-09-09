"""Capacity accounting and supported serving configurations for feature recompute."""


def feature_capacity(args):
    window, sinks, history = (
        args.dflash_recompute_window,
        args.dflash_recompute_sinks,
        args.dflash_recompute_history,
    )
    if min(window, sinks, history) < 0:
        raise ValueError("DFlash feature budgets must be nonnegative")
    return window + sinks + history if window else args.max_req_total_len + 3 * args.mtp_step + 8


def feature_pool_bytes(args, hidden_size, element_size):
    return (args.running_max_req_size + 1) * (feature_capacity(args) * (hidden_size * element_size + 8) + 4)


def validate_feature_recompute(args):
    feature_capacity(args)
    if args.run_mode != "normal" or args.dp != 1:
        raise ValueError("dflash_recompute currently requires normal deployment and dp=1")
    if not args.disable_dynamic_prompt_cache:
        raise ValueError("dflash_recompute requires --disable_dynamic_prompt_cache to retain complete feature state")
    for option in (
        "enable_cpu_cache",
        "enable_dp_prompt_cache_fetch",
        "diverse_mode",
        "enable_tpsp_mix_mode",
        "enable_prefill_decode_mixed",
        "enable_decode_microbatch_overlap",
        "enable_prefill_microbatch_overlap",
        "mtp_dynamic_verify",
    ):
        if getattr(args, option, False):
            raise ValueError(f"dflash_recompute does not yet support --{option}")

"""Explicit experimental output identity and checkpoint compatibility."""
MODES = ("state", "terminal_correction")


def output_mode(args):
    mode = getattr(args, "diffusion_output", "state")
    if mode not in MODES:
        raise ValueError(f"Unknown diffusion_output={mode!r}")
    return mode


def mode_stem(stem, mode):
    suffix = "_terminal_correction"
    return stem + suffix if mode == "terminal_correction" and not stem.endswith(suffix) else stem


def check_checkpoint_mode(state, expected):
    actual = state.get("extra", {}).get("diffusion_output", "state")
    if actual != expected:
        raise ValueError(f"Refiner diffusion_output mismatch: checkpoint={actual}, requested={expected}; "
                         "use matching trained weights. Initialization is separate from resume/test.")

"""Stage-2D estimated-geometry diffusion adaptation without post-sampling alpha scaling.

Strict ablation of train_cdrdi_diffusion_estimated.py:
- keep 10% identity / 90% nonregistered sampling;
- keep the same raw synthetic geometry ranges and Jacobian constraint;
- keep all optimizer/model/checkpoint/evaluation settings from the baseline trainer;
- remove only the post-sampling alpha scaling of dx/dy/theta/control/local_field.

For paired-seed comparability, the alpha RNG draw is still consumed but its
value is intentionally not applied.  This keeps the later random-number stream
aligned with the baseline as closely as possible.
"""

from __future__ import annotations

from typing import List

import torch

from cdrdi_geometry import SyntheticGeometry, sample_synthetic_geometry
import train_cdrdi_diffusion_estimated as baseline


def _identity_geometry(
    batch: int,
    h: int,
    w: int,
    *,
    device,
    dtype,
    control_grid: int,
) -> SyntheticGeometry:
    return SyntheticGeometry(
        dx=torch.zeros(batch, device=device, dtype=dtype),
        dy=torch.zeros(batch, device=device, dtype=dtype),
        theta_deg=torch.zeros(batch, device=device, dtype=dtype),
        control=torch.zeros(
            batch, 2, control_grid, control_grid, device=device, dtype=dtype
        ),
        local_field=torch.zeros(batch, 2, h, w, device=device, dtype=dtype),
    )


def _cat_geometry(items: List[SyntheticGeometry]) -> SyntheticGeometry:
    return SyntheticGeometry(
        dx=torch.cat([g.dx for g in items], dim=0),
        dy=torch.cat([g.dy for g in items], dim=0),
        theta_deg=torch.cat([g.theta_deg for g in items], dim=0),
        control=torch.cat([g.control for g in items], dim=0),
        local_field=torch.cat([g.local_field for g in items], dim=0),
    )


def sample_training_geometry_noalpha(
    batch: int,
    h: int,
    w: int,
    *,
    device,
    dtype,
    generator,
    args,
) -> SyntheticGeometry:
    """Baseline mixed geometry sampler with alpha scaling disabled."""
    items: List[SyntheticGeometry] = []
    for _ in range(int(batch)):
        identity_draw = float(
            torch.rand((), device=device, generator=generator).item()
        )
        if identity_draw < float(args.identity_probability):
            items.append(
                _identity_geometry(
                    1,
                    h,
                    w,
                    device=device,
                    dtype=dtype,
                    control_grid=args.control_grid,
                )
            )
            continue

        phi = sample_synthetic_geometry(
            h,
            w,
            device=device,
            dtype=dtype,
            generator=generator,
            max_translation=args.max_translation,
            max_rotation_deg=args.max_rotation_deg,
            max_local_px=args.max_local_px,
            control_grid=args.control_grid,
            min_jacobian=args.min_jacobian,
        )

        # Consume exactly the same extra random draw as the baseline sampler so
        # later stochastic choices stay aligned for the same seed.  The sampled
        # alpha is intentionally NOT multiplied into the deformation.
        _unused_alpha = float(args.min_strength) + (
            1.0 - float(args.min_strength)
        ) * float(torch.rand((), device=device, generator=generator).item())
        del _unused_alpha

        items.append(phi)

    return _cat_geometry(items)


def main() -> None:
    # train_cdrdi_diffusion_estimated resolves this module-global sampler at
    # runtime in train_one_epoch(), so replacing only this symbol preserves all
    # other baseline behavior and command-line arguments.
    baseline.sample_training_geometry = sample_training_geometry_noalpha
    print(
        "NOALPHA_ABLATION enabled: training keeps identity/nonregistered ratio "
        "and raw deformation ranges, but does not apply post-sampling alpha."
    )
    baseline.main()


if __name__ == "__main__":
    main()

"""No-alpha ablation for final GIGI-CDRDI mixed adaptation.

This is a strict wrapper around train_innovation3_gigi_cdrdi.py.  The baseline
training protocol is unchanged except that the post-sampling deformation
strength multiplier alpha is not applied.

Baseline mixed sampling:
    identity with probability args.identity_probability;
    otherwise sample full synthetic geometry, then multiply every geometry
    component by alpha ~ U(args.min_strength, 1).

No-alpha ablation:
    identity with exactly the same probability;
    otherwise keep the sampled synthetic geometry at its native sampled
    strength.  The alpha random draw is still consumed, but ignored, so a fixed
    seed preserves the baseline random stream as closely as possible.

Validation/test geometry generation remains untouched and therefore continues
to use the original full-range evaluation protocol.
"""

from __future__ import annotations

from typing import List

import torch

from cdrdi_geometry import SyntheticGeometry, sample_synthetic_geometry
import train_innovation3_gigi_cdrdi as baseline


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
    """Mixed sampler identical to baseline except alpha is not applied."""
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

        # Keep the RNG stream paired with the baseline experiment.  The old
        # trainer used this draw to scale dx/dy/theta/control/local_field.
        # Here the draw is deliberately consumed but never applied.
        _unused_alpha = float(args.min_strength) + (
            1.0 - float(args.min_strength)
        ) * float(torch.rand((), device=device, generator=generator).item())
        del _unused_alpha

        items.append(phi)

    return _cat_geometry(items)


def main() -> None:
    # train() resolves sample_training_geometry from its module globals at
    # runtime.  Replacing this one symbol leaves every other training/test
    # behavior, CLI default, frozen checkpoint and monitor unchanged.
    baseline.sample_training_geometry = sample_training_geometry_noalpha
    print(
        "NOALPHA_ABLATION enabled: mixed GIGI-CDRDI training uses the same "
        "identity probability and full raw deformation sampler, with no "
        "post-sampling alpha multiplier. Validation/test are unchanged."
    )
    baseline.main()


if __name__ == "__main__":
    main()

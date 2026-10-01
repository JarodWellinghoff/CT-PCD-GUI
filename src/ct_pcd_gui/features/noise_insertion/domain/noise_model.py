from __future__ import annotations

import numpy as np


def add_poisson_noise_log_domain(
    attenuation: np.ndarray,
    incident_photon_statistics: np.ndarray,
    mas_factor: float,
    electronic_noise: float = 0.0,
    rng: np.random.Generator | None = None,
    *,
    fine_tune_factor: float | None = None,
) -> np.ndarray:
    """Apply the v2 projection-domain noise model.

    ``mas_factor`` remains the nominal dose fraction used for output metadata.
    The fine-tune calibration is read from the process-safe ``MasFactor`` value
    object used by ``NoiseJobConfig`` unless explicitly overridden here.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 < mas_factor <= 1.0):
        raise ValueError("mAs factor must be in (0, 1].")

    resolved_fine_tune = (
        float(fine_tune_factor)
        if fine_tune_factor is not None
        else float(getattr(mas_factor, "fine_tune_factor", 1.0))
    )
    if resolved_fine_tune <= 0.0:
        raise ValueError("Fine-tune factor must be greater than 0.")

    # The v2 reader performs rescale arithmetic in float32, then promotes to
    # float64 for the variance calculation. Quantize here so the established
    # executor's float64 input follows the same numerical path.
    p_a = np.asarray(attenuation, dtype=np.float32).astype(np.float64)
    n1 = np.asarray(incident_photon_statistics, dtype=np.float64).ravel()
    if p_a.ndim != 2:
        raise ValueError(f"Expected a 2-D projection, received shape {p_a.shape}.")

    height, width = p_a.shape
    if n1.size == height:
        n1_map = np.broadcast_to(n1[:, np.newaxis], (height, width))
    elif n1.size == width:
        n1_map = np.broadcast_to(n1[np.newaxis, :], (height, width))
    elif n1.size == height * width:
        n1_map = n1.reshape(height, width)
    else:
        raise ValueError(
            f"PhotonStatistics length ({n1.size}) does not match projection "
            f"shape {p_a.shape} on either axis."
        )

    attenuation_factor = np.exp(-np.clip(p_a, -50.0, 50.0))
    n1_detected = np.maximum(n1_map * attenuation_factor, 1e-6)
    n2_detected = np.maximum(
        n1_detected * float(mas_factor) * resolved_fine_tune,
        1e-6,
    )

    variance = (1.0 / n2_detected - 1.0 / n1_detected) * (
        1.0 + electronic_noise / n2_detected + electronic_noise / n1_detected
    )
    variance = np.maximum(variance, 0.0)
    noisy = p_a + np.sqrt(variance) * rng.standard_normal(size=p_a.shape)
    return noisy.astype(np.float32)

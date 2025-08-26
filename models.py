"""
Phenomenological SNSPD models for a teaching/demo sandbox.
The goal: produce realistic-looking trends with simple equations and tunable constants.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class Constants:
    # Kinetic inductance constant to land Lk in ~10^2 nH range for realistic geometries.
    # Lk_nH ≈ K_LK * length_um / (width_nm * thickness_nm)
    K_LK: float = 30.0  # nH * nm^2 / um

    # IDE logistic curve steepness
    IDE_STEEPNESS: float = 20.0

    # IDE logistic center (baseline at 1550 nm, 100 nm width)
    IDE_CENTER_BASE: float = 0.9  # fraction of Ic

    # Wavelength/geometry sensitivities for IDE center
    IDE_CENTER_LAMBDA_SENS: float = 0.05  # per (Δλ/500 nm)
    IDE_CENTER_WIDTH_SENS: float = 0.05   # per (Δw/50 nm)

    # DCR model
    DCR_BASE_HZ: float = 1.0       # baseline at Ib/Ic=0.8, T=2 K
    DCR_BIAS_GAIN: float = 15.0    # exp slope vs (Ib/Ic)
    DCR_E_OVER_K: float = 20.0     # "activation" in Kelvin (phenomenological)
    DCR_BB_BASE: float = 10.0      # blackbody/background floor at 1550 nm

    # Coupling factor (fiber alignment etc.)
    COUPLING: float = 0.9

    # Jitter model
    JITTER_RISE_PS: float = 100.0  # nominal rise time for slope calculation
    JITTER_NOISE_COEFF: float = 0.02  # scales 1/slope term
    JITTER_BASE_PS: float = 20.0      # baseline (geometry/hotspot variance)

    # Pulse amplitude scaling (arbitrary units)
    AMP_SCALE: float = 1.0

    # Latching heuristic thresholds
    LATCH_THRESH: float = 3.0
    LATCH_SCALE_LK_NH: float = 100.0  # scale Lk into heuristic
    LATCH_REF_RLOAD: float = 50.0

const = Constants()


def clamp(x, lo, hi):
    return max(lo, min(float(x), hi))


def kinetic_inductance_nH(length_um: float, width_nm: float, thickness_nm: float, c: Constants = const) -> float:
    """Lk in nH."""
    length_um = max(1e-3, length_um)
    width_nm = max(1e-3, width_nm)
    thickness_nm = max(1e-3, thickness_nm)
    lk = c.K_LK * length_um / (width_nm * thickness_nm)
    return float(lk)


def ide_internal(ib_over_ic: float, wavelength_nm: float, width_nm: float, c: Constants = const) -> float:
    """
    Internal detection efficiency as a logistic vs normalized bias.
    The logistic "center" shifts with wavelength and width (narrower wires help long-λ).
    """
    ib = clamp(ib_over_ic, 0.0, 1.1)
    # Shift center around base using wavelength and width
    dlam = (wavelength_nm - 1550.0) / 500.0
    dw = (100.0 - width_nm) / 50.0
    center = c.IDE_CENTER_BASE + c.IDE_CENTER_LAMBDA_SENS * dlam + c.IDE_CENTER_WIDTH_SENS * dw
    center = float(np.clip(center, 0.7, 0.98))
    x = c.IDE_STEEPNESS * (ib - center)
    ide = 1.0 / (1.0 + np.exp(-x))
    return float(np.clip(ide, 0.0, 1.0))


def sde_system(ide: float, absorption: float, c: Constants = const) -> float:
    """System detection efficiency SDE = absorption × coupling × IDE."""
    ide = float(np.clip(ide, 0.0, 1.0))
    absorption = float(np.clip(absorption, 0.0, 1.0))
    sde = absorption * c.COUPLING * ide
    return float(np.clip(sde, 0.0, 1.0))


def dcr_hz(ib_over_ic: float, T_K: float, wavelength_nm: float, c: Constants = const) -> float:
    """Dark-count rate: Arrhenius-like × bias gain + blackbody floor (very rough)."""
    ib = clamp(ib_over_ic, 0.0, 1.1)
    T = max(0.5, float(T_K))
    bias_term = np.exp(c.DCR_BIAS_GAIN * (ib - 0.8))
    arrhenius = np.exp(-c.DCR_E_OVER_K / T)
    bb = c.DCR_BB_BASE * (wavelength_nm / 1550.0) ** 2
    dcr = c.DCR_BASE_HZ * bias_term * arrhenius + bb
    return float(max(0.0, dcr))


def pulse_waveform(t_ns: np.ndarray, ib_over_ic: float, rload_ohm: float, lk_nH: float, c: Constants = const):
    """
    Simple SNSPD voltage pulse (arb. units): fast rise, exponential decay with tau = Lk / R.
    """
    lk_H = max(1e-12, lk_nH * 1e-9)
    R = max(1.0, rload_ohm)
    tau_ns = (lk_H / R) * 1e9  # ns
    # Peak amplitude scales with ib_over_ic (bounded) and arbitrarily with c.AMP_SCALE
    ib = float(np.clip(ib_over_ic, 0.0, 1.1))
    v_peak = c.AMP_SCALE * (0.5 + 0.5 * ib)

    # Approximate rise with a 1 - exp(-t/tr) with tr = JITTER_RISE_PS/1000 ns, then decay exp(-t/tau)
    tr_ns = c.JITTER_RISE_PS / 1000.0
    rise = 1.0 - np.exp(-np.maximum(0, t_ns) / max(1e-3, tr_ns))
    decay = np.exp(-np.maximum(0, t_ns) / max(1e-3, tau_ns))
    v = v_peak * rise * decay
    return v, tau_ns, v_peak


def jitter_fwhm_ps(v_peak: float, c: Constants = const) -> float:
    """
    Combine amplitude/slope-limited jitter with a baseline:
    jitter_amp_ps ~ C * tr / slope ≈ C * tr / (V_peak / tr) = C * tr^2 / V_peak  (phenomenological)
    """
    tr_ps = c.JITTER_RISE_PS
    # Avoid div by zero
    jitter_amp = c.JITTER_NOISE_COEFF * (tr_ps ** 2) / max(1e-6, v_peak)
    total = np.sqrt(jitter_amp ** 2 + c.JITTER_BASE_PS ** 2)
    return float(total)


def latching_risk(ib_over_ic: float, rload_ohm: float, lk_nH: float, c: Constants = const):
    """
    Heuristic: higher bias, lower Lk, lower Rload can raise risk of latching.
    score = (Ib/Ic) * (Rload / 50) * (100 / Lk[nH])
    """
    score = float(np.clip(ib_over_ic, 0.0, 1.2)) * (rload_ohm / c.LATCH_REF_RLOAD) * (c.LATCH_SCALE_LK_NH / max(1e-3, lk_nH))
    if score < 1.0:
        level = "Low"
    elif score < c.LATCH_THRESH:
        level = "Moderate"
    else:
        level = "High"
    return score, level

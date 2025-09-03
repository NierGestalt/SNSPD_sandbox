# app.py — SNSPD Performance Sandbox (with presets)
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from models import (
    kinetic_inductance_nH,
    ide_internal,
    sde_system,
    dcr_hz,
    pulse_waveform,
    jitter_fwhm_ps,
    latching_risk,
    const,
)

# ------------------ Page header ------------------
st.set_page_config(page_title="SNSPD Performance Sandbox", layout="wide")
st.title("SNSPD Performance Sandbox")
st.caption(
    "Phenomenological, slider-driven demo to explore IDE/SDE, DCR, jitter, pulse shape, and latching heuristics. "
    "This model is accurate in a limited sense (quantum-source trends) but simplifies real device physics."
)

# ------------------ Presets ------------------
# Slider/session_state keys the app uses
KEYS = [
    "ib_over_ic", "lambda_nm", "temp_K", "absorption", "coupling",
    "width_nm", "thickness_nm", "fill_factor", "length_um", "rload_ohm",
]

PRESETS = {
    "Match-our-spec": {
        # Goal: high SDE @ 1550 nm (NbTiN ~90 nm, Lk~353 nH, 50 Ω)
        "ib_over_ic": 1.02,
        "lambda_nm": 1550.0,
        "temp_K": 4.0,
        "absorption": 0.97,
        "coupling": 1.00,
        "width_nm": 90.0,
        "thickness_nm": 7.0,
        "fill_factor": 0.50,
        "length_um": 7413.0,   # -> ~353 nH in this toy model
        "rload_ohm": 50.0,
    },
    "High-throughput (faster reset)": {
        # Goal: shorter tau for count-rate talking point
        "ib_over_ic": 0.95,
        "lambda_nm": 1550.0,
        "temp_K": 4.0,
        "absorption": 0.90,
        "coupling": 0.95,
        "width_nm": 90.0,
        "thickness_nm": 7.0,
        "fill_factor": 0.50,
        "length_um": 5000.0,   # -> ~238 nH
        "rload_ohm": 100.0,
    },
    "Low-noise demo": {
        # Goal: show DCR reduction (cooler + shorter λ) with some SDE trade
        "ib_over_ic": 0.86,
        "lambda_nm": 800.0,
        "temp_K": 2.0,
        "absorption": 0.97,
        "coupling": 0.95,
        "width_nm": 90.0,
        "thickness_nm": 7.0,
        "fill_factor": 0.50,
        "length_um": 7413.0,   # same geometry as spec
        "rload_ohm": 50.0,
    },
}

def apply_preset(values: dict, label: str):
    for k, v in values.items():
        st.session_state[k] = v
    st.session_state["_last_preset"] = label
    st.toast(f"Preset applied: {label}", icon="✅")
    st.rerun()

def preset_bar():
    st.subheader("One-click presets")
    c1, c2, c3 = st.columns(3)
    if c1.button("Match-our-spec", use_container_width=True):
        apply_preset(PRESETS["Match-our-spec"], "Match-our-spec")
    if c2.button("High-throughput (faster reset)", use_container_width=True):
        apply_preset(PRESETS["High-throughput (faster reset)"], "High-throughput")
    if c3.button("Low-noise demo", use_container_width=True):
        apply_preset(PRESETS["Low-noise demo"], "Low-noise demo")

preset_bar()

# ------------------ Sidebar controls ------------------
with st.sidebar:
    st.header("Controls")

    st.slider("Normalized Bias (Ib/Ic)", 0.0, 1.10, 0.85, 0.01, key="ib_over_ic")
    st.slider("Wavelength (nm)", 800, 2200, 1550, 10, key="lambda_nm")
    st.slider("Temperature (K)", 0.8, 10.0, 2.0, 0.1, key="temp_K")
    st.slider("Optical Absorption (0–1)", 0.0, 1.0, 0.80, 0.01, key="absorption")
    st.slider("Coupling Factor (0–1)", 0.5, 1.0, const.COUPLING, 0.01, key="coupling")

    st.divider()
    st.subheader("Geometry")
    st.slider("Wire width (nm)", 50, 150, 100, 1, key="width_nm")
    st.slider("Thickness (nm)", 4, 10, 7, 1, key="thickness_nm")
    st.slider("Fill factor", 0.30, 0.70, 0.50, 0.01, key="fill_factor")
    st.slider("Total meander length (μm)", 100.0, 20000.0, 8000.0, 100.0, key="length_um")

    st.divider()
    st.subheader("Readout")
    st.slider("Load resistance (Ω)", 25, 150, 50, 1, key="rload_ohm")

# Pull values from session_state
ib_over_ic = float(st.session_state["ib_over_ic"])
wavelength = float(st.session_state["lambda_nm"])
T = float(st.session_state["temp_K"])
absorption = float(st.session_state["absorption"])
coupling = float(st.session_state["coupling"])
width = float(st.session_state["width_nm"])
thickness = float(st.session_state["thickness_nm"])
fill = float(st.session_state["fill_factor"])
length = float(st.session_state["length_um"])
rload = float(st.session_state["rload_ohm"])

# If the model reads a global coupling, update it (kept from your original)
const.COUPLING = coupling

# ------------------ Model evaluations ------------------
lk_nH = kinetic_inductance_nH(length, width, thickness)
ide = ide_internal(ib_over_ic, wavelength, width)
sde = sde_system(ide, absorption)
dcr = dcr_hz(ib_over_ic, T, wavelength)

# Pulse grid & derived timing
t_ns = np.linspace(0, 20.0, 1000)
v, tau_ns, v_peak = pulse_waveform(t_ns, ib_over_ic, rload, lk_nH)
jitter_ps = jitter_fwhm_ps(v_peak)
risk_score, risk_level = latching_risk(ib_over_ic, rload, lk_nH)

# ------------------ Layout ------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Efficiency & Noise")
    m1, m2, m3 = st.columns(3)
    m1.metric("IDE (internal)", f"{ide*100:.1f}%")
    m2.metric("SDE (system)", f"{sde*100:.1f}%")
    m3.metric("DCR", f"{dcr:.1f} Hz")

    fig1, ax1 = plt.subplots()
    xs = np.linspace(0.5, 1.10, 120)
    ax1.plot(xs, [ide_internal(x, wavelength, width) for x in xs], label="IDE vs Ib/Ic")
    ax1.plot(xs, [sde_system(ide_internal(x, wavelength, width), absorption) for x in xs], label="SDE vs Ib/Ic")
    ax1.set_xlabel("Ib/Ic")
    ax1.set_ylabel("Efficiency")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("Pulse & Timing")
    m4, m5, m6 = st.columns(3)
    m4.metric("Kinetic Inductance", f"{lk_nH:.1f} nH")
    m5.metric("Decay constant τ", f"{tau_ns:.2f} ns")
    m6.metric("Jitter (FWHM)", f"{jitter_ps:.1f} ps")

    fig2, ax2 = plt.subplots()
    ax2.plot(t_ns, v)
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Pulse (arb. units)")
    st.pyplot(fig2)

st.divider()
st.subheader("Latching Heuristic")
st.write(f"**Risk: {risk_level}** (score = {risk_score:.2f}) — higher bias, lower $L_k$, and lower $R_\\mathrm{{load}}$ increase score.")
st.caption("Heuristic demo in a phenomenological toy model, not a microscopic NbTiN nanowire simulator.")

# Optional: quick note after applying a preset
if "_last_preset" in st.session_state:
    with st.expander("Preset notes", expanded=False):
        lp = st.session_state["_last_preset"]
        if lp == "Match-our-spec":
            st.write("Spec point: bias near $I_c$, high absorption/coupling → high SDE; τ≈L/R with 50 Ω.")
        elif lp == "High-throughput":
            st.write("Shorter τ via higher $R_{load}$ and smaller effective $L_k$; some efficiency trade.")
        elif lp == "Low-noise demo":
            st.write("Cooler + shorter λ → lower DCR; SDE drops since we’re not cranking bias.")

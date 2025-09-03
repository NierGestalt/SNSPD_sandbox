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

st.set_page_config(page_title="SNSPD Performance Sandbox", layout="wide")

st.title("SNSPD Performance Sandbox")
st.caption("Phenomenological slider driven demo to explore IDE/SDE, DCR, jitter, pulse shape, and latching heuristics.")

with st.sidebar:
    st.header("Controls")

    ib_over_ic = st.slider("Normalized Bias (Ib/Ic)", 0.0, 1.05, 0.85, 0.01)
    wavelength = st.slider("Wavelength (nm)", 800, 2200, 1550, 10)
    T = st.slider("Temperature (K)", 0.8, 10.0, 2.0, 0.1)
    absorption = st.slider("Optical Absorption (0–1)", 0.0, 1.0, 0.80, 0.01)
    coupling = st.slider("Coupling Factor (0–1)", 0.5, 1.0, const.COUPLING, 0.01)
    # Update coupling live
    const.COUPLING = float(coupling)

    st.divider()
    st.subheader("Geometry")
    width = st.slider("Wire width (nm)", 50, 150, 100, 1)
    thickness = st.slider("Thickness (nm)", 4, 10, 7, 1)
    fill = st.slider("Fill factor", 0.3, 0.7, 0.5, 0.01)
    length = st.slider("Total meander length (μm)", 100.0, 20000.0, 8000.0, 100.0)

    st.divider()
    st.subheader("Readout")
    rload = st.slider("Load resistance (Ω)", 25, 150, 50, 1)

col1, col2 = st.columns(2, gap="large")

# Compute models
lk_nH = kinetic_inductance_nH(length, width, thickness)
ide = ide_internal(ib_over_ic, wavelength, width)
sde = sde_system(ide, absorption)
dcr = dcr_hz(ib_over_ic, T, wavelength)
# Pulse grid
t_ns = np.linspace(0, 20.0, 1000)
v, tau_ns, v_peak = pulse_waveform(t_ns, ib_over_ic, rload, lk_nH)
jitter_ps = jitter_fwhm_ps(v_peak)
risk_score, risk_level = latching_risk(ib_over_ic, rload, lk_nH)

with col1:
    st.subheader("Efficiency & Noise")
    st.metric("IDE (internal)", f"{ide*100:.1f}%")
    st.metric("SDE (system)", f"{sde*100:.1f}%")
    st.metric("DCR", f"{dcr:.1f} Hz")

    fig1, ax1 = plt.subplots()
    xs = np.linspace(0.5, 1.05, 120)
    ax1.plot(xs, [ide_internal(x, wavelength, width) for x in xs], label="IDE vs Ib/Ic")
    ax1.plot(xs, [sde_system(ide_internal(x, wavelength, width), absorption) for x in xs], label="SDE vs Ib/Ic")
    ax1.set_xlabel("Ib/Ic")
    ax1.set_ylabel("Efficiency")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("Pulse & Timing")
    st.metric("Kinetic Inductance", f"{lk_nH:.1f} nH")
    st.metric("Decay constant τ", f"{tau_ns:.2f} ns")
    st.metric("Jitter (FWHM)", f"{jitter_ps:.1f} ps")

    fig2, ax2 = plt.subplots()
    ax2.plot(t_ns, v)
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Pulse (arb. units)")
    st.pyplot(fig2)

st.divider()
st.subheader("Latching Heuristic")
st.write(f"**Risk: {risk_level}** (score={risk_score:.2f}) — higher bias, lower Lk, and lower Rload increase score.")
st.caption("Heuristic only; for teaching/demo.")



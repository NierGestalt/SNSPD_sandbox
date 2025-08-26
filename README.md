# SNSPD Performance Sandbox (Streamlit)

A minimal, **phenomenological** simulator for Superconducting Nanowire Single‑Photon Detectors (SNSPDs).
Use sliders to explore how bias current, geometry, temperature, and optical absorption influence:
- **SDE / IDE**
- **Dark‑count rate (DCR)**
- **Timing jitter**
- **Pulse waveform (R–L readout)**
- **Latching risk (heuristic)**

> This is a *teaching/demo* tool. The models are intentionally simple; constants are tuned to produce reasonable shapes and orders of magnitude. Tweak `constants` inside `models.py` if you have lab‑specific values.

## Quickstart

```bash
# 1) create a fresh environment (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) run the app
streamlit run app.py
```

## Files
- **app.py** – Streamlit UI and plots
- **models.py** – Phenomenological models (IDE, DCR, SDE, Lk, pulses, jitter, latching)
- **tests.py** – Quick sanity tests (run: `python tests.py`)
- **requirements.txt** – Minimal dependencies

## Notes
- SDE = absorption × coupling × IDE
- Lk ∝ length / (width × thickness) with a tuned constant to land in the 10^2 nH ballpark
- Pulse decay time τ ≈ Lk / Rload
- Jitter combines amplitude‑slope and baseline contributions quadratically

## License
MIT – do whatever you want, but please cite your own lab’s assumptions when you present results.

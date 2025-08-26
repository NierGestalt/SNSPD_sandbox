import numpy as np
from models import (
    kinetic_inductance_nH,
    ide_internal,
    sde_system,
    dcr_hz,
    pulse_waveform,
    jitter_fwhm_ps,
)

def approx_monotone_increasing(f, xs):
    vals = [f(x) for x in xs]
    return all(vals[i] <= vals[i+1] + 1e-9 for i in range(len(vals)-1))

def test_lk_scaling():
    lk1 = kinetic_inductance_nH(8000, 100, 7)
    lk2 = kinetic_inductance_nH(16000, 100, 7)
    assert lk2 > lk1
    lk3 = kinetic_inductance_nH(8000, 80, 7)
    assert lk3 > lk1  # narrower → larger Lk

def test_ide_bias_monotonic():
    xs = np.linspace(0.6, 1.05, 20)
    assert approx_monotone_increasing(lambda x: ide_internal(x, 1550, 100), xs)

def test_sde_bounds():
    ide = 0.8
    sde = sde_system(ide, 0.9)
    assert 0 <= sde <= 1.0

def test_dcr_bias():
    low = dcr_hz(0.7, 2.0, 1550)
    high = dcr_hz(0.95, 2.0, 1550)
    assert high > low

def test_pulse_and_jitter():
    import numpy as np
    t = np.linspace(0, 20.0, 1000)
    v, tau_ns, v_peak = pulse_waveform(t, 0.9, 50, 300)
    assert v.max() > 0
    j = jitter_fwhm_ps(v_peak)
    assert j > 0

if __name__ == "__main__":
    # Run the tests without pytest
    test_lk_scaling()
    test_ide_bias_monotonic()
    test_sde_bounds()
    test_dcr_bias()
    test_pulse_and_jitter()
    print("All sanity tests passed.")

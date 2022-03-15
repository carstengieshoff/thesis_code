from typing import Optional, Union

import numpy as np
from scipy.fft import rfft
from scipy.signal import filtfilt


def spectral_envelope(signal: np.array, h: Optional[np.array] = None) -> np.array:
    """Calculates the spectral envelope of a signal.

    This follows sections 1 and 4 of `The Spectral Envelope and Its Applications (Stoffer, 1998)`, see
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.2369&rep=rep1&type=pdf.

    Args:
        signal: np.array of shape (len_signal, dim_signal) for which to calculate the spectral envelope.

    Returns:
        lam: np.array of teh spectral envelope at the fundamental frequencies.
    """
    z = signal / signal.std(axis=0)
    In = rfft(z, axis=0, norm="ortho")
    fz = np.array([In[i, :].reshape(-1, 1) @ In[i, :].reshape(1, -1) for i in range(In.shape[0])])
    if h is not None:
        h = h / h.sum()
        fz = filtfilt(h, 1, fz, axis=0)
    lam = np.array([np.abs(np.linalg.eig(f)[0]).max() for f in fz])

    return np.abs(lam)


def get_spectral_envelope_max(signal: np.array, h: Optional[np.array] = None) -> Union[int, np.array]:
    """Return index of maximal peak in spectral envelope.

    See 'spectral_envelope'.

    Args:
       signal: np.array of shape (len_signal, dim_signal) for which to the index.

    """
    lam = spectral_envelope(signal=signal, h=h)
    idx = np.argmax(lam)
    return idx.squeeze()


def idx_to_lag(idx: int, N: int) -> int:
    return int(np.ceil(N / idx))


def idx_to_freq(idx: int, N: int, FS: int) -> float:
    return FS * idx / N


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    from signals import Chirp, Sinusoid

    FS = 200
    sec = 1
    sin1 = Sinusoid(frequency=5, sampling_rate=FS, sec=sec, noise_rate=0.1)
    sin2 = Sinusoid(frequency=2, sampling_rate=FS, sec=sec, noise_rate=0.2)
    sin3 = Sinusoid(frequency=5, sampling_rate=FS, sec=sec, noise_rate=0.5)
    chirp = Chirp(frequency_start=1, frequency_end=2, sampling_rate=FS, sec=sec, noise_rate=0.1)

    sin1.generate()
    sin2.generate()
    sin3.generate()
    chirp.generate()

    data = np.vstack([sin1.data.T, sin2.data.T, sin3.data.T, chirp.data.T]).T

    env = spectral_envelope(signal=data, h=None)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(env)
    ax[0].set_title("Spectral Envelope")
    max_freq = get_spectral_envelope_max(data)
    ax[0].vlines(max_freq, ymin=0, ymax=1.2 * env.max(), label=f"max at {max_freq}", linestyle="dashed", color="black")
    ax[0].legend()

    data_f = np.abs(rfft(data, axis=0))
    ax[1].plot(data_f)
    ax[1].set_title("Individual Spectra")

    plt.show()

    N = data.shape[0]
    print(np.ceil(N / max_freq))
    print(max_freq / N * FS)

"""Windowing, mocap range / radial velocity / Doppler, and the radar STFT."""
import numpy as np
from scipy.signal import spectrogram


def make_sequences(data_dict, nperseg,  noverlap):
    step = nperseg -  noverlap
    out = {}
    for fname, data in data_dict.items():
        n_windows = (data.shape[0] -nperseg) // step + 1
        seqs = []
        for i in range(n_windows):
            start = i * step
            seqs.append(data[start:start+nperseg])
        out[fname] = np.array(seqs)
    return out

def compute_mocap_range(mocap_dict, centroid):
    """
    mocap_dict: dict[file_name -> np.array [T, N, 3]]
    centroid: np.array [1,1,3]
    return: dict[file_name -> np.array [T, N, 1]]
    """
    out = {}
    assert centroid.shape == (1,1,3), "centroid must be a 3D array"
    for fname, mocap_np in mocap_dict.items():
        out[fname] = np.linalg.norm(mocap_np - centroid, axis=2)[..., None]
    return out


def compute_mocap_vel(range_dict, fps):
    """
    range_dict: dict[file_name -> np.array [T, N, 1]]
    fps: mocap sampling rate
    return: dict[file_name -> np.array [T, N, 1]]
    """
    out = {}
    dt = 1.0 / fps
    for fname, r in range_dict.items():
        vel =  np.gradient(r, dt, axis=0) 
        out[fname] = vel
    return out

def estimate_doppler_frequency(vel_dict, wavelength):
    """
    vel_dict: dict[file_name -> np.array  [T, N, 1]]
    wavelength: radar wavelength (m)
    return: dict[file_name -> np.array [T, N, 1]]
    """
    out = {}
    for fname, vr in vel_dict.items():
        doppler = (2.0 * vr) / wavelength
        out[fname] = doppler
    return out


def iq_to_spectrogram(radar_dict, fs, nperseg, noverlap):
    """
    Computes the micro-Doppler spectrogram from IQ data.
    Returns magnitude spectrum in dB. Frequency bins are in raw FFT order
    (no fftshift); apply np.fft.fftshift at visualization time only.
    """
    sxx_dict, f_dict, t_dict = {}, {}, {}
    for fname, radar_series in radar_dict.items():
        iq_series = radar_series[:, 0] + 1j * radar_series[:, 1]
        f, t, Sxx_mag = spectrogram(
            iq_series,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend='constant',
            return_onesided=False,
            scaling='spectrum',
            mode='magnitude'
        )
        sxx_dict[fname] = (20 * np.log10(Sxx_mag + 1e-12)).T
        f_dict[fname] = f
        t_dict[fname] = t
    return f_dict, t_dict, sxx_dict

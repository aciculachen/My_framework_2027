"""Raw csv (data/raw/v3) -> data/processed.

    <split>_raw_full_sequences.pkl   {"mocap": {recording: [T, 53, 3]}}  radar-relative positions, m, 256 Hz
    <split>_final_data.pkl           radar_sxx_normalized [W, 256] and provenance (recording, window index)
    scalers.pkl                      StandardScalers for mocap and radar_sxx, fitted on the train split
    meta.pkl                         STFT frequency / time axes and window parameters
"""
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

from data import constants, paths, utils

SPLITS = ("train", "val", "eval")


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def load_data(file_list, data_path, startup_delay_mocap_s=0, startup_delay_radar_s=0, stop_read=None,
              col_name="Frame", radar_skiprows=0, num_markers=constants.N_MARKER,
              mocap_fps=constants.MC_SP_HZ, radar_fps=constants.RD_SP_HZ):
    """Radar: {recording: [T, 2]} I/Q; MoCap: {recording: [T, N, 3]} in m."""
    radar_dict, mocap_dict = {}, {}
    for f in file_list:
        radar_df = pd.read_csv(f"{data_path}/radar/{f}.csv", low_memory=False, skiprows=radar_skiprows)
        radar_np = radar_df.drop(radar_df.columns[0], axis=1).apply(pd.to_numeric, errors="coerce").to_numpy()
        radar_dict[f] = radar_np[_round_half_up(startup_delay_radar_s * radar_fps):]

        mocap_df = pd.read_csv(f"{data_path}/mocap/{f}.csv", low_memory=False, header=1)
        if stop_read and col_name in mocap_df.columns and stop_read in mocap_df[col_name].values:
            mocap_df = mocap_df.loc[:mocap_df.index[mocap_df[col_name] == stop_read][0] - 1]
        drop_cols = [c for c in [col_name, "Unnamed: 160"] if c in mocap_df.columns]
        coord_df = mocap_df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
        coords = coord_df.loc[coord_df.notna().sum(axis=1) == num_markers * 3].to_numpy()   # full rows only
        if coords.shape[1] != num_markers * 3:
            raise ValueError(f"Unexpected mocap coordinate width in {f}: {coords.shape[1]}")
        mocap_np = coords.reshape(-1, num_markers, 3) * 1e-3                                  # mm -> m
        mocap_dict[f] = mocap_np[_round_half_up(startup_delay_mocap_s * mocap_fps):]
    return radar_dict, mocap_dict


def trim_to_same_duration(mocap_dict, radar_dict, mc_sp, rd_sp=constants.RD_SP_HZ):
    for name in mocap_dict:
        dur = min(mocap_dict[name].shape[0] / mc_sp, radar_dict[name].shape[0] / rd_sp)
        mocap_dict[name] = mocap_dict[name][:min(mocap_dict[name].shape[0], int(np.floor(dur * mc_sp)))]
        radar_dict[name] = radar_dict[name][:min(radar_dict[name].shape[0], int(np.floor(dur * rd_sp)))]


def upsample(mocap_data, mc_sp, n_out, rd_sp=constants.RD_SP_HZ):
    """Linear interpolation of every marker coordinate onto the radar time grid."""
    T, N, D = mocap_data.shape
    t_in, t_out = np.arange(T) / mc_sp, np.arange(n_out) / rd_sp
    out = np.zeros((n_out, N, D))
    for n in range(N):
        for d in range(D):
            fill = (mocap_data[0, n, d], mocap_data[-1, n, d])
            out[:, n, d] = interp1d(t_in, mocap_data[:, n, d], kind="linear", bounds_error=False, fill_value=fill)(t_out)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    config = yaml.safe_load(open(paths.CONFIG_PATH))
    nperseg, noverlap = config["stft"]["nperseg"], config["stft"]["noverlap"]
    delay_s = config["delay"]["startup_us"] * 1e-6
    delay_radar_s = delay_s if config["delay"]["apply_to_radar"] else 0.0
    mc_sp = float(config["mocap_fps"])
    radar_origin = constants.RADAR_POS_M.mean(axis=0).reshape(1, 1, 3)
    out_dir = paths.PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mocap, radar = {s: {} for s in SPLITS}, {s: {} for s in SPLITS}
    for split, ds in config["datasets"].items():
        r, m = load_data(ds["file_list"], paths.RAW_DATA_DIR / "v3", delay_s, delay_radar_s,
                         mocap_fps=mc_sp, radar_fps=constants.RD_SP_HZ, **ds["loader_params"])
        radar[split].update(r)
        mocap[split].update(m)

    final, sxx_axes = {}, None
    for split in SPLITS:
        trim_to_same_duration(mocap[split], radar[split], mc_sp)
        for name in mocap[split]:                              # 256 Hz, positions relative to the radar
            mocap[split][name] = upsample(mocap[split][name], mc_sp, radar[split][name].shape[0]) - radar_origin
        with open(out_dir / f"{split}_raw_full_sequences.pkl", "wb") as f:
            pickle.dump({"mocap": dict(mocap[split])}, f)

        windows = utils.make_sequences(mocap[split], nperseg=nperseg, noverlap=noverlap)
        f_ax, t_ax, sxx = utils.iq_to_spectrogram(radar[split], fs=constants.RD_SP_HZ, nperseg=nperseg, noverlap=noverlap)
        sxx_axes = (f_ax, t_ax)
        final[split] = {"mocap": np.concatenate(list(windows.values())),
                        "radar_sxx": np.concatenate(list(sxx.values())),
                        "provenance": [{"filename": name, "original_idx": i}
                                       for name, w in windows.items() for i in range(w.shape[0])]}
        assert len(final[split]["radar_sxx"]) == len(final[split]["provenance"])
        logging.info(f"{split}: {len(final[split]['provenance'])} windows")

    train = final["train"]
    scalers = {"mocap": StandardScaler().fit(train["mocap"].reshape(-1, train["mocap"].shape[2] * train["mocap"].shape[3])),
               "radar_sxx": StandardScaler().fit(train["radar_sxx"].reshape(-1, 1))}
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    for split in SPLITS:
        sxx = final[split]["radar_sxx"]
        norm = scalers["radar_sxx"].transform(sxx.reshape(-1, 1)).reshape(sxx.shape).astype(sxx.dtype, copy=False)
        with open(out_dir / f"{split}_final_data.pkl", "wb") as f:
            pickle.dump({"radar_sxx_normalized": norm, "provenance": final[split]["provenance"]}, f)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"radar_f": sxx_axes[0], "radar_t": sxx_axes[1],
                     "window_params": {"nperseg": nperseg, "noverlap": noverlap, "hop_size": nperseg - noverlap}}, f)
    logging.info(f"saved {out_dir}")

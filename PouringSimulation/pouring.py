from ast import List
from re import S

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import optimize
from matplotlib.patches import Rectangle, ConnectionPatch

try:
    import seaborn as sns
    sns.set_style("dark")
except ImportError:
    pass

SIGNAL_COLOR = "steelblue"
TRIGGER_COLOR = "tomato"

def load_folder(folderpath):
    folder = Path(folderpath)
    if not folder.is_dir():
        raise ValueError(f"{folderpath} is not a valid directory.")
    
    data_files = list(folder.glob("*.csv"))
    if not data_files:
        raise ValueError(f"No CSV files found in {folderpath}.")
    
    all_data = []
    for file in data_files:
        time, aom_rise, trigger = load(file)
        all_data.append((time, aom_rise, trigger))
    
    return all_data

def load(filename):
    # Row 0: column names, row 1: units, row 2: empty — skip all three
    df = pd.read_csv(filename, delimiter=';', decimal=',', skiprows=[1, 2], na_values=['∞'])
    time = df["Time"].values.astype(float)
    aom_rise = df["Channel A"].values.astype(float)
    trigger_raw = np.array(pd.to_numeric(df["Channel B"], errors='coerce'), dtype=float)
    trigger = np.where(np.isnan(trigger_raw), np.inf, trigger_raw)
    # print(df.describe())
    return time, aom_rise, trigger

def plot_all(data_list):
    _ = plt.figure(figsize=(10, 6))
    for time, aom_rise, trigger in data_list:
        plt.plot(time, aom_rise, label="AOM Rise", color="blue", alpha=0.5)
        plt.plot(time, trigger, label="Trigger", color="orange", alpha=0.5)
    plt.xlabel("Time (us)")
    plt.ylabel("Signal (arb. units)")
    plt.title("AOM Rise and Trigger Over Time")
    plt.legend([])
    plt.show()

def plot(time, aom_rise, trigger):
    _ = plt.figure(figsize=(10, 6))
    plt.plot(time, aom_rise, label="AOM Rise", color=SIGNAL_COLOR)
    # plt.scatter(time, aom_rise, s=4, alpha=0.4, color=SIGNAL_COLOR, facecolors="none")
    plt.plot(time, trigger, label="Trigger", color=TRIGGER_COLOR)
    # plt.scatter(time, trigger, s=4, alpha=0.4, color=TRIGGER_COLOR, facecolors="none")
    plt.xlabel("Time (us)")
    plt.ylabel("Signal (arb. units)")
    plt.title("AOM Rise and Trigger Over Time")
    plt.legend()
    plt.show()

def flatten(data_list):
    flat_time = np.concatenate([time for time, _, _ in data_list])
    flat_aom_rise = np.concatenate([aom_rise for _, aom_rise, _ in data_list])
    flat_trigger = np.concatenate([trigger for _, _, trigger in data_list])

    # now sort for time and apply the same sorting to the rest
    sorted_indices = np.argsort(flat_time)
    flat_time = flat_time[sorted_indices]
    flat_aom_rise = flat_aom_rise[sorted_indices]
    flat_trigger = flat_trigger[sorted_indices]

    return flat_time, flat_aom_rise, flat_trigger

def normalize_all(data_list):
    normalized_data = []
    for time, aom_rise, trigger in data_list:
        norm_aom_rise = normalize(aom_rise)
        norm_trigger = normalize(trigger)
        normalized_data.append((time, norm_aom_rise, norm_trigger))
    return normalized_data

def normalize(data):
    mask = np.isfinite(data)
    revmask = ~mask
    maxval = np.nanmax(data[mask])
    data[revmask] = maxval
    minval = np.nanmin(data[mask])
    return (data - minval) / (maxval - minval)

def exp_func(t, A, tau, C):
        return A * (1 - np.exp(-t / tau)) + C

def fit_exponential(time, signal, guess=None):
    if guess is not None and type(guess) != str and len(guess) == 3:
        A_guess, tau_guess, C_guess = guess
    elif isinstance(guess, str):
        if guess.lower() == "rise":
            A_guess = -1
            tau_guess = 20
            C_guess = 1
        elif guess.lower() == "fall":
            A_guess = 1
            tau_guess = 80
            C_guess = 0
        else:
            raise ValueError("Invalid guess string. Use 'rise' or 'fall'.")
    else:
        A_guess = 1
        tau_guess = 80
        C_guess = 0

    popt, _ = optimize.curve_fit(exp_func, time, signal, p0=[A_guess, tau_guess, C_guess])
    return popt  # Returns the fitted parameters A, tau, and C

def clamp_saturated(signal, upper_quantile=0.95, lower_quantile=0.05):
    upper = np.quantile(signal, upper_quantile)
    lower = np.quantile(signal, lower_quantile)
    clamped = signal.copy()
    clamped[clamped >= upper] = 0.95
    clamped[clamped <= lower] = 0.0
    return clamped

def bin_signal(time, signal, n_bins=500):
    """Average signal into n_bins equally-spaced time bins."""
    edges = np.linspace(time[0], time[-1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(time, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    sums = np.bincount(idx, weights=signal, minlength=n_bins).astype(float)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    means = np.where(counts > 0, sums / counts, np.nan)
    # fill empty bins by nearest neighbour
    nans = np.isnan(means)
    if nans.any():
        xp = centers[~nans]
        fp = means[~nans]
        means[nans] = np.interp(centers[nans], xp, fp)
    return centers, means

def interpolate_signal(time, signal, t_query, edge="rise",
                       upper_quantile=0.95, lower_quantile=0.05,
                       n_bins=500):
    bin_time, bin_signal_avg = bin_signal(time, signal, n_bins=n_bins)
    clamped = clamp_saturated(bin_signal_avg, upper_quantile, lower_quantile)

    if edge == "rise":
        left_fill, right_fill = 0.0, 1.0
    elif edge == "fall":
        left_fill, right_fill = 1.0, 0.0
    else:
        raise ValueError("edge must be 'rise' or 'fall'")

    return np.interp(t_query, bin_time, clamped, left=left_fill, right=right_fill)

def moving_average(data, window_size):
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def lin_interp(x1,x2,t1,t2, t):
    if t1==t2:
        raise ValueError("t1 and t2 cannot be the same for interpolation.")
    elif t1 > t2:
        raise ValueError("t1 must be less than t2 for interpolation.")
    m = (x2 - x1) / (t2 - t1)
    c = x1 - m * t1
    return m * t + c

def interp_data(time, signal, t_query: np.ndarray):
    min_time, max_time = np.min(time), np.max(time)
    signal_query = np.zeros_like(t_query)
    for i, t in enumerate(t_query):
        if t <= min_time:
            signal_query[i] = signal[0]
        elif t >= max_time:
            signal_query[i] = signal[-1]
        else:
            idx_right = np.searchsorted(time, t, side='right')
            idx_left = idx_right - 1
            signal_query[i] = lin_interp(signal[idx_left], signal[idx_right],
                                        time[idx_left], time[idx_right], t)
    return signal_query


# Now: load TiSaph & Mephisto data, normalize, plot together and create an interpolated timeseries for each.
# Then write a function that can shift the time series to align them, based on the trigger signal!

def shift_time_series(time, dt, mode="p"):
    if mode == "p":
        return time + dt
    elif mode == "m":
        return time - dt
    else:
        raise ValueError("mode must be 'p' for positive shift or 'm' for negative shift.")
    
def trig_time(time, trigger, threshold=0.3, mode="rise"):
    """Can be used to find response|trigger time"""
    if mode == "rise":
        crossings = np.where((trigger[:-1] < threshold) & (trigger[1:] >= threshold))[0]
    elif mode == "fall":
        crossings = np.where((trigger[:-1] >= threshold) & (trigger[1:] < threshold))[0]
    else:
        raise ValueError("mode must be 'rise' or 'fall'.")
    if len(crossings) == 0:
        raise ValueError(f"No trigger crossing found (mode='{mode}', threshold={threshold}).")
    return time[crossings[0]]
    
def find_offset(time, signal, trigger, threshold=0.3, tmode="rise", smode="rise"):
    """Return the offset between signal response to trigger"""
    trig_t = trig_time(time, trigger, threshold, tmode)
    sign_t = trig_time(time, signal, threshold, smode)
    dt = sign_t - trig_t
    return dt

def plot_query_fit_full(ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt, smode="rise"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_query, interp_aom, label="AOM Rise", color=SIGNAL_COLOR)
    ax.plot(t_query_tc, exp_func(t_query_tc, *popt), label="Fit", color="tab:green")
    ax.plot(t_query_tc, interp_trig_tc, label="Trigger", color=TRIGGER_COLOR)
    trigger_time = trig_time(t_query_tc, interp_trig_tc, threshold=0.3, mode="rise")
    offs = find_offset(t_query, interp_aom, interp_trig, threshold=0.3, tmode="rise", smode=smode)
    ax.axvline(trigger_time, color="red", linestyle="--", label=f"Trigger Time: {trigger_time:.2f} us")
    ax.scatter([trigger_time, trigger_time + offs], [-0.02, -0.02], color="cyan", marker="|", label=f"Signal Time: {trigger_time + offs:.2f} us")
    ax.plot([trigger_time, trigger_time + offs], [-0.02, -0.02], color="cyan")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Signal (arb. units)")
    ax.set_title("AOM Rise and Trigger Over Time")
    ax.plot(ntime, naom, color="grey", alpha=0.3, label="Raw AOM Rise")
    ax.plot(ntime, ntrig, color="grey", alpha=0.3, label="Raw Trigger")
    ax.legend(loc="center left")
    
    zoom_x0, zoom_x1 = -2, 5
    _mask_aom  = (t_query    >= zoom_x0) & (t_query    <= zoom_x1)
    _mask_trig = (t_query_tc >= zoom_x0) & (t_query_tc <= zoom_x1)
    _ys = np.concatenate([
        interp_aom[_mask_aom],
        exp_func(t_query_tc[_mask_trig], *popt),
        interp_trig_tc[_mask_trig],
    ])
    zoom_y0 = _ys.min() - 0.05 * (_ys.max() - _ys.min())
    zoom_y1 = _ys.max() + 0.05 * (_ys.max() - _ys.min())

    axins = ax.inset_axes((0.62, 0.33, 0.36, 0.36))
    axins.plot(ntime, naom, color="grey", alpha=0.3)
    axins.plot(ntime, ntrig, color="grey", alpha=0.3)
    axins.plot(t_query, interp_aom, color=SIGNAL_COLOR)
    axins.plot(t_query_tc, exp_func(t_query_tc, *popt), color="tab:green")
    axins.plot(t_query_tc, interp_trig_tc, color=TRIGGER_COLOR)
    axins.axvline(trigger_time, color="red", linestyle="--")
    _mask_sc = (np.array([trigger_time, trigger_time + offs]) >= zoom_x0) & \
            (np.array([trigger_time, trigger_time + offs]) <= zoom_x1)
    axins.scatter(np.array([trigger_time, trigger_time + offs])[_mask_sc],
                np.array([-0.02, -0.02])[_mask_sc], color="cyan", marker="|")
    axins.plot([trigger_time, trigger_time + offs], [-0.02, -0.02], color="cyan")
    axins.set_xlim(zoom_x0, zoom_x1)
    axins.set_ylim(zoom_y0, zoom_y1)
    axins.set_xlabel("Time (us)", fontsize=7)
    axins.tick_params(labelsize=7)

    rect = Rectangle((zoom_x0, zoom_y0), zoom_x1 - zoom_x0, zoom_y1 - zoom_y0,
                    linewidth=1, edgecolor="gray", facecolor="none", linestyle="--", zorder=5)
    ax.add_patch(rect)
    for corner in [(zoom_x0, zoom_y0), (zoom_x1, zoom_y0), (zoom_x0, zoom_y1), (zoom_x1, zoom_y1)]:
        fig.add_artist(ConnectionPatch(
            xyA=corner, coordsA=ax.transData,
            xyB=corner, coordsB=axins.transData,
            color="gray", linestyle="--", linewidth=0.8,
        ))
    plt.tight_layout()
    plt.show()

def prepare(fpath, max_time=np.inf, guess="rise"):
    all_data = load_folder(fpath)
    flat_data = flatten(all_data)
    
    # if a max time is given, filter out data beyond that time:
    if np.isfinite(max_time):
        mask = flat_data[0] <= max_time
        flat_data = tuple(arr[mask] for arr in flat_data)

    ntime, naom, ntrig = flat_data[0], normalize(flat_data[1]), normalize(flat_data[2])
    trigger_threshold = 0.15
    trigger_crossings = np.where((ntrig[:-1] < trigger_threshold) & (ntrig[1:] >= trigger_threshold))[0]
    ntime_tc = ntime[trigger_crossings[0]:]
    naom_tc = naom[trigger_crossings[0]:]
    ntrig_tc = ntrig[trigger_crossings[0]:]
    popt = fit_exponential(ntime_tc, naom_tc, guess=guess)  # A=-1, C=1 rise | A=1, C=0 decay
    print(f"Fitted parameters: A={popt[0]:.3f}, tau={popt[1]:.3f} us, C={popt[2]:.3f}")
    ntime, naom, ntrig = flat_data[0], normalize(flat_data[1]), normalize(flat_data[2])
    t_query = np.linspace(ntime[0], ntime[-1], 5000)
    interp_aom = interp_data(ntime, naom, t_query)
    interp_trig = interp_data(ntime, ntrig, t_query)
    t_query_tc = np.linspace(ntime_tc[0], ntime_tc[-1], 1000)
    interp_aom_tc = interp_data(ntime_tc, naom_tc, t_query_tc)
    interp_trig_tc = interp_data(ntime_tc, ntrig_tc, t_query_tc)
    return ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt


def main():

    folderpath = Path(__file__).parent / "data" / "TiSaph"
    ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt = prepare(folderpath, max_time=3, guess="fall")
    plot_query_fit_full(ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt, smode="fall")

    # folderpath = Path(__file__).parent / "data" / "Mephisto"
    # ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt = prepare(folderpath, guess="fall")
    # plot_query_fit_full(ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt)

    # folderpath = Path(__file__).parent / "data" / "TiSaph1"
    # ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt = prepare(folderpath, guess="rise")
    # plot_query_fit_full(ntime, naom, ntrig, t_query, interp_aom, interp_trig, t_query_tc, interp_aom_tc, interp_trig_tc, popt, smode="rise")

if __name__ == "__main__":
    main()


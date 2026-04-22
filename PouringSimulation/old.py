import numpy as np
kB = 1.380649e-23              # Boltzmann constant [J/K]
hbar = 1.054571817e-34         # Reduced Planck constant [J·s]
eV_to_J = 1.602176634e-19      # eV to Joules conversion
m = 2.273347e-25               # Mass of Cs-137: 136.907 Da [kg]
w0_static_um = 1.4             # Static tweezer waist [μm]
w0_dynamic_um = 1.2            # Dynamic tweezer waist [μm]
w0_um = 1.4                    # Reference waist (static) [μm]
wavelength_dynamic_nm = 933.0  # Auxiliary trap wavelength [nm]
wavelength_static_nm = 936.0   # Static trap wavelength [nm]
Gamma = 2*np.pi * 5.22e6            # Natural linewidth of Cs D2 transition [rad/s] ?is this right?
Delta_dynamic = -2 * np.pi * 30.4 * 1.0e12      # Assuming 933nm light: 30.40 THz detuning [rad/s]
Delta_static = -2 * np.pi * 31.43 * 1.0e12      # Assuming 936nm light: 31.43 THz detuning [rad/s]
typical_distance_um = 4.6      # Typical distance between traps [μm]

w0_um = w0_um
w0_SI = w0_um * 1e-6
U0_J = kB * 287e-6  # Convert μK to Joules
ω0 = np.sqrt(4 * U0_J / (m * w0_SI**2))
t0 = 1 / ω0

print(t0)
print("="   *50)


# Script to analyze the rising edges and falling edges of the AOM's
import seaborn as sns
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf

FIT =  False

FPATH = Path("C:\\Users\\bjarn\\OneDrive\\Dokumente\\UCPH\\MasterThesis\\AOM_Flank\\20260319_TiSaph1")
FPATH2 = Path("C:\\Users\\bjarn\\OneDrive\\Dokumente\\UCPH\\MasterThesis\\AOM_Flank\\20260319_Mephisto")
FPATH3 = Path("C:\\Users\\bjarn\\OneDrive\\Dokumente\\UCPH\\MasterThesis\\AOM_Flank\\20260319_combined")
FPATH4 = Path("C:\\Users\\bjarn\\OneDrive\\Dokumente\\UCPH\\MasterThesis\\AOM_Flank\\20260319_bkg")
folder_paths = [FPATH,FPATH2,FPATH3,FPATH4]

def read_csv_to_df(filepath, skiprows=0):
    df = pd.read_csv(filepath, delimiter=';', skiprows=skiprows, decimal=',')
    # Manually convert Channel B if it's still a string
    if df['Channel B'].dtype == 'object' or df['Channel B'].dtype == 'str':
        # Replace infinity symbol with 5.0, then replace commas with periods and convert to numeric
        df['Channel B'] = df['Channel B'].str.replace('∞', '5.0').str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
    return df

def get_all_files_in_folder(path: Path):
    return list(path.glob("*.csv"))

def sigmoid(x, a, b, c, d):
    """Sigmoid function for fitting rising/falling edges"""
    return a / (1 + np.exp(-b * (x - c))) + d

def error_function(x, a, b, c, d):
    """Error function for fitting rising/falling edges"""
    return a * erf(b * (x - c)) + d

def fit_channel(time, channel_data):
    """Fit a channel's data with both sigmoid and error function"""
    # Sigmoid fit
    try:
        p0_sigmoid = [np.max(channel_data) - np.min(channel_data), 0.1, np.median(time), np.min(channel_data)]
        popt_sigmoid, _ = curve_fit(sigmoid, time, channel_data, p0=p0_sigmoid, maxfev=10000)
        fit_sigmoid = sigmoid(time, *popt_sigmoid)
    except Exception as e:
        print(f"Sigmoid fit failed: {e}")
        fit_sigmoid = channel_data
    
    # Error function fit
    try:
        p0_erf = [(np.max(channel_data) - np.min(channel_data))/2, 0.1, np.median(time), np.mean(channel_data)]
        popt_erf, _ = curve_fit(error_function, time, channel_data, p0=p0_erf, maxfev=10000)
        fit_erf = error_function(time, *popt_erf)
    except Exception as e:
        print(f"Error function fit failed: {e}")
        fit_erf = channel_data
    
    return fit_sigmoid, fit_erf

files = get_all_files_in_folder(FPATH)
print(f"Found {len(files)} CSV files in {FPATH}")

# First pass: read all files and determine common time grid
list_of_dfs = []
all_times = []

print("Reading files and collecting time points...")
for file in tqdm(files):
    df = read_csv_to_df(file, skiprows=[1])
    list_of_dfs.append(df)
    all_times.extend(df['Time'].values)

# Create common time grid using the full range with uniform spacing
common_time = np.linspace(min(all_times), max(all_times), num=3000)
print(f"Common time grid: {len(common_time)} points from {common_time[0]:.2f} to {common_time[-1]:.2f} µs")  # FIXED: ms -> µs

# Second pass: interpolate all data to common time grid and fit
interpolated_dfs = []
fitted_channel_a_sigmoid = []
fitted_channel_b_sigmoid = []
fitted_channel_a_erf = []
fitted_channel_b_erf = []

print("\nInterpolating and fitting data...")
for df in tqdm(list_of_dfs):
    # Interpolate to common time grid
    interp_a = interp1d(df['Time'], df['Channel A'], kind='linear', fill_value='extrapolate')
    interp_b = interp1d(df['Time'], df['Channel B'], kind='linear', fill_value='extrapolate')
    
    channel_a_interp = interp_a(common_time)
    channel_b_interp = interp_b(common_time)
    
    # Create interpolated dataframe
    df_interp = pd.DataFrame({
        'Time': common_time,
        'Channel A': channel_a_interp,
        'Channel B': channel_b_interp
    })
    interpolated_dfs.append(df_interp)
    
    # Fit the data with both sigmoid and error function (only if FIT is True)
    if FIT:
        fit_a_sigmoid, fit_a_erf = fit_channel(common_time, channel_a_interp)
        fit_b_sigmoid, fit_b_erf = fit_channel(common_time, channel_b_interp)
        
        fitted_channel_a_sigmoid.append(fit_a_sigmoid)
        fitted_channel_b_sigmoid.append(fit_b_sigmoid)
        fitted_channel_a_erf.append(fit_a_erf)
        fitted_channel_b_erf.append(fit_b_erf)

# Compute averages
all_channel_a = np.array([df['Channel A'].values for df in interpolated_dfs])
all_channel_b = np.array([df['Channel B'].values for df in interpolated_dfs])

average_raw = pd.DataFrame({
    'Time': common_time,
    'Channel A': np.mean(all_channel_a, axis=0),
    'Channel B': np.mean(all_channel_b, axis=0)
})

if FIT:
    average_fitted_sigmoid = pd.DataFrame({
        'Time': common_time,
        'Channel A': np.mean(fitted_channel_a_sigmoid, axis=0),
        'Channel B': np.mean(fitted_channel_b_sigmoid, axis=0)
    })

    average_fitted_erf = pd.DataFrame({
        'Time': common_time,
        'Channel A': np.mean(fitted_channel_a_erf, axis=0),
        'Channel B': np.mean(fitted_channel_b_erf, axis=0)
    })

print("\nRaw data statistics:")
print(average_raw.describe())

# Plot the average
sns.set_style("dark")
mpl.rcParams.update(  
    {
        "xtick.color": "black",
        "ytick.color": "black",
    }
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Channel A - Plot all individual runs first (in background)
for i, df in enumerate(interpolated_dfs):
    ax1.plot(df['Time'], df['Channel A'], '-', linewidth=0.5, alpha=0.2, color='gray')
if FIT:
    for i in range(len(fitted_channel_a_sigmoid)):
        ax1.plot(common_time, fitted_channel_a_sigmoid[i], '-', linewidth=0.5, alpha=0.2, color='blue')
        ax1.plot(common_time, fitted_channel_a_erf[i], '-', linewidth=0.5, alpha=0.2, color='orange')

# Plot averages on top
ax1.plot(average_raw['Time'], average_raw['Channel A'], '-', linewidth=2.5, color='black', label='Mean Raw Data')
if FIT:
    ax1.plot(average_fitted_sigmoid['Time'], average_fitted_sigmoid['Channel A'], '-', linewidth=2.5, color='darkblue', label='Mean Sigmoid Fit')
    ax1.plot(average_fitted_erf['Time'], average_fitted_erf['Channel A'], '-', linewidth=2.5, color='darkorange', label='Mean Error Function Fit')
ax1.set_ylabel('Voltage (V)')
ax1.set_title('Channel A - Average of Rising/Falling Edges')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Channel B - Plot all individual runs first (in background)
for i, df in enumerate(interpolated_dfs):
    ax2.plot(df['Time'], df['Channel B'], '-', linewidth=0.5, alpha=0.2, color='gray')
if FIT:
    for i in range(len(fitted_channel_b_sigmoid)):
        ax2.plot(common_time, fitted_channel_b_sigmoid[i], '-', linewidth=0.5, alpha=0.2, color='blue')
        # ax2.plot(common_time, fitted_channel_b_erf[i], '-', linewidth=0.5, alpha=0.2, color='orange')

# Plot averages on top
ax2.plot(average_raw['Time'], average_raw['Channel B'], '-', linewidth=2.5, color='black', label='Mean Raw Data')
if FIT:
    ax2.plot(average_fitted_sigmoid['Time'], average_fitted_sigmoid['Channel B'], '-', linewidth=2.5, color='darkblue', label='Mean Sigmoid Fit')
    # ax2.plot(average_fitted_erf['Time'], average_fitted_erf['Channel B'], '--', linewidth=2, label='Mean Error Function Fit')
ax2.set_xlabel('Time (µs)')       # FIXED: ms -> µs
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Channel B - Average of Rising/Falling Edges')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""
SWMF Visualization Library
===========================

A reusable Python module for comparing SWMF (Space Weather Modeling Framework)
simulation results against observational data.

Features:
- Parse SWMF PARAM.in metadata
- Read simulation outputs (satellite trajectories, synthetic images)
- Read observational data (in-situ, AIA images)
- Generate comparison plots for in-situ and remote sensing diagnostics

Author: SWMF Visualization Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os
from datetime import datetime
from typing import Optional, Tuple, Dict

# ============================================================
# PHYSICAL CONSTANTS (CGS)
# ============================================================
PROTON_MASS_CGS = 1.6726e-24  # g
BOLTZMANN_CGS = 1.3807e-16    # erg/K
GAUSS_TO_NT = 1.0e5           # 1 Gauss = 1e5 nT

# ============================================================
# AIA COLORMAP CONFIGURATION
# ============================================================
AIA_CMAPS = {
    '94': 'sdoaia94' if 'sdoaia94' in plt.colormaps() else 'inferno',
    '131': 'sdoaia131' if 'sdoaia131' in plt.colormaps() else 'inferno',
    '171': 'sdoaia171' if 'sdoaia171' in plt.colormaps() else 'Greens_r',
    '193': 'sdoaia193' if 'sdoaia193' in plt.colormaps() else 'copper',
    '211': 'sdoaia211' if 'sdoaia211' in plt.colormaps() else 'BuPu_r',
    '304': 'sdoaia304' if 'sdoaia304' in plt.colormaps() else 'Reds_r',
    '335': 'sdoaia335' if 'sdoaia335' in plt.colormaps() else 'Blues_r',
}


# ============================================================
# METADATA PARSING
# ============================================================

def parse_start_time(param_file: str) -> datetime:
    """
    Parse #STARTTIME block from a SWMF PARAM.in file.

    Parameters
    ----------
    param_file : str
        Path to the PARAM.in file.

    Returns
    -------
    datetime
        Event start time extracted from the file.

    Raises
    ------
    FileNotFoundError
        If PARAM.in does not exist.
    ValueError
        If #STARTTIME block is not found.
    """
    if not os.path.isfile(param_file):
        raise FileNotFoundError(f"PARAM.in not found: {param_file}")

    with open(param_file, 'r') as f:
        lines = f.readlines()

    # Find #STARTTIME line
    idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('#STARTTIME'):
            idx = i
            break
    if idx is None:
        raise ValueError("#STARTTIME block not found in PARAM.in")

    # Parse 7 fields: iYear, iMonth, iDay, iHour, iMinute, iSecond, FracSecond
    fields = {}
    keys = ['iYear', 'iMonth', 'iDay', 'iHour', 'iMinute', 'iSecond', 'FracSecond']
    for j, key in enumerate(keys):
        val_str = lines[idx + 1 + j].split()[0]
        fields[key] = int(float(val_str))

    event_time = datetime(
        fields['iYear'], fields['iMonth'], fields['iDay'],
        fields['iHour'], fields['iMinute'], fields['iSecond']
    )
    return event_time


# ============================================================
# FILE DISCOVERY UTILITIES
# ============================================================

def find_obs_file(obs_dir: str, prefix: str, event_time: datetime) -> Optional[str]:
    """
    Locate the observation file matching the event time.

    Searches for files like: <prefix>_YYYY_MM_DDThh_mm_ss.out

    Parameters
    ----------
    obs_dir : str
        Directory containing observation files.
    prefix : str
        File prefix (e.g., 'omni', 'sta').
    event_time : datetime
        Event start time.

    Returns
    -------
    str or None
        Path to the observation file, or None if not found.
    """
    tag = event_time.strftime('%Y_%m_%dT%H_%M_%S')
    pattern = os.path.join(obs_dir, f'{prefix}_{tag}.out')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Fallback: match by date only
    tag_date = event_time.strftime('%Y_%m_%d')
    pattern2 = os.path.join(obs_dir, f'{prefix}_{tag_date}*.out')
    matches2 = glob.glob(pattern2)
    if matches2:
        return sorted(matches2)[0]

    print(f"  [WARNING] No observation file found for prefix='{prefix}', date={tag}")
    return None


def find_sat_file(ih_dir: str, spacecraft: str) -> Optional[str]:
    """
    Find the satellite trajectory file for a given spacecraft in IH/.

    Parameters
    ----------
    ih_dir : str
        Path to the IH directory.
    spacecraft : str
        Spacecraft name (e.g., 'earth', 'sta').

    Returns
    -------
    str or None
        Path to the .sat file, or None if not found.
    """
    pattern = os.path.join(ih_dir, f'trj_{spacecraft}_*.sat')
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[-1]  # Take latest iteration
    print(f"  [WARNING] No SAT file found for {spacecraft} in {ih_dir}")
    return None


def find_los_file(sc_dir: str, instrument: str = 'sdo_aia') -> Optional[str]:
    """
    Find the latest LOS output file in SC/.

    Parameters
    ----------
    sc_dir : str
        Path to the SC directory.
    instrument : str, optional
        Instrument name (default: 'sdo_aia').

    Returns
    -------
    str or None
        Path to the LOS .out file, or None if not found.
    """
    pattern = os.path.join(sc_dir, f'los_{instrument}_*.out')
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[-1]
    print(f"  [WARNING] No LOS file found for {instrument} in {sc_dir}")
    return None


def find_aia_fits(obs_dir: str, event_time: datetime, channel: str = '193') -> Optional[str]:
    """
    Find AIA FITS file matching event date and channel.

    Parameters
    ----------
    obs_dir : str
        Directory containing AIA FITS files.
    event_time : datetime
        Event start time.
    channel : str, optional
        AIA wavelength channel (default: '193').

    Returns
    -------
    str or None
        Path to the FITS file, or None if not found.
    """
    date_str = event_time.strftime('%Y-%m-%d')
    pattern = os.path.join(obs_dir, f'aia.lev1.{channel}A_{date_str}*.fits')
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[0]
    print(f"  [WARNING] No AIA FITS for channel {channel}A on {date_str}")
    return None


# ============================================================
# DATA READING — IN-SITU
# ============================================================

def read_sat_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    Read an SWMF IH satellite (.sat) file.

    Computes physical quantities:
    - Radial velocity (ur) in km/s
    - Number density (ndens) in cm^-3
    - Ion temperature (ti) and electron temperature (te) in K
    - Total magnetic field magnitude (bmag) in nT (including Alfvén wave energy)

    Parameters
    ----------
    filepath : str
        Path to the .sat file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: date, ur, ndens, ti, te, bmag.
        Returns None if file not found.
    """
    if not os.path.isfile(filepath):
        print(f"  [WARNING] SAT file not found: {filepath}")
        return None

    df = pd.read_csv(filepath, skiprows=1, sep=r'\s+')
    df.columns = df.columns.str.lower()

    # Construct datetime
    df['date'] = pd.to_datetime(
        df[['year', 'mo', 'dy', 'hr', 'mn', 'sc']].astype(int).astype(str).apply(' '.join, axis=1),
        format='%Y %m %d %H %M %S'
    )

    # Radial velocity (km/s)
    r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['ur'] = (df['ux']*df['x'] + df['uy']*df['y'] + df['uz']*df['z']) / r

    # Number density (cm^-3) — rho is in g/cm^3
    df['ndens'] = df['rho'] / PROTON_MASS_CGS

    # Temperature (K) — T = p * m_p / (rho * k_B)
    df['ti'] = df['p'] * PROTON_MASS_CGS / (df['rho'] * BOLTZMANN_CGS)
    df['te'] = df['pe'] * PROTON_MASS_CGS / (df['rho'] * BOLTZMANN_CGS)

    # Total magnetic field (nT) including Alfvén wave energy
    df['bmag'] = np.sqrt(df['bx']**2 + df['by']**2 + df['bz']**2
                         + 4.0 * np.pi * (df['i01'] + df['i02'])) * GAUSS_TO_NT

    return df


def read_obs_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    Read an SWMF-formatted observation file (omni_*.out or sta_*.out).

    Parameters
    ----------
    filepath : str
        Path to the observation file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: date, V_tot, Rho, Temperature, B_tot, Br.
        Returns None if file not found.
    """
    if not os.path.isfile(filepath):
        print(f"  [WARNING] Observation file not found: {filepath}")
        return None

    df = pd.read_csv(filepath, skiprows=3, sep=r'\s+')

    # Construct datetime
    df['date'] = pd.to_datetime(
        df[['year', 'mo', 'dy', 'hr', 'mn', 'sc']].astype(int).astype(str).apply(' '.join, axis=1),
        format='%Y %m %d %H %M %S'
    )

    # Clean bad/fill values (commonly large negative or 9999-type)
    for col in ['Rho', 'V_tot', 'Temperature', 'B_tot', 'Br']:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
            df.loc[df[col] > 1e6, col] = np.nan

    return df


# ============================================================
# DATA READING — REMOTE SENSING
# ============================================================

def log_transform(image: np.ndarray, vmin: float = 0.1) -> np.ndarray:
    """
    Apply logarithmic transform to image data.

    Handles negative values and zero by clipping to a minimum threshold.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    vmin : float, optional
        Minimum value for clipping (default: 0.1).

    Returns
    -------
    np.ndarray
        Log10-transformed image.
    """
    return np.log10(np.clip(image, vmin, None))


def read_los_file(filepath: str, channel: str = '193') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read an SWMF synthetic LOS file (e.g., los_sdo_aia_*.out).

    Parameters
    ----------
    filepath : str
        Path to the LOS .out file.
    channel : str, optional
        AIA channel to extract (e.g., '193', '171').

    Returns
    -------
    x_2d, y_2d, img_2d : (ndarray, ndarray, ndarray) or (None, None, None)
        2D coordinate arrays (in solar radii) and synthetic image.
    """
    if not os.path.isfile(filepath):
        print(f"  [WARNING] LOS file not found: {filepath}")
        return None, None, None

    # Parse header (5 lines)
    with open(filepath, 'r') as f:
        header1 = f.readline()
        header2 = f.readline().split()
        dims = f.readline().split()
        obspos = f.readline().split()
        colnames = f.readline().split()

    nx, ny = int(dims[0]), int(dims[1])
    nheader = 5

    # Read data
    data = np.loadtxt(filepath, skiprows=nheader)

    # Map column name
    col_target = f'AIA:{channel}'
    if col_target not in colnames:
        print(f"  [WARNING] Channel '{col_target}' not found. Available: {colnames}")
        return None, None, None

    col_idx = colnames.index(col_target)
    x_1d = data[:, 0]
    y_1d = data[:, 1]
    img_1d = data[:, col_idx]

    # Reshape to 2D
    x_2d = x_1d.reshape(ny, nx)
    y_2d = y_1d.reshape(ny, nx)
    img_2d = img_1d.reshape(ny, nx)

    return x_2d, y_2d, img_2d


def read_aia_obs_file(filepath: str, channel: str = '193') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read the processed AIA observation file (AIA_Observations_*.out).

    Parameters
    ----------
    filepath : str
        Path to the AIA observation .out file.
    channel : str, optional
        AIA channel to extract (default: '193').

    Returns
    -------
    x_2d, y_2d, img_2d : (ndarray, ndarray, ndarray) or (None, None, None)
        2D coordinate arrays (in solar radii) and observed image (linear intensity).
    """
    if not os.path.isfile(filepath):
        print(f"  [WARNING] AIA obs file not found: {filepath}")
        return None, None, None

    # Parse header (4 lines)
    with open(filepath, 'r') as f:
        header1 = f.readline()
        header2 = f.readline().split()
        dims = f.readline().split()
        colnames = f.readline().split()

    nx, ny = int(dims[0]), int(dims[1])
    nheader = 4

    # Read data
    data = np.loadtxt(filepath, skiprows=nheader)

    # Map column name
    col_target = f'AIA:{channel}'
    if col_target not in colnames:
        print(f"  [WARNING] Channel '{col_target}' not in obs file. Available: {colnames}")
        return None, None, None

    col_idx = colnames.index(col_target)
    x_2d = data[:, 0].reshape(ny, nx)
    y_2d = data[:, 1].reshape(ny, nx)
    img_2d = data[:, col_idx].reshape(ny, nx)

    return x_2d, y_2d, img_2d


def read_aia_fits(filepath: str) -> Optional[np.ndarray]:
    """
    Read an SDO/AIA FITS file and return the image data.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.

    Returns
    -------
    np.ndarray or None
        2D image array (DN or DN/s), or None if file not found or astropy unavailable.
    """
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        print("  [WARNING] astropy not installed — cannot read FITS files.")
        return None

    if not os.path.isfile(filepath):
        print(f"  [WARNING] FITS file not found: {filepath}")
        return None

    with pyfits.open(filepath) as hdul:
        img = hdul[1].data if len(hdul) > 1 else hdul[0].data

    return img.astype(float)


# ============================================================
# BINARY SWMF 2D SLICE READING
# ============================================================

def read_swmf_binary_2d(filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Read a binary SWMF 2D slice file (Fortran unformatted, IDL format).
    
    Typical file: z=0_var_3_n00080000.out
    
    File structure:
    1. Title record (string)
    2. Header record: it, time, ndim, n_var, n_gen, nx, ny, nz
    3. Variable names record (space-separated string)
    4. Coordinate arrays: X, Y, Z (even if 2D, Z exists but nz=1)
    5. Data arrays for each variable
    
    Parameters
    ----------
    filepath : str
        Path to the binary SWMF output file.
    
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'X': 2D array of X coordinates [Rsun]
        - 'Y': 2D array of Y coordinates [Rsun]
        - 'var_name': 2D array for each variable
        - '_metadata': dict with file header info
        Returns None if file not found or read fails.
    
    Notes
    -----
    Fortran unformatted files use "record markers": 4-byte integers before
    and after each logical record. These must be read but can be ignored.
    """
    if not os.path.isfile(filepath):
        print(f"  [WARNING] Binary file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            # Helper to read a Fortran record
            def read_fortran_record(f, dtype, count=None):
                """Read one Fortran unformatted record."""
                # Read leading record marker
                lead = np.fromfile(f, dtype=np.int32, count=1)
                if len(lead) == 0:
                    return None
                record_size = lead[0]
                
                # Read data
                if dtype == 'S1':  # String data
                    data = np.fromfile(f, dtype='S1', count=record_size)
                    result = b''.join(data).decode('ascii', errors='ignore').strip()
                else:
                    if count is None:
                        count = record_size // np.dtype(dtype).itemsize
                    data = np.fromfile(f, dtype=dtype, count=count)
                    result = data
                
                # Read trailing record marker
                trail = np.fromfile(f, dtype=np.int32, count=1)
                if len(trail) == 0 or trail[0] != record_size:
                    print(f"  [WARNING] Record marker mismatch: {lead[0]} vs {trail[0] if len(trail) > 0 else 'EOF'}")
                
                return result
            
            # 1. Read title
            title = read_fortran_record(f, 'S1')
            if title is None:
                print(f"  [ERROR] Failed to read title from {filepath}")
                return None
            
            # 2. Read header
            header = read_fortran_record(f, np.int32, count=2)  # it, ndim
            if header is None or len(header) < 2:
                print(f"  [ERROR] Failed to read header from {filepath}")
                return None
            
            it = header[0]
            ndim = header[1]
            
            # Read more header data (depends on file format version)
            # Read the time and remaining integers
            header2 = read_fortran_record(f, np.float32, count=1)  # time
            time_val = header2[0] if header2 is not None else 0.0
            
            # Read n_var and dimensions
            header3 = read_fortran_record(f, np.int32)
            if header3 is None or len(header3) < 3:
                print(f"  [ERROR] Failed to read dimension info from {filepath}")
                return None
            
            n_var = header3[0]
            nx = header3[1]
            ny = header3[2]
            nz = header3[3] if len(header3) > 3 else 1
            
            # 3. Read variable names
            varnames_str = read_fortran_record(f, 'S1')
            if varnames_str is None:
                print(f"  [ERROR] Failed to read variable names from {filepath}")
                return None
            
            var_names = varnames_str.split()
            
            # Verify we have the right number of variables
            if len(var_names) != n_var + ndim:
                print(f"  [WARNING] Variable count mismatch: expected {n_var + ndim}, got {len(var_names)}")
            
            # 4. Read coordinate arrays
            x_1d = read_fortran_record(f, np.float32, count=nx)
            y_1d = read_fortran_record(f, np.float32, count=ny)
            
            if nz > 1:
                z_1d = read_fortran_record(f, np.float32, count=nz)
            else:
                z_1d = np.array([0.0])
            
            # 5. Read data arrays for each variable
            data_dict = {}
            
            # Store metadata
            data_dict['_metadata'] = {
                'title': title,
                'it': it,
                'time': time_val,
                'ndim': ndim,
                'n_var': n_var,
                'nx': nx,
                'ny': ny,
                'nz': nz,
                'var_names': var_names
            }
            
            # Read all variables
            coord_names = var_names[:ndim]
            physics_names = var_names[ndim:]
            
            for var_name in physics_names:
                if nz == 1:
                    # 2D slice
                    raw_data = read_fortran_record(f, np.float32, count=nx * ny)
                    if raw_data is None or len(raw_data) != nx * ny:
                        print(f"  [WARNING] Failed to read complete data for {var_name}")
                        continue
                    # Reshape to (ny, nx) for proper orientation
                    data_2d = raw_data.reshape((ny, nx))
                else:
                    # 3D data (not typical for z=0 slices but handle it)
                    raw_data = read_fortran_record(f, np.float32, count=nx * ny * nz)
                    if raw_data is None:
                        continue
                    data_2d = raw_data.reshape((nz, ny, nx))[0, :, :]  # Take first z-slice
                
                data_dict[var_name] = data_2d
            
            # Create 2D coordinate meshgrids
            X, Y = np.meshgrid(x_1d, y_1d)
            data_dict['X'] = X
            data_dict['Y'] = Y
            
            return data_dict
            
    except Exception as e:
        print(f"  [ERROR] Exception reading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_swmf_2d_slice(
    data: Dict[str, np.ndarray],
    var_name: str,
    log_scale: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    overlay_solar_surface: bool = True,
    figsize: Tuple[float, float] = (10, 9)
) -> Optional[plt.Figure]:
    """
    Visualize a 2D slice from SWMF binary output.
    
    Parameters
    ----------
    data : dict
        Dictionary returned by read_swmf_binary_2d().
    var_name : str
        Name of the variable to plot (e.g., 'Rho', 'T', 'Ux').
    log_scale : bool, optional
        Apply logarithmic color scale (default: True).
    vmin, vmax : float, optional
        Color scale limits. If None, uses percentiles.
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis').
    title : str, optional
        Plot title. If None, uses variable name.
    overlay_solar_surface : bool, optional
        Draw white circle at r=1.0 Rsun (default: True).
    figsize : tuple, optional
        Figure size (default: (10, 9)).
    
    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure, or None if data invalid.
    """
    if data is None:
        print("  [ERROR] Cannot plot: data is None")
        return None
    
    if var_name not in data:
        print(f"  [ERROR] Variable '{var_name}' not found in data.")
        print(f"  Available variables: {[k for k in data.keys() if k != '_metadata' and k not in ['X', 'Y']]}")
        return None
    
    X = data['X']
    Y = data['Y']
    Z = data[var_name]
    
    # Apply log scale if requested
    if log_scale:
        Z_plot = np.log10(np.abs(Z) + 1e-30)  # Avoid log(0)
        cbar_label = f'log$_{{10}}$({var_name})'
    else:
        Z_plot = Z
        cbar_label = var_name
    
    # Determine color limits
    if vmin is None:
        vmin = np.nanpercentile(Z_plot, 2)
    if vmax is None:
        vmax = np.nanpercentile(Z_plot, 98)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    mesh = ax.pcolormesh(X, Y, Z_plot, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    
    # Overlay solar surface
    if overlay_solar_surface:
        theta = np.linspace(0, 2 * np.pi, 200)
        limb_x = np.cos(theta)
        limb_y = np.sin(theta)
        ax.plot(limb_x, limb_y, 'w-', lw=1.5, label='Solar Surface')
    
    # Formatting
    ax.set_xlabel(r'$X$ [$R_\odot$]', fontsize=14)
    ax.set_ylabel(r'$Y$ [$R_\odot$]', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Title
    if title is None:
        metadata = data.get('_metadata', {})
        time_val = metadata.get('time', 0.0)
        title = f'{var_name} at t = {time_val:.2f} [simulation units]'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, label=cbar_label, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    return fig


# ============================================================
# METRICS READING
# ============================================================

def read_metrics_file(filepath: str) -> Optional[Dict[str, float]]:
    """
    Read a CR*_omni.txt or CR*_sta.txt metrics file.

    Format: Dist_U = 0.1993 (one per line)

    Parameters
    ----------
    filepath : str
        Path to the metrics file.

    Returns
    -------
    dict or None
        Dictionary of metrics like {'Dist_U': 0.1993, ...}.
        Returns None if file not found.
    """
    if not os.path.isfile(filepath):
        return None

    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=')
                metrics[key.strip()] = float(val.strip())
    return metrics


# ============================================================
# VISUALIZATION — IN-SITU COMPARISON
# ============================================================

def plot_insitu_comparison(
    sim_df: Optional[pd.DataFrame],
    obs_df: Optional[pd.DataFrame],
    spacecraft_label: str,
    model_label: str = 'SWMF'
) -> Optional[plt.Figure]:
    """
    Create a 4-panel in-situ comparison plot.

    Panels: Radial Velocity, Number Density, Temperature, |B|.
    Observation in black, Simulation in red.

    Parameters
    ----------
    sim_df : pd.DataFrame or None
        Simulation data from read_sat_file().
    obs_df : pd.DataFrame or None
        Observation data from read_obs_file().
    spacecraft_label : str
        Spacecraft/location label (e.g., 'OMNI', 'STEREO-A').
    model_label : str, optional
        Model label for legend (default: 'SWMF').

    Returns
    -------
    plt.Figure or None
        Matplotlib figure object, or None if no data available.
    """
    if sim_df is None and obs_df is None:
        print(f"  [SKIP] No data available for {spacecraft_label}")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ylabels = [
        r'$U_r$ [km/s]',
        r'$N_p$ [cm$^{-3}$]',
        r'Temperature [K]',
        r'$|B|$ [nT]'
    ]

    # --- Observation (black) ---
    if obs_df is not None:
        axes[0].plot(obs_df['date'], obs_df['V_tot'], 'k-', lw=0.8, label=spacecraft_label + ' Obs')
        axes[1].plot(obs_df['date'], obs_df['Rho'], 'k-', lw=0.8)
        axes[2].plot(obs_df['date'], obs_df['Temperature'], 'k-', lw=0.8)
        axes[3].plot(obs_df['date'], obs_df['B_tot'], 'k-', lw=0.8)

    # --- Simulation (red) ---
    if sim_df is not None:
        axes[0].plot(sim_df['date'], sim_df['ur'], 'r-', lw=0.8, label=model_label)
        axes[1].plot(sim_df['date'], sim_df['ndens'], 'r-', lw=0.8)
        axes[2].plot(sim_df['date'], sim_df['ti'], 'r-', lw=0.8)
        axes[3].plot(sim_df['date'], sim_df['bmag'], 'r-', lw=0.8)

    # --- Formatting ---
    for i, ax in enumerate(axes):
        ax.set_ylabel(ylabels[i], fontsize=12, fontweight='bold')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(labelsize=10)

    # Temperature axis: scientific notation
    axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # X-axis
    if sim_df is not None:
        axes[0].set_xlim(sim_df['date'].iloc[0], sim_df['date'].iloc[-1])
        start_label = sim_df['date'].iloc[0].strftime('%d-%b-%Y %H:%M')
    elif obs_df is not None:
        axes[0].set_xlim(obs_df['date'].iloc[0], obs_df['date'].iloc[-1])
        start_label = obs_df['date'].iloc[0].strftime('%d-%b-%Y %H:%M')
    else:
        start_label = ''

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axes[3].xaxis.set_major_locator(plt.MaxNLocator(8))
    axes[3].set_xlabel(f'Start Time ({start_label})', fontsize=12, fontweight='bold')

    axes[0].legend(frameon=False, loc='upper left', fontsize=11, prop={'weight': 'bold'})

    fig.suptitle(f'In-Situ Comparison — {spacecraft_label}', fontsize=14, fontweight='bold', y=0.99)
    plt.subplots_adjust(left=0.12, bottom=0.08, right=0.97, top=0.96, hspace=0.1)

    return fig


# ============================================================
# VISUALIZATION — REMOTE SENSING COMPARISON
# ============================================================

def visualize_euv_grid(
    data_list: list,
    channels: list,
    title: str = ""
) -> plt.Figure:
    """
    Visualize a list of EUV images in a multi-column grid.

    Useful for displaying multiple wavelength channels from either
    simulation or observation data.

    Parameters
    ----------
    data_list : list of tuples
        List of (x_2d, y_2d, img_2d) tuples for each channel.
    channels : list of str
        List of AIA channel names (e.g., ['171', '193', '211']).
    title : str, optional
        Overall figure title (default: "").

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    n_images = len(data_list)
    if n_images == 0:
        print("  [WARNING] No images to display.")
        return None

    # Determine grid layout (try to make it roughly square)
    if n_images <= 3:
        nrows, ncols = 1, n_images
    elif n_images <= 6:
        nrows, ncols = 2, 3
    elif n_images <= 9:
        nrows, ncols = 3, 3
    else:
        nrows = int(np.ceil(np.sqrt(n_images)))
        ncols = int(np.ceil(n_images / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (data, channel) in enumerate(zip(data_list, channels)):
        ax = axes[idx]
        x_2d, y_2d, img_2d = data

        if img_2d is not None:
            # Apply log transform
            img_log = log_transform(img_2d)

            # Get colormap for this channel
            cmap = AIA_CMAPS.get(channel, 'hot')

            # Draw image
            im = ax.pcolormesh(x_2d, y_2d, img_log, cmap=cmap, shading='auto')

            # Draw solar limb
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), 'w-', lw=0.8)

            # Set limits and aspect
            ax.set_xlim(-1.29, 1.29)
            ax.set_ylim(-1.29, 1.29)
            ax.set_aspect('equal')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='log$_{10}$(DN/s)', shrink=0.85)
        else:
            ax.text(0.5, 0.5, f'No data\n{channel}Å', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

        # Panel title
        ax.set_title(f'AIA {channel}Å', fontsize=13, fontweight='bold')
        ax.set_xlabel('X [R$_\\odot$]')
        ax.set_ylabel('Y [R$_\\odot$]')

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    return fig


def plot_remote_comparison(
    sim_data,
    obs_data,
    channel: str,
    event_time: datetime,
    cmap: Optional[str] = None
) -> plt.Figure:
    """
    Create a side-by-side EUV image comparison plot.

    Left: Synthetic SWMF LOS image.
    Right: Observed AIA image.
    Both in log10 scale with shared colorbar.

    Parameters
    ----------
    sim_data : tuple of (x_2d, y_2d, img_2d)
        Simulation data from read_los_file().
    obs_data : tuple of (x_2d, y_2d, img_2d)
        Observation data from read_aia_obs_file().
    channel : str
        AIA wavelength channel (e.g., '193').
    event_time : datetime
        Event start time for plot title.
    cmap : str, optional
        Colormap name. If None, uses AIA_CMAPS[channel].

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if cmap is None:
        cmap = AIA_CMAPS.get(channel, 'hot')

    x_sim, y_sim, img_sim = sim_data
    x_obs, y_obs, img_obs = obs_data

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: Synthetic SWMF image ----
    ax_sim = axes[0]
    if img_sim is not None:
        img_sim_log = log_transform(img_sim)

        # Draw solar limb
        theta = np.linspace(0, 2*np.pi, 200)
        ax_sim.plot(np.cos(theta), np.sin(theta), 'w-', lw=0.8)

        im1 = ax_sim.pcolormesh(x_sim, y_sim, img_sim_log, cmap=cmap, shading='auto')
        ax_sim.set_xlim(-1.5, 1.5)
        ax_sim.set_ylim(-1.5, 1.5)
        ax_sim.set_aspect('equal')
    else:
        ax_sim.text(0.5, 0.5, 'No synthetic data', ha='center', va='center',
                    transform=ax_sim.transAxes, fontsize=14, color='gray')

    ax_sim.set_title(f'Synthetic SWMF — AIA {channel}Å', fontsize=13, fontweight='bold')
    ax_sim.set_xlabel('X [R$_\\odot$]')
    ax_sim.set_ylabel('Y [R$_\\odot$]')

    # ---- Right: Observation ----
    ax_obs = axes[1]
    if img_obs is not None and x_obs is not None:
        img_obs_log = log_transform(img_obs)

        # Draw solar limb
        theta = np.linspace(0, 2*np.pi, 200)
        ax_obs.plot(np.cos(theta), np.sin(theta), 'w-', lw=0.8)

        im2 = ax_obs.pcolormesh(x_obs, y_obs, img_obs_log, cmap=cmap, shading='auto')
        ax_obs.set_xlim(-1.5, 1.5)
        ax_obs.set_ylim(-1.5, 1.5)
        ax_obs.set_aspect('equal')
    else:
        ax_obs.text(0.5, 0.5, 'No observation data', ha='center', va='center',
                    transform=ax_obs.transAxes, fontsize=14, color='gray')

    ax_obs.set_title(f'Observed SDO/AIA {channel}Å — {event_time.strftime("%Y-%m-%d")}',
                     fontsize=13, fontweight='bold')
    ax_obs.set_xlabel('X [R$_\\odot$]')
    ax_obs.set_ylabel('Y [R$_\\odot$]')

    # Match colorbar limits
    if img_sim is not None and img_obs is not None:
        # vmin = max(np.nanmin(img_sim_log), np.nanmin(img_obs_log))
        # vmax = min(np.nanmax(img_sim_log), np.nanmax(img_obs_log))
        vmin = -1.0
        vmax = 4.0
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)
        plt.colorbar(im2, ax=axes, label=f'log$_{{10}}$(Intensity [DN/s])', shrink=0.85)
    elif img_sim is not None:
        plt.colorbar(im1, ax=ax_sim, label=f'log$_{{10}}$(Intensity)', shrink=0.85)
    elif img_obs is not None:
        plt.colorbar(im2, ax=ax_obs, label=f'log$_{{10}}$(Intensity)', shrink=0.85)

    fig.suptitle(f'EUV Remote Sensing Comparison — AIA {channel}Å',
                 fontsize=15, fontweight='bold', y=1.01)
    # plt.tight_layout()

    return fig


def plot_wavelength_comparison_grid(
    sim_data_list: list,
    obs_data_list: list,
    channels: list,
    event_time: datetime,
    max_cols: int = 3
) -> Optional[plt.Figure]:
    """
    Create a multi-row wavelength comparison grid with simulation and observation.

    This function creates a grid layout where each "block" contains:
    - Top row: Simulation images for up to `max_cols` channels
    - Bottom row: Corresponding observation images

    If there are more channels than `max_cols`, they wrap to additional blocks.

    Parameters
    ----------
    sim_data_list : list of tuples
        List of (x_2d, y_2d, img_2d) for simulation data, one per channel.
    obs_data_list : list of tuples
        List of (x_2d, y_2d, img_2d) for observation data, one per channel.
    channels : list of str
        List of AIA channel names (e.g., ['171', '193', '211']).
    event_time : datetime
        Event start time for plot title.
    max_cols : int, optional
        Maximum number of columns per block (default: 3).

    Returns
    -------
    plt.Figure or None
        Matplotlib figure object, or None if no data available.

    Notes
    -----
    - Each channel uses observation-based vmin/vmax scaling
    - Both simulation and observation use the same color scale per channel
    - Colormap is selected from AIA_CMAPS based on wavelength
    - Solar limb (r=1.0) is drawn as white circle on each panel
    """
    n_channels = len(channels)
    if n_channels == 0 or len(sim_data_list) == 0 or len(obs_data_list) == 0:
        print("  [WARNING] No data available for wavelength comparison.")
        return None

    # Calculate grid layout
    num_blocks = int(np.ceil(n_channels / max_cols))
    num_cols = min(n_channels, max_cols)
    num_rows = 2 * num_blocks  # Each block has 2 rows (sim + obs)

    # Calculate figure size
    width = 5 * num_cols
    height = 4 * num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    
    # Handle single row/col case
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Solar limb coordinates
    theta = np.linspace(0, 2*np.pi, 200)
    limb_x = np.cos(theta)
    limb_y = np.sin(theta)

    # Process each channel
    for idx, channel in enumerate(channels):
        # Determine position in grid
        block_idx = idx // max_cols
        col_idx = idx % max_cols
        sim_row = 2 * block_idx
        obs_row = 2 * block_idx + 1

        # Get data
        x_sim, y_sim, img_sim = sim_data_list[idx] if idx < len(sim_data_list) else (None, None, None)
        x_obs, y_obs, img_obs = obs_data_list[idx] if idx < len(obs_data_list) else (None, None, None)

        # Get colormap
        cmap = AIA_CMAPS.get(channel, 'hot')

        # --- Process observation to get vmin/vmax ---
        vmin, vmax = -1.0, 4.0  # Default range
        if img_obs is not None:
            img_obs_log = log_transform(img_obs)
            vmin = np.nanpercentile(img_obs_log, 1)
            vmax = np.nanpercentile(img_obs_log, 99)

        # --- Plot Simulation (top row of block) ---
        ax_sim = axes[sim_row, col_idx]
        im_sim = None
        if img_sim is not None:
            img_sim_log = log_transform(img_sim)
            
            im_sim = ax_sim.pcolormesh(x_sim, y_sim, img_sim_log, 
                                       cmap=cmap, shading='auto', 
                                       vmin=vmin, vmax=vmax)
            ax_sim.plot(limb_x, limb_y, 'w-', lw=0.8)
            ax_sim.set_xlim(-1.29, 1.29)
            ax_sim.set_ylim(-1.29, 1.29)
            ax_sim.set_aspect('equal')
        else:
            ax_sim.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax_sim.transAxes, fontsize=10, color='gray')
            ax_sim.set_xlim(-1.29, 1.29)
            ax_sim.set_ylim(-1.29, 1.29)

        # Label y-axis only for leftmost column
        if col_idx == 0:
            ax_sim.set_ylabel('SWMF Synthetic', fontsize=11, fontweight='bold')
        else:
            ax_sim.set_yticklabels([])

        # Title for each channel (always show on simulation rows)
        ax_sim.set_title(f'{channel} Å', fontsize=12, fontweight='bold')

        ax_sim.set_xticklabels([])
        ax_sim.tick_params(axis='both', which='both', length=0)

        # --- Plot Observation (bottom row of block) ---
        ax_obs = axes[obs_row, col_idx]
        im_obs = None
        if img_obs is not None:
            img_obs_log = log_transform(img_obs)
            
            im_obs = ax_obs.pcolormesh(x_obs, y_obs, img_obs_log, 
                                       cmap=cmap, shading='auto',
                                       vmin=vmin, vmax=vmax)
            ax_obs.plot(limb_x, limb_y, 'w-', lw=0.8)
            ax_obs.set_xlim(-1.29, 1.29)
            ax_obs.set_ylim(-1.29, 1.29)
            ax_obs.set_aspect('equal')
        else:
            ax_obs.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax_obs.transAxes, fontsize=10, color='gray')
            ax_obs.set_xlim(-1.29, 1.29)
            ax_obs.set_ylim(-1.29, 1.29)

        # Label y-axis only for leftmost column
        if col_idx == 0:
            ax_obs.set_ylabel('SDO/AIA Obs', fontsize=11, fontweight='bold')
        else:
            ax_obs.set_yticklabels([])

        ax_obs.set_xticklabels([])
        ax_obs.tick_params(axis='both', which='both', length=0)
        
        # # Add colorbar spanning both simulation and observation
        # if im_sim is not None or im_obs is not None:
        #     mappable = im_obs if im_obs is not None else im_sim
        #     cbar = plt.colorbar(mappable, 
        #                        ax=[ax_sim, ax_obs], 
        #                        label=f'log$_{{10}}$(DN/s)', 
        #                        shrink=0.9, 
        #                        pad=0.02)

    # Hide unused subplots
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate which channel this position corresponds to
            block = row // 2
            channel_idx = block * max_cols + col
            
            if channel_idx >= n_channels:
                axes[row, col].axis('off')

    # Overall title
    date_str = event_time.strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f'Multi-Wavelength EUV Comparison — {date_str}', 
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    return fig

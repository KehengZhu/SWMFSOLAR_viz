# SWMF Visualization Library (`swmf_viz.py`)

A comprehensive Python library for comparing Space Weather Modeling Framework (SWMF) simulation results against observational data.

## Overview

`swmf_viz.py` provides a complete toolkit for analyzing and visualizing SWMF simulation outputs, with support for:

- **In-situ diagnostics**: Time-series comparison of solar wind parameters at Earth and STEREO spacecraft
- **Remote sensing**: Synthetic EUV image generation and comparison with SDO/AIA observations
- **Multi-wavelength analysis**: Grid visualization across multiple AIA channels (94Å–335Å)
- **Binary data reading**: Parse native SWMF 2D slice outputs (Fortran unformatted binary)
- **Quantitative metrics**: Calculate normalized distance metrics between simulation and observation

## Features

### 🔍 Data Parsing & Discovery
- Automatic PARAM.in metadata extraction
- Smart file discovery for satellite trajectories, LOS outputs, and observations
- Support for FITS, ASCII, and binary SWMF formats

### 📊 Visualization Capabilities
- **4-panel in-situ plots**: Radial velocity, density, temperature, magnetic field
- **Multi-wavelength EUV grids**: Compare up to 7 AIA channels simultaneously
- **Side-by-side comparison**: Synthetic vs. observed imagery with matched color scaling
- **2D slice visualization**: Density, temperature, magnetic field distributions

### 🧮 Physical Units & Conversions
- CGS → SI unit conversions
- Number density from mass density
- Ion/electron temperature from pressure
- Total magnetic field including Alfvén wave energy

---

## Installation

### Requirements

```bash
pip install numpy pandas matplotlib astropy
```

**Optional dependencies:**
- `astropy` (for FITS file support)
- Standard SDO/AIA colormaps (for authentic channel colors)

### Quick Start

```python
import swmf_viz as sv
import matplotlib.pyplot as plt
from datetime import datetime

# Parse simulation metadata
event_time = sv.parse_start_time('Results/Run_Max/run01/PARAM.in')

# Load in-situ data
sat_file = sv.find_sat_file('Results/Run_Max/run01/IH', 'earth')
sim_data = sv.read_sat_file(sat_file)

# Create comparison plot
obs_file = sv.find_obs_file('Results/obsdata', 'omni', event_time)
obs_data = sv.read_obs_file(obs_file)

fig = sv.plot_insitu_comparison(sim_data, obs_data, 'OMNI')
plt.show()
```

---

## Module Structure

### 1. Physical Constants

```python
PROTON_MASS_CGS = 1.6726e-24  # g
BOLTZMANN_CGS = 1.3807e-16    # erg/K
GAUSS_TO_NT = 1.0e5           # Gauss to nanoTesla
```

### 2. AIA Colormap Configuration

```python
AIA_CMAPS = {
    '94': 'sdoaia94',    # Hot plasma (6 MK)
    '131': 'sdoaia131',  # Flare plasma (10 MK)
    '171': 'sdoaia171',  # Quiet corona (0.6 MK)
    '193': 'sdoaia193',  # Hot active regions (1.5 MK)
    '211': 'sdoaia211',  # Active regions (2 MK)
    '304': 'sdoaia304',  # Chromosphere/transition region
    '335': 'sdoaia335',  # Active region loops (2.5 MK)
}
```

---

## API Reference

### Metadata Parsing

#### `parse_start_time(param_file: str) -> datetime`

Extract simulation start time from SWMF PARAM.in file.

**Parameters:**
- `param_file`: Path to PARAM.in

**Returns:**
- `datetime` object with event start time

**Raises:**
- `FileNotFoundError`: If PARAM.in doesn't exist
- `ValueError`: If #STARTTIME block not found

**Example:**
```python
event_time = sv.parse_start_time('Results/run01/PARAM.in')
# Output: datetime(2020, 7, 1, 12, 0, 0)
```

---

### File Discovery

#### `find_obs_file(obs_dir: str, prefix: str, event_time: datetime) -> Optional[str]`

Locate observation file matching event time.

**File naming convention:** `<prefix>_YYYY_MM_DDThh_mm_ss.out`

**Parameters:**
- `obs_dir`: Directory containing observation files
- `prefix`: File prefix (`'omni'`, `'sta'`)
- `event_time`: Event start time

**Returns:**
- Path to observation file, or `None` if not found

---

#### `find_sat_file(ih_dir: str, spacecraft: str) -> Optional[str]`

Find satellite trajectory file in IH/ directory.

**Parameters:**
- `ih_dir`: Path to IH directory
- `spacecraft`: Spacecraft name (`'earth'`, `'sta'`)

**Returns:**
- Path to `.sat` file, or `None`

---

#### `find_los_file(sc_dir: str, instrument: str = 'sdo_aia') -> Optional[str]`

Find latest LOS (line-of-sight) output file in SC/.

**Parameters:**
- `sc_dir`: Path to SC directory
- `instrument`: Instrument name (default: `'sdo_aia'`)

**Returns:**
- Path to LOS `.out` file, or `None`

---

#### `find_aia_fits(obs_dir: str, event_time: datetime, channel: str = '193') -> Optional[str]`

Find AIA FITS file matching event date and wavelength channel.

**Parameters:**
- `obs_dir`: Directory with FITS files
- `event_time`: Event start time
- `channel`: AIA wavelength (`'94'`, `'131'`, `'171'`, `'193'`, `'211'`, `'304'`, `'335'`)

**Returns:**
- Path to FITS file, or `None`

---

### Data Reading — In-Situ

#### `read_sat_file(filepath: str) -> Optional[pd.DataFrame]`

Read SWMF IH satellite trajectory file (`.sat`).

**Output columns:**
- `date`: Timestamp
- `ur`: Radial velocity [km/s]
- `ndens`: Number density [cm⁻³]
- `ti`: Ion temperature [K]
- `te`: Electron temperature [K]
- `bmag`: Total magnetic field [nT] (includes Alfvén wave energy)

**Unit conversions:**
- Density: `rho [g/cm³] → ndens = rho / m_p`
- Temperature: `T = p * m_p / (rho * k_B)`
- B-field: Gauss → nT, includes `sqrt(4π * (I01 + I02))`

**Returns:**
- `pd.DataFrame` with physical quantities

---

#### `read_obs_file(filepath: str) -> Optional[pd.DataFrame]`

Read SWMF-formatted observation file.

**Output columns:**
- `date`: Timestamp
- `V_tot`: Total velocity [km/s]
- `Rho`: Number density [cm⁻³]
- `Temperature`: [K]
- `B_tot`: Total magnetic field [nT]
- `Br`: Radial B-field component [nT]

**Data cleaning:**
- Removes fill values (large negative numbers, `9999`-type flags)

---

### Data Reading — Remote Sensing

#### `log_transform(image: np.ndarray, vmin: float = 0.1) -> np.ndarray`

Apply logarithmic transform to image data.

**Parameters:**
- `image`: Input image array
- `vmin`: Minimum clipping value (default: 0.1)

**Returns:**
- `log10(image)` with negative/zero values clipped

---

#### `read_los_file(filepath: str, channel: str = '193') -> Tuple[Optional[np.ndarray], ...]`

Read SWMF synthetic LOS file (e.g., `los_sdo_aia_*.out`).

**Parameters:**
- `filepath`: Path to LOS output
- `channel`: AIA channel to extract

**Returns:**
- `(x_2d, y_2d, img_2d)`: Coordinate grids [R☉] and synthetic image

**File format:**
- 5-line header (title, dimensions, variable names)
- Columns: X, Y, AIA:94, AIA:131, ..., AIA:335

---

#### `read_aia_obs_file(filepath: str, channel: str = '193') -> Tuple[Optional[np.ndarray], ...]`

Read processed AIA observation file (`AIA_Observations_*.out`).

**Parameters:**
- `filepath`: Path to observation file
- `channel`: AIA channel

**Returns:**
- `(x_2d, y_2d, img_2d)`: Coordinate grids [R☉] and observed image (linear intensity)

---

#### `read_aia_fits(filepath: str) -> Optional[np.ndarray]`

Read SDO/AIA FITS file using `astropy`.

**Parameters:**
- `filepath`: Path to FITS file

**Returns:**
- 2D image array [DN or DN/s], or `None` if file not found

**Requirements:**
- `from astropy.io import fits as pyfits`

---

### Binary Data Reading

#### `read_swmf_binary_2d(filepath: str) -> Optional[Dict[str, np.ndarray]]`

Read binary SWMF 2D slice file (Fortran unformatted).

**File examples:**
- `z=0_var_3_n00080000.out` (equatorial plane)
- `y=0_var_3_n00080000.out` (meridional plane)

**Returns:**
```python
{
    'X': 2D coordinate array [R☉],
    'Y': 2D coordinate array [R☉],
    'Rho': 2D density array [g/cm³],
    'T': 2D temperature array [K],
    'Ux', 'Uy', 'Uz': Velocity components,
    'Bx', 'By', 'Bz': Magnetic field components,
    'P': Pressure,
    '_metadata': {
        'it': iteration number,
        'time': simulation time,
        'ndim': dimensions (2),
        'nx', 'ny': grid sizes,
        'varnames': list of variables
    }
}
```

**Notes:**
- Automatically handles Fortran record markers
- Reshapes 1D data to 2D grids (ny × nx)
- Returns `None` if file corrupt or not found

---

#### `plot_swmf_2d_slice(data: Dict, var_name: str, log_scale: bool = True, ...) -> Optional[plt.Figure]`

Visualize a 2D slice from SWMF binary output.

**Parameters:**
- `data`: Dictionary from `read_swmf_binary_2d()`
- `var_name`: Variable to plot (`'Rho'`, `'T'`, `'Ux'`, etc.)
- `log_scale`: Apply log₁₀ color scale (default: `True`)
- `vmin`, `vmax`: Color limits (auto-scaled if `None`)
- `cmap`: Matplotlib colormap (default: `'viridis'`)
- `title`: Plot title
- `overlay_solar_surface`: Draw circle at r=1.0 R☉ (default: `True`)
- `figsize`: Figure size (default: `(10, 9)`)

**Returns:**
- Matplotlib figure object

**Example:**
```python
data = sv.read_swmf_binary_2d('z=0_var_3_n00080000.out')
fig = sv.plot_swmf_2d_slice(data, 'Rho', log_scale=True, cmap='plasma')
plt.show()
```

---

### Metrics

#### `read_metrics_file(filepath: str) -> Optional[Dict[str, float]]`

Read quantitative comparison metrics file.

**File format:** `Dist_U = 0.1993` (one metric per line)

**Returns:**
```python
{
    'Dist_U': 0.1993,  # Velocity distance
    'Dist_N': 0.2145,  # Density distance
    'Dist_T': 0.3021,  # Temperature distance
    'Dist_B': 0.1756   # B-field distance
}
```

**Lower values** indicate better agreement with observations.

---

### Visualization — In-Situ

#### `plot_insitu_comparison(sim_df, obs_df, spacecraft_label: str, model_label: str = 'SWMF') -> Optional[plt.Figure]`

Create 4-panel in-situ comparison plot.

**Panels:**
1. Radial velocity [km/s]
2. Number density [cm⁻³]
3. Temperature [K] (scientific notation)
4. Total magnetic field [nT]

**Parameters:**
- `sim_df`: Simulation data from `read_sat_file()`
- `obs_df`: Observation data from `read_obs_file()`
- `spacecraft_label`: Display label (`'OMNI'`, `'STEREO-A'`)
- `model_label`: Model name for legend (default: `'SWMF'`)

**Returns:**
- Matplotlib figure with 4 vertically-stacked panels

**Color convention:**
- **Black**: Observation
- **Red**: Simulation

---

### Visualization — Remote Sensing

#### `visualize_euv_grid(data_list: list, channels: list, title: str = "") -> plt.Figure`

Visualize multiple EUV images in a grid layout.

**Parameters:**
- `data_list`: List of `(x_2d, y_2d, img_2d)` tuples
- `channels`: List of channel names (`['171', '193', '211']`)
- `title`: Overall figure title

**Returns:**
- Matplotlib figure with multi-column grid

**Layout:**
- 1–3 images: 1 row
- 4–6 images: 2 rows
- 7–9 images: 3 rows
- 10+ images: 4 rows

---

#### `plot_remote_comparison(sim_data, obs_data, channel: str, event_time: datetime, cmap: Optional[str] = None) -> plt.Figure`

Side-by-side comparison of synthetic vs. observed EUV imagery.

**Parameters:**
- `sim_data`: `(x_sim, y_sim, img_sim)` from `read_los_file()`
- `obs_data`: `(x_obs, y_obs, img_obs)` from `read_aia_obs_file()`
- `channel`: AIA wavelength (`'193'`, etc.)
- `event_time`: Event timestamp for title
- `cmap`: Colormap (uses `AIA_CMAPS` if `None`)

**Returns:**
- 2-column figure (Synthetic | Observed)

**Features:**
- Matched logarithmic color scaling
- Solar surface overlay (white circle at r=1.0)
- Channel-specific colormaps

---

#### `plot_wavelength_comparison_grid(sim_data_list, obs_data_list, channels: list, event_time: datetime, max_cols: int = 3) -> Optional[plt.Figure]`

Comprehensive multi-wavelength comparison grid.

**Parameters:**
- `sim_data_list`: List of synthetic image tuples
- `obs_data_list`: List of observed image tuples
- `channels`: List of channel names
- `event_time`: Event timestamp
- `max_cols`: Maximum columns per row (default: 3)

**Returns:**
- Multi-row grid with each wavelength pair stacked vertically

**Design:**
- Each channel → 2 rows (Synthetic + Observed)
- Observation-based dynamic color scaling
- Handles missing data gracefully

---

## Usage Examples

### Example 1: Basic In-Situ Comparison

```python
import swmf_viz as sv
import matplotlib.pyplot as plt

# Setup paths
run_path = 'Results/Run_Max/run01'
obs_path = 'Results/obsdata'
event_time = sv.parse_start_time(f'{run_path}/PARAM.in')

# Load Earth data
sat_file = sv.find_sat_file(f'{run_path}/IH', 'earth')
obs_file = sv.find_obs_file(obs_path, 'omni', event_time)

sim_data = sv.read_sat_file(sat_file)
obs_data = sv.read_obs_file(obs_file)

# Create comparison plot
fig = sv.plot_insitu_comparison(sim_data, obs_data, 'OMNI')
plt.savefig('comparison_earth.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 2: Multi-Wavelength EUV Grid

```python
import swmf_viz as sv
import matplotlib.pyplot as plt

# Load synthetic LOS data for multiple channels
channels = ['171', '193', '211', '304']
los_file = sv.find_los_file('Results/run01/SC', 'sdo_aia')

sim_data_list = []
for ch in channels:
    x, y, img = sv.read_los_file(los_file, ch)
    if img is not None:
        sim_data_list.append((x, y, img))

# Create grid visualization
fig = sv.visualize_euv_grid(sim_data_list, channels, 
                             title='Synthetic Multi-Wavelength EUV')
plt.show()
```

---

### Example 3: Binary 2D Slice Analysis

```python
import swmf_viz as sv
import matplotlib.pyplot as plt

# Read binary slice file
data = sv.read_swmf_binary_2d('Results/run01/SC/z=0_var_3_n00080000.out')

if data:
    print("Available variables:", [k for k in data.keys() 
                                    if k not in ['X', 'Y', '_metadata']])
    
    # Visualize density
    fig1 = sv.plot_swmf_2d_slice(data, 'Rho', log_scale=True, 
                                  cmap='plasma', 
                                  title='Density in Equatorial Plane')
    
    # Visualize temperature
    fig2 = sv.plot_swmf_2d_slice(data, 'T', log_scale=True, 
                                  cmap='hot',
                                  title='Temperature Distribution')
    
    plt.show()
```

---

### Example 4: Calculate Quantitative Metrics

```python
import swmf_viz as sv
import glob

# Find all metrics files
metric_files = glob.glob('Results/run01/CR*_omni.txt')

for mf in metric_files:
    metrics = sv.read_metrics_file(mf)
    if metrics:
        print(f"\n{mf}:")
        print(f"  Velocity:    {metrics['Dist_U']:.4f}")
        print(f"  Density:     {metrics['Dist_N']:.4f}")
        print(f"  Temperature: {metrics['Dist_T']:.4f}")
        print(f"  B-field:     {metrics['Dist_B']:.4f}")
```

---

## File Format Reference

### PARAM.in Structure

```
#STARTTIME
2020            iYear
7               iMonth  
1               iDay
12              iHour
0               iMinute
0               iSecond
0.0             FracSecond
```

### Satellite File (.sat)

```
it year mo dy hr mn sc msc x y z rho ux uy uz p bx by bz pe i01 i02 ...
0 2020  7  1 12  0  0   0 0.01 0.0 0.0 1.2e-16 40.0 0.0 0.0 2.3e-10 ...
```

### Observation File (.out)

```
Satellite data file from OMNI database
Columns: year mo dy hr mn sc V_tot Rho Temperature B_tot Br ...
---
2020 7  1 12  0  0 350.5 8.2 1.5e5 5.3 -2.1 ...
```

### LOS Output File

```
Title: Synthetic AIA observations
nx ny
180 180
Columns: X Y AIA:94 AIA:131 AIA:171 AIA:193 AIA:211 AIA:304 AIA:335
-1.2 -1.2 1.23e4 5.67e5 ...
```

---

## Best Practices

### Color Scaling
- **In-situ plots**: Linear scales for velocity, density, B-field; scientific notation for temperature
- **EUV images**: Logarithmic scaling (log₁₀) with `vmin` clipping at 0.1
- **Match observation scales**: Use observation data percentiles for comparison plots

### Data Quality
- Check for fill values in observation files (typically `-999`, `9999`)
- Handle missing channels gracefully (some events may lack certain wavelengths)
- Verify coordinate alignment between synthetic and observed images

### Performance
- Binary slice files can be large (100+ MB); use targeted variable extraction
- For batch processing, reuse parsed metadata instead of re-reading PARAM.in
- Consider downsampling high-resolution images for quick previews

---

## Coordinate Systems

### Heliocentric Earth Equatorial (HEE)
- **Origin**: Sun center
- **Z-axis**: Ecliptic north
- **X-axis**: Sun-Earth line
- Used in IH satellite trajectories

### Solar Radius Units
- All spatial coordinates in **R☉** (solar radii)
- 1 R☉ = 6.96 × 10⁵ km
- Typical domain: 1.0 R☉ (photosphere) to 20 R☉

---

## Troubleshooting

### Common Issues

**Q: Function returns `None` instead of data**
- Check file paths are absolute or correct relative paths
- Verify file naming conventions match expected patterns
- Ensure files aren't empty or corrupted

**Q: Colors look wrong in EUV plots**
- Install SDO/AIA colormaps: `pip install sunpy` (provides `sdoaia*` cmaps)
- Or use fallback colormaps (automatically applied)

**Q: Temperature values seem off**
- Verify units: output is in Kelvin (K)
- Check if using ion temperature (`ti`) vs electron temperature (`te`)
- Typical solar wind: 10⁴–10⁶ K

**Q: Binary file reading fails**
- Ensure file is Fortran unformatted (not HDF5 or NetCDF)
- Check byte order (little-endian vs big-endian)
- Verify file hasn't been corrupted during transfer

---

## Contributing

### Extending the Library

To add new functionality:

1. **New data formats**: Add reader function in appropriate section
2. **New visualizations**: Follow naming convention `plot_<diagnostic>_<type>()`
3. **New metrics**: Add to `read_metrics_file()` or create specialized reader
4. **Unit tests**: Include sample files and expected outputs

### Code Style
- Follow existing naming conventions (snake_case for functions)
- Add type hints: `def func(arg: str) -> Optional[pd.DataFrame]`
- Include docstrings with Parameters/Returns/Raises sections
- Use physical units in comments (km/s, cm⁻³, nT)

---

## References

### SWMF Documentation
- [SWMF Manual](http://herot.engin.umich.edu/~gtoth/SWMF/doc/HTML/SWMF/)
- [BATS-R-US Documentation](http://herot.engin.umich.edu/~gtoth/BATSRUS/doc/HTML/)

### SDO/AIA
- [AIA Instrument Paper (Lemen et al. 2012)](https://doi.org/10.1007/s11207-011-9776-8)
- [SunPy Documentation](https://docs.sunpy.org/)

### STEREO
- [STEREO Mission Overview](https://stereo.gsfc.nasa.gov/)
- [PLASTIC Instrument](https://stereo.gsfc.nasa.gov/instruments/plastic.shtml)

---

## License

This software is part of the SWMF framework. See [SWMF License](LICENSE.txt) for details.

---

## Contact


---

## Version History


---

*Last updated: February 2026*

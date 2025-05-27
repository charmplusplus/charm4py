# MiniWeather - Distributed Weather Simulation Mini-App

This is a Charm4Py port of [MiniWeather](https://github.com/mrnorman/miniWeather), an atmospheric dynamics mini-app developed by Oak Ridge National Laboratory researchers. 
This example demonstrates GPU acceleration techniques using Charm4Py, combining Charm4Py's distributed parallel computing capabilities with Numba CUDA for GPU acceleration to simulate atmospheric flows with various physical phenomena.

## Features

- **Charm4Py Distributed Computing**: Demonstrates scalable parallel execution across multiple processes/nodes
- **GPU Acceleration**: Leverages Numba CUDA for high-performance computation on NVIDIA GPUs
- **Multiple Physics Models**: Supports various atmospheric phenomena including:
  - Collision dynamics
  - Thermal convection
  - Mountain wave propagation
  - Turbulence modeling
  - Density current flows
  - Injection dynamics
- **Domain Decomposition**: Automatic 2D domain decomposition for distributed memory parallelism
- **Conservation Tracking**: Monitors mass and energy conservation throughout simulation
- **Flexible Output**: Configurable data output at specified intervals
- **Visualization Support**: Built-in tools for creating animated visualizations

## Requirements

### Dependencies
- Python 3.7+
- NumPy
- Numba (with CUDA support)
- Matplotlib (for visualization)
- imageio (for GIF creation)

### Hardware Requirements
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Multiple CPU cores recommended for distributed execution

## Installation

1. Install required Python packages:
```bash
pip install numpy numba matplotlib imageio
```

2. Ensure CUDA is properly installed and configured for Numba:
```bash
python -c "from numba import cuda; print(cuda.gpus)"
```

## Usage

### Basic Simulation

Run a basic thermal convection simulation:
```bash
python3 -m charmrun.start +p4 miniweather/miniweather.py --data_spec thermal --sim_time 10.0 --nx_glob 400 --nz_glob 200
```

### Distributed Execution

Run with domain decomposition across multiple chares:
```bash
python3 -m charmrun.start +p8 miniweather/miniweather.py \
    --data_spec mountain_waves \
    --num_chares_x 2 \
    --num_chares_z 2 \
    --nx_glob 800 \
    --nz_glob 400 \
    --sim_time 20.0 \
    --output_freq 50 \
    --output_dir simulation_output
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--nx_glob` | 200 | Number of global cells in x-direction |
| `--nz_glob` | 100 | Number of global cells in z-direction |
| `--sim_time` | 1.0 | Simulation time in seconds |
| `--max_iters` | 10000 | Maximum number of iterations |
| `--data_spec` | thermal | Physics model to simulate |
| `--num_chares_x` | 1 | Number of chares in x-direction |
| `--num_chares_z` | 1 | Number of chares in z-direction |
| `--output_freq` | 0 | Output frequency (0 = no output) |
| `--output_dir` | output_data_charm | Output directory |

### Physics Models

Available `--data_spec` options:
- `collision`: Collision dynamics
- `thermal`: Thermal convection (default)
- `mountain_waves`: Mountain wave propagation
- `turbulence`: Turbulence modeling
- `density_current`: Density current flows
- `injection`: Injection dynamics

## Output Format

When `--output_freq` > 0, the simulation outputs `.npz` files containing:
- `state`: 4D array with simulation variables (density, momentum, temperature)
- `etime`: Simulation time
- `chare_nx`, `chare_nz`: Local domain sizes
- `chare_i_beg`, `chare_k_beg`: Global starting indices

File naming convention: `data_iter_XXXXXX_chare_YY_ZZ.npz`

## Visualization

### Creating Animated GIFs

Use the visualization script to create animated GIFs from simulation output:

```bash
python miniweather/create_visualization.py simulation_output --out density_evolution.gif --qoi 0
```

### Visualization Options

| Option | Default | Description |
|--------|---------|-------------|
| `input_dir` | (required) | Directory containing .npz output files |
| `--out` | simulation_qoi0.gif | Output GIF filename |
| `--qoi` | 0 | Quantity of Interest index to visualize |

### Quantities of Interest (QoI)

The simulation tracks 4 primary variables:
- **QoI 0**: Density perturbations
- **QoI 1**: Horizontal wind velocity (u-component)
- **QoI 2**: Vertical wind velocity (w-component) 
- **QoI 3**: Potential temperature perturbations

### Visualization Examples

Create visualizations for different variables:
```bash
# Density evolution
python miniweather/create_visualization.py output_data_charm --out density.gif --qoi 0

# Horizontal wind patterns
python miniweather/create_visualization.py output_data_charm --out u_wind.gif --qoi 1

# Vertical wind patterns  
python miniweather/create_visualization.py output_data_charm --out w_wind.gif --qoi 2

# Temperature perturbations
python miniweather/create_visualization.py output_data_charm --out temperature.gif --qoi 3
```

## Example Workflows

### Medium Size Simulation
```bash
# High-resolution thermal convection study
python3 -m charmrun.start +p16 miniweather/miniweather.py \
    --data_spec thermal \
    --nx_glob 1600 \
    --nz_glob 800 \
    --num_chares_x 4 \
    --num_chares_z 4 \
    --sim_time 100.0 \
    --output_freq 100 \
    --output_dir thermal_highres

# Create visualization
python miniweather/create_visualization.py thermal_highres --out thermal_simulation.gif --qoi 0
```

### Mountain Wave Study
```bash
# Mountain wave propagation
python3 -m charmrun.start +p8 miniweather/miniweather.py \
    --data_spec mountain_waves \
    --nx_glob 800 \
    --nz_glob 400 \
    --num_chares_x 2 \
    --num_chares_z 2 \
    --sim_time 50.0 \
    --output_freq 25

# Visualize vertical velocity (mountain waves)
python miniweather/create_visualization.py output_data_charm --out mountain_waves.gif --qoi 2
```

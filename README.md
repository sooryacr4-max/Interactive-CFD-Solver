# Interactive CFD Solver

A collection of Python-based computational fluid dynamics (CFD) solvers for various fluid flow problems. Each script provides an interactive solution to different CFD scenarios with visualization capabilities.

## Overview

This repository contains multiple CFD solvers implemented in Python, each targeting specific fluid dynamics problems. The solvers use numerical methods to solve the governing equations of fluid flow and provide visual output to help understand the flow behavior.

## Prerequisites

### Required Python Packages

Before running any of the CFD solvers, ensure you have the following packages installed:

- `numpy` - Numerical computing with arrays
- `scipy` - Scientific computing and numerical algorithms  
- `matplotlib` - Plotting and visualization
- `math` - Mathematical functions (included with Python)

### Installation

Install the required packages using pip:

```bash
pip install numpy scipy matplotlib
```

Or if you're using conda:

```bash
conda install numpy scipy matplotlib
```

## Usage

1. **Clone or download** this repository to your local machine
2. **Navigate** to the repository directory
3. **Run any CFD solver** using Python:

```bash
python <solver_filename>.py
```

Each solver will:
- Execute the numerical simulation
- Display results through matplotlib visualizations
- Show flow field properties like velocity, pressure, or temperature distributions

## Repository Structure

```
interactive-cfd-solver/
├── README.md
├── CHANNELFLOW.py                          # Couette flow solver
├── animation.py                            # 1D C-D nozzle flow with shock (with numerical viscosity)
├── animation2(shockwithoutviscosity).py    # 1D C-D nozzle flow with shock (without numerical viscosity)
├── finalmacormack(animated).py             # 1D C-D nozzle flow without shocks (MacCormack scheme)
└── prandtmeyer.py                          # Prandtl-Meyer expansion flow solver
```

## Solvers Description

### 1. CHANNELFLOW.py
**Couette Flow Solver**
- Simulates flow between two parallel plates
- One plate moving, creating shear-driven flow
- Demonstrates viscous flow behavior

### 2. animation.py
**1D Converging-Diverging Nozzle Flow (With Numerical Viscosity)**
- Solves compressible flow through a C-D nozzle
- Includes shock wave formation in the diverging section
- Uses numerical viscosity for stability and shock capturing

### 3. animation2(shockwithoutviscosity).py
**1D Converging-Diverging Nozzle Flow (Without Numerical Viscosity)**
- Similar to animation.py but without artificial viscosity
- Shows shock formation with sharper discontinuities
- Demonstrates the effect of numerical viscosity on shock resolution

### 4. finalmacormack(animated).py
**1D C-D Nozzle Flow (Shock-Free, MacCormack Scheme)**
- Solves smooth nozzle flow without shock waves
- Uses MacCormack finite difference scheme
- Ideal for studying isentropic flow through nozzles

### 5. prandtmeyer.py
**Prandtl-Meyer Expansion Flow**
- Solves supersonic expansion around corners
- Demonstrates expansion fan formation
- Important for supersonic aerodynamics applications

## Features

- **Interactive Visualizations**: Real-time plotting of flow fields and properties
- **Multiple CFD Problems**: Various fluid dynamics scenarios covered
- **Educational Focus**: Clear implementations suitable for learning CFD concepts
- **Standalone Scripts**: Each solver can be run independently

## Getting Started

1. Ensure all dependencies are installed
2. Choose a CFD problem you want to explore:
   - **Viscous flows**: Run `CHANNELFLOW.py` for Couette flow
   - **Compressible nozzle flows**: Try `animation.py`, `animation2(shockwithoutviscosity).py`, or `finalmacormack(animated).py`
   - **Supersonic expansions**: Use `prandtmeyer.py`
3. Run the corresponding Python script:
   ```bash
   python CHANNELFLOW.py
   # or
   python animation.py
   # etc.
   ```
4. Interact with the visualization windows to explore results

## Notes

- **Numerical Viscosity**: The difference between `animation.py` and `animation2(shockwithoutviscosity).py` demonstrates the role of numerical viscosity in shock capturing schemes
- **MacCormack Scheme**: The `finalmacormack(animated).py` uses a specific finite difference method well-suited for hyperbolic equations
- **Animated Results**: Several solvers include time-dependent visualizations showing flow evolution

## Contributing

Feel free to contribute additional CFD solvers or improvements to existing ones. Please ensure your code follows the same structure and includes appropriate documentation.





---

**Note**: These solvers are primarily designed for educational and research purposes. For production CFD applications, consider using established CFD software packages.

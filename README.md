# Reservoir Simulation

A 3D Reservoir Simulation project developed in Python.

## Features

- **Core Simulation**: 3D spatial discretization, Rock, Fluid, and Well models.
- **Implicit Solver**: Finite-difference pressure solver with transmissibility and sparse matrix acceleration.
- **Eclipse Integration**: Support for reading standard `.DATA` input decks using the `opm` library.

## Setting up the Environment

This project uses a standard Python virtual environment. It requires `numpy`, `scipy`, `pytest`, and `opm`.

To set up the environment and install dependencies, run the following commands from the root directory of the project:

```bash
# 1. Create a virtual environment named .venv
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install the required packages
pip install numpy scipy pytest opm resdata
```

## Running the Simulation

To run the main simulation using the provided sample Eclipse deck:

```bash
python3 main.py
```

## Running the Tests

We use `pytest` for unit testing the core simulation structures.

```bash
pytest
```

## Debugging

Configurations for VS Code are provided in `.vscode/launch.json` for both individual tests and the main application.


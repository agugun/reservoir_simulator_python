# High-Performance Reservoir Simulator (Python)

An industrial-grade 3D Black-Oil Reservoir Simulator developed entirely in modern Python.

This project implements a Fully Implicit (FIM) Finite-Difference mathematical core utilizing state-of-the-art native super-computing paradigms—including **JAX XLA compilations** and **CSR Sparse Direct Solvers**—fundamentally proving Python's viability for executing multi-million-cell grid simulations natively on CPU & GPU clusters.

## 🚀 Key Performance Features
- **Core Physics**: 3D spatial discretization, Rock, Multi-Phase Fluid thermodynamic tracking, and comprehensive implicit well limits.
- **XLA JIT Native Bindings**: Unrolls analytical mathematical Jacobians flawlessly via `@jax.jit` compiling instantly to pure localized C++ computational matrices natively.
- **SuperLU Sparse Solvers**: Explicitly solves non-linear algebra strictly using Compressed Sparse Row (`csr_matrix`) structures scaling cleanly and circumventing $O(N^3)$ dense memory bounds.
- **Advanced PID Time-stepping**: Integrates rigorous Newton iteration traces dynamically manipulating $\Delta t$, specifically mapping industrial mathematical targets mirroring Eclipse configurations securely.
- **Eclipse/OPM Compatibility**: Directly parses generic `.DATA` input decks, executing cleanly and outputting explicitly structured `.PRT` matrices and `.DBG` tracking blocks achieving flawless mathematical reporting parity globally.

## 📚 Technical Engineering Library

Explore the theoretical mechanics, mathematical execution, and architectural domains driving the simulator.

### 1. Philosophy & Code Architecture
* **[Software Engineering & Domain Modeling](documentation/00_software_engineering.md)**: Details the conceptual mapping between physical geology, structural petrophysics, thermodynamic fluid constraints, and their isolated Object-Oriented class boundaries. Includes the complete system Mermaid UML diagram.
* **[Simulator Software Architecture](documentation/04_simulator_architecture.md)**: Summarizes the decoupled software pipeline flowing from Eclipse legacy deck ingestion, through internal numerical computations, and concluding with standardized reporting pipelines (`.INIT`, `.PRT`, `.DBG`).

### 2. Core Reservoir Physics
* **[Governing Equations](documentation/01_governing_equations.md)**: Formulates the fundamental continuous mass-conservation Partial Differential Equations (PDEs) and Darcy Laws bounding the Black-Oil system.
* **[Conceptual Thermodynamics & PvT State Constraints](documentation/02_conceptual_physics.md)**: Examines how bounded, non-linear fluid phase characteristics act dynamically under pressure. Explains "Dynamic Variable Substitution" for saturated phase transitions ($R_{s}$ states).

### 3. Numerical Mathematics
* **[Numerical Methods & Discretization](documentation/03_numerical_methods.md)**: Details the discrete Fully Implicit Matrix (FIM) solver, Single-Point Upwinding, geometric harmonic transmissibilities, and the performance scaling using **Native Google JAX** analytical evaluation chain-rules.

---

## 🛠️ Installation & CUDA Hardware Setup

This simulator incorporates high-dimensional mathematical backends uniquely tracking massive geometric tensor bounds structurally natively. 

### 1. Initialize Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. General CPU Installation
Install the exact frozen Python version footprints locally guaranteeing stable bounds explicitly:
```bash
pip install -r requirements.txt
```

### 3. Activating NVIDIA GPU Bindings (Critical ⚠️)
Because the structural JAX CUDA-12 evaluation footprint measures exclusively well over 3.5 Gigabytes dynamically, standard systems inherently violently lock up throwing `[Errno 28] No space left on device` mapping standard extractions implicitly onto generic `/tmp` kernel constraints natively.

To permanently unlock the absolute execution execution speed physically constrained computationally over Nvidia arrays, execute your pipeline dynamically tracking caching strictly toward your massive `/home` drive partitions gracefully inherently protecting memory mappings completely:

```bash
# 1. Purge generic pipelines natively isolating limits:
pip cache purge 

# 2. Extract GPU bounds dynamically directly structurally bypassing standard bounds recursively:
mkdir -p /home/$USER/tmp_pip
TMPDIR=/home/$USER/tmp_pip pip install --upgrade "jax[cuda12]"
```

Verify your active bindings executing seamlessly natively inside standard Python dynamically:
```bash
python3 -c "import jax; print(jax.devices())"
```
*(If properly instantiated natively you should universally see `[CudaDevice(id=0)]` elegantly mapped implicitly!)*

---

## ⚡ Running the Simulation Engine

Execute the FIM non-linear array loops actively tracing convergence steps natively bounding sequences limitlessly against the provided testing sample perfectly mapped towards `SPE1CASE1` industrial configurations exclusively exactly.

```bash
source .venv/bin/activate
python3 main.py --input-file data/sample/SPE1CASE1.DATA
```

## 🧪 Industrial Analytical Tracking

As the overarching algorithm dynamically loops continuous evaluation checks iteratively natively, it universally isolates flawlessly exact structural formats mapping inherently towards exactly tracked **Print (.PRT)** sequences dynamically and explicitly **Debug (.DBG)** matrices comprehensively isolated statically alongside your specific executing folder cleanly!

```bash
# Trace output metrics explicitly observing the FIM matrix blocks intrinsically:
cat data/sample/SPE1CASE1.PRT
```

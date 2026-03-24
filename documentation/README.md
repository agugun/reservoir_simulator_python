# 🛢️ Python Reservoir Simulator Documentation

Welcome to the technical documentation for the native Python-based 3D Fully Implicit Reservoir Simulator. 

This simulator achieves **industrial mathematical and binary parity** with legacy reservoir engines like OPM Flow and Eclipse 100. It natively tracks true 3-phase multi-component Black-Oil interactions, processes industrial `.DATA` deck formats, and is structurally optimized using highly advanced **Google JAX** Automatic Differentiation ($O(1)$ analytic Jacobians) for robust non-linear solver stability.

---

## 📚 Documentation Index

Please navigate the structural modules below to explore the theoretical mechanics, mathematical execution, and architectural domains driving the system.

### 1. Philosophy & Code Architecture
* **[Engineering Philosophy & Domain Modeling](00_engineering_philosophy.md)**: Details the conceptual mapping between physical geology, structural petrophysics, thermodynamic fluid constraints, and their isolated Object-Oriented class boundaries globally. Includes the complete system Mermaid UML diagram.
* **[Simulator Software Architecture](04_simulator_architecture.md)**: Summarizes the decoupled software pipeline flowing from Eclipse legacy deck ingestion, through internal numerical computations, and concluding with standardized exact binary and generic ASCII reporting pipelines (`.INIT`, `.UNRST`, `.PRT`, `.DBG`).

### 2. Core Reservoir Physics
* **[Governing Equations](01_governing_equations.md)**: Formulates the fundamental continuous mass-conservation Partial Differential Equations (PDEs) and empirically extended multi-phase Darcy Laws bounding the Black-Oil system over discrete rock control volumes.
* **[Conceptual Thermodynamics & PvT State Constraints](02_conceptual_physics.md)**: Examines how bounded, non-linear fluid phase characteristics continuously act dynamically under fluctuating reservoir pressures. Explains the core implementation limits of "Dynamic Variable Substitution" explicitly addressing dynamically structured saturated phase transitions ($R_{s}$ states).

### 3. Numerical Mathematics
* **[Numerical Methods & Continuous Mathematical Discretization](03_numerical_methods.md)**: Extensively details the continuous structural implementation behind the discrete Fully Implicit Matrix (FIM) solver. Covers Single-Point Phase-Mobility Upwinding, geometric harmonic transmissibilities, limit-bounded Ruiz equilibrations, and specifically highlights the revolutionary scaling performance utilizing Native $O(1)$ Google JAX analytical evaluation chain-rules mapping precision identical analytical non-linear flow partials automatically.

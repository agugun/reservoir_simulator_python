# Conceptual Physics

The simulator mirrors the advanced Black-Oil physical assumptions found specifically in OPM Flow and Eclipse 100 protocols. This involves highly non-linear thermodynamic mapping using standardized table interpolations.

## 1. Thermodynamic PVT Grids

Fluids do not stay static; their volumetric properties dynamically shift based upon immediate Local Cell Pressures ($P$) and Solution Gas Fractions ($R_s$).

### PVDG (Dry Gas)
Calculates purely pressure-dependent behaviors of a free gas phase:
*   **Formation Volume Factor ($B_g$)**: Expands highly non-linearly at low pressures, causing grid instability when $S_g$ flashes.
*   **Viscosity ($\mu_g$)**: Gas viscosity scaling.

### PVTO (Live Oil)
Oil behavior is heavily bound strictly to the `Rs` state.
When oil is _undersaturated_ (Current $R_s$ < Maximum $R_{sat}$ at $P$), all properties transition linearly off the maximum saturation bubble-point nodes governed by undersaturated arrays. When _saturated_ ($R_s = R_{sat}$), oil directly obeys the strictly bounded surface-curve limits.

## 2. Dynamic Variable Substitution

Tracking 3-phase mathematics using fixed variables ($P, S_o, S_g, S_w$) is impossible when tracking miscible systems like Dissolved Gas. We use **Variable Substitution** to map a secondary state vector $Y$:

1.  **Saturated State (`is_sat = True`)**:
    *   Primary unknowns: $P$ (Pressure), $S_g$ (Gas Saturation).
    *   The oil is fully saturated with gas. Therefore, $R_s$ is mathematically strictly equal exactly to $R_{sat}(P)$, letting the solver calculate $S_g > 0$ structurally.
2.  **Undersaturated State (`is_sat = False`)**:
    *   Primary unknowns: $P$ (Pressure), $R_s$ (Solution Gas Ratio).
    *   Because the oil is undersaturated, all gas is completely dissolved. $S_g$ strictly $= 0$.

During the Newton-Raphson iterations, if $S_g$ falls below $0$, the cell triggers an algebraic switch strictly transferring its secondary variable into $R_s$. Likewise, if $R_s$ eclipses $R_{sat}$, it flashes into free gas, switching $Y$ directly to $S_g$.

## 3. Rock Properties & Capillarity

### Relative Permeability
Fluid velocities strongly interact based exclusively on internal cell multi-phase contact surfaces ($S_w, S_g$ tables `SGWFN / SWOFN`). We map values directly tracking continuous arrays interpolating via exact tabular limits.

### Capillary Pressure
The offset in spatial pressures between phase phases ($P_{cow}, P_{cgo}$) tracks hysteresis boundaries natively calculated in interpolation arrays, scaling the structural Darcy gradients significantly upon high structural dips.

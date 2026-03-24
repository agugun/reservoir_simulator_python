# Numerical Methods

The complexity of coupled multi-phase flow requires advanced, stable mathematical methods to navigate immense non-linearity inherent in reservoir mechanics.

## 1. Fully Implicit Finite Volume

The PDEs are converted into discrete algebraic approximations over $K$ finite control volumes using the **Fully Implicit Method (FIM)**. 
All dynamic parameters—Pressures ($P^{n+1}$), Saturations ($S^{n+1}$), and Transmissibilities—are evaluated strictly at the *forward* time step.

While FIM guarantees unconditional numerical stability permitting highly adaptive time-steps ($\Delta t$), it forces the solver to dynamically resolve a massive system of non-linear equations perfectly continuously:
$$ \mathbf{R}(X^{n+1}) = 0 $$

Where $\mathbf{R}$ encapsulates the full Mass Balance residual vector for Oil and Gas structurally across all active blocks.

## 2. Upwinded Phase Transmissibility

Fluid flow $F_{ij}$ between adjacent grid blocks $i$ and $j$ derives from Darcy's Law:
$$ F_{ij, p} = T_{ij} \cdot \lambda_{p, upwind} \cdot \left( \Phi_j - \Phi_i \right) $$

*   **Geometric Transmissibility ($T_{ij}$)**: Calculated structurally by resolving the harmonic average of rock Permeabilities ($K$) mapped alongside the interface Areas ($A$) and absolute grid distances ($\Delta x$).
*   **Phase Mobility ($\lambda_p$)**: $k_{rp} / (\mu_p B_p)$. 
*   **Upwinding Protocol**: To maintain mathematical propagation stability, $\lambda_p$ is selected strictly from the upstream cell driving the potential flow $\Phi$.

## 3. Newton-Raphson & JAX Automatic Differentiation

Because $\mathbf{R}(X^{n+1}) = 0$ is highly non-linear, we linearize the matrix systematically:
$$ \mathbf{J} \cdot \delta X = - \mathbf{R} $$
Where $\mathbf{J}$ is the Jacobian Matrix ($\partial \mathbf{R} / \partial X$).

### Functional Programming & Analytic Derivatives

Classical iterative solvers rely on numerically perturbed Jacobian structures (calculating finite-difference $O(N)$ slopes physically across thousands of structural evaluations limitlessly). This bottleneck natively destroys Python's evaluation speed.

Instead, the entire functional residual array in `simulator.py` is mapped explicitly to use non-mutating arrays handled solely by `jax.numpy`.
Using `jax.jacfwd(self._calc_residuals_jax, argnums=0)` constructs the precisely exact mathematical partial-derivatives automatically leveraging graph-compiled symbolic chain-rules mapped at $O(1)$ constant execution scaling.

### Tikhonov Regularization & Ruiz Equilibration

Because phases frequently dynamically appear or disappear locally (variable phase vanishing), $\mathbf{J}$ inevitably becomes ill-conditioned. 
To navigate sparse inversion limitations, the rows and columns are structurally bounded (Ruiz Equilibration mapping) while matrix diagonals accept hard epsilon-limits (Tikhonov scaling) restricting structural collapse.

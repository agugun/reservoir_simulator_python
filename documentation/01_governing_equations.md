# Governing Equations

This reservoir simulator resolves the highly non-linear dynamics of multi-phase fluid flow in porous media using the **Black-Oil Formulation**. The fundamental physical principle is the conservation of mass applied over a finite control volume.

## 1. Mass Conservation PDEs

For a standard 3-phase system (Oil, Water, Gas), the continuous partial differential equations for isothermal flow in a porous medium are formulated as follows:

### Oil Phase Equation
$$ \frac{\partial}{\partial t} \left( \phi \frac{S_o}{B_o} \right) + \nabla \cdot \left( \frac{1}{B_o} \vec{u}_o \right) - q_o = 0 $$

### Water Phase Equation
$$ \frac{\partial}{\partial t} \left( \phi \frac{S_w}{B_w} \right) + \nabla \cdot \left( \frac{1}{B_w} \vec{u}_w \right) - q_w = 0 $$

### Gas Phase Equation
$$ \frac{\partial}{\partial t} \left( \phi \frac{S_g}{B_g} + \phi \frac{R_s S_o}{B_o} \right) + \nabla \cdot \left( \frac{1}{B_g} \vec{u}_g + \frac{R_s}{B_o} \vec{u}_o \right) - q_g = 0 $$

Where:
*   $\phi$ : Rock porosity (pressure-dependent).
*   $S_p$ : Phase saturation for $p \in \{o, w, g\}$.
*   $B_p$ : Phase Formation Volume Factor ($FVF$).
*   $R_s$ : Solution Gas-Oil ratio (Dissolved gas in the oleic phase).
*   $\vec{u}_p$ : Darcy phase velocity vector.
*   $q_p$ : Well source/sink mass flow rate per unit volume.

## 2. Darcy's Law

The phase velocities $\vec{u}_p$ are calculated explicitly through the empirical extension of Darcy's Law for multi-phase porous media flow:

$$ \vec{u}_p = - \mathbf{K} \frac{k_{rp}}{\mu_p} \left( \nabla P_p - \gamma_p \nabla Z \right) $$

Where:
*   $\mathbf{K}$ : Absolute permeability tensor.
*   $k_{rp}$ : Phase relative permeability ($k_{ro}, k_{rw}, k_{rg}$).
*   $\mu_p$ : Phase viscosity.
*   $P_p$ : Phase pressure ($P_o = P, P_w = P - P_{cow}, P_g = P + P_{cgo}$).
*   $\gamma_p$ : Specific phase gravity gradient ($\rho_p g$).
*   $Z$ : Cell reference depth (elevation potential).

## 3. The Algebraic Summation Constraint

Because the pore volume must always be completely saturated with fluid, the sum of all phase saturations mathematically must equal unity:

$$ S_o + S_w + S_g = 1 $$

By resolving the non-linear inter-dependencies introduced by $B_o(P, R_s)$ and $k_{rp}(S_p)$, the solver iteratively locks global grid values ensuring exactly $0$ mass-accumulation error over $\Delta t$.

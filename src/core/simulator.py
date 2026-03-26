import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .model import ReservoirModel
import jax
jax.config.update("jax_enable_x64", True)

class Simulator:
    """Core reservoir simulator engine."""
    
    def __init__(self, model: ReservoirModel):
        self.model = model
        self.time = 0.0
        self.setup_transmissibility()

    def setup_transmissibility(self):
        """
        Calculate inter-block transmissibilities in x, y, and z directions.
        Uses harmonic averaging for permeability.
        """
        grid = self.model.grid
        rock = self.model.rock
        
        # Calculate pore volumes (convert ft3 to bbl for consistency with STB units)
        self.pore_volume = (grid.get_cell_volume() * rock.porosity) / 5.61458
        
        # Pre-calculate Well Indices (PI)
        self.well_indices = {}
        for well in self.model.wells:
            self.well_indices[well.name] = self.calculate_well_index(well)

        # Helper lambda for harmonic average
        def harmonic_avg(k1, k2, d1, d2):
            # To avoid division by zero, use a small epsilon where k=0
            eps = 1e-15
            k1 = np.where(k1 == 0, eps, k1)
            k2 = np.where(k2 == 0, eps, k2)
            return (d1 + d2) / ((d1 / k1) + (d2 / k2))

        # Transmissibility in X direction (size: nx-1, ny, nz)
        self.Tx = np.zeros((grid.nx - 1, grid.ny, grid.nz))
        if grid.nx > 1:
            # Distance from cell centers: (dx[i] + dx[i+1]) / 2
            d1 = grid.dx[:-1, :, :] / 2.0
            d2 = grid.dx[1:, :, :] / 2.0
            k_avg_x = harmonic_avg(rock.perm_x[:-1, :, :], rock.perm_x[1:, :, :], d1, d2)
            
            # Area = dy * dz (Average area of the two cells)
            area_x = (grid.dy[:-1, :, :] * grid.dz[:-1, :, :] + 
                      grid.dy[1:, :, :] * grid.dz[1:, :, :]) / 2.0
            
            # Tx = 0 if either cell is inactive
            activity_x = grid.actnum[:-1, :, :] * grid.actnum[1:, :, :]
            self.Tx = 0.001127 * k_avg_x * area_x / (d1 + d2) * activity_x

        # Transmissibility in Y direction (size: nx, ny-1, nz)
        self.Ty = np.zeros((grid.nx, grid.ny - 1, grid.nz))
        if grid.ny > 1:
            d1 = grid.dy[:, :-1, :] / 2.0
            d2 = grid.dy[:, 1:, :] / 2.0
            k_avg_y = harmonic_avg(rock.perm_y[:, :-1, :], rock.perm_y[:, 1:, :], d1, d2)
            
            area_y = (grid.dx[:, :-1, :] * grid.dz[:, :-1, :] + 
                      grid.dx[:, 1:, :] * grid.dz[:, 1:, :]) / 2.0
            
            activity_y = grid.actnum[:, :-1, :] * grid.actnum[:, 1:, :]
            self.Ty = 0.001127 * k_avg_y * area_y / (d1 + d2) * activity_y

        # Transmissibility in Z direction (size: nx, ny, nz-1)
        self.Tz = np.zeros((grid.nx, grid.ny, grid.nz - 1))
        if grid.nz > 1:
            d1 = grid.dz[:, :, :-1] / 2.0
            d2 = grid.dz[:, :, 1:] / 2.0
            k_avg_z = harmonic_avg(rock.perm_z[:, :, :-1], rock.perm_z[:, :, 1:], d1, d2)
            area_z = (grid.dx[:, :, :-1] * grid.dy[:, :, :-1] + 
                      grid.dx[:, :, 1:] * grid.dy[:, :, 1:]) / 2.0
            
            activity_z = grid.actnum[:, :, :-1] * grid.actnum[:, :, 1:]
            self.Tz = 0.001127 * k_avg_z * area_z / (d1 + d2) * activity_z
            
        self._compile_jax_functions()

    def _compile_jax_functions(self):
        import jax
        print("Compiling JAX XLA Graphs for O(1) step execution...")
        
        # A pure functional wrapper for the residual calculation isolating the physical solver natively
        def residual_fn(p, Y, is_sat, p_old, sg_old, rs_old, dt):
            return self._calc_residuals_jax(p, Y, is_sat, p_old, sg_old, rs_old, dt)
            
        self._jitted_residual = jax.jit(residual_fn)
        # JIT the jacobian function explicitly with has_aux=True
        self._jitted_jacobian = jax.jit(jax.jacfwd(residual_fn, argnums=(0, 1), has_aux=True))

    def calculate_well_index(self, well) -> float:
        """
        Calculates the Productivity Index (PI) using Peaceman's formula.
        Returns PI in STB/d/psi * cP (needs division by viscosity later).
        """
        i, j, k = well.location
        grid = self.model.grid
        rock = self.model.rock
        
        kx = rock.perm_x[i, j, k]
        ky = rock.perm_y[i, j, k]
        kz = rock.perm_z[i, j, k]
        
        dx = grid.dx[i, j, k]
        dy = grid.dy[i, j, k]
        dz = grid.dz[i, j, k]
        
        # Peaceman radius ro
        # ro = 0.28 * sqrt( (ky/kx)^0.5 * dx^2 + (kx/ky)^0.5 * dy^2 ) / ( (ky/kx)^0.25 + (kx/ky)^0.25 )
        # Assuming isotropic kx, ky for simplicity in numerator
        ratio_yx = np.sqrt(ky / kx) if kx > 0 else 1.0
        ratio_xy = np.sqrt(kx / ky) if ky > 0 else 1.0
        
        num = np.sqrt(ratio_yx * dx**2 + ratio_xy * dy**2)
        den = np.power(ratio_yx, 0.5) + np.power(ratio_xy, 0.5)
        ro = 0.28 * num / den
        
        # WI = 2 * pi * k * h / (ln(ro/rw) + skin)
        # Using conversion factor 0.001127 for Darcy's law in field units
        k_eff = np.sqrt(kx * ky)
        wi = (2 * np.pi * 0.001127 * k_eff * dz) / (np.log(ro / well.well_radius) + well.skin)
        
        return max(0.0, float(wi))
    def _calc_residuals_jax(self, p, Y, is_sat, p_old, sg_old, rs_old, dt):
        import jax.numpy as jnp
        
        rsat = self.model.fluid.get_rsat(p)
        sg = jnp.where(is_sat, Y, 0.0)
        rs = jnp.where(is_sat, rsat, Y)
        
        grid = self.model.grid
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        N = grid.total_cells
        vp_base = jnp.array(self.pore_volume.flatten(order='F'))
        
        cr = getattr(self.model.rock, 'compressibility', 3e-6)
        pref = 14.7
        
        vp = vp_base * (1.0 + cr * (p - pref))
        vp_old = vp_base * (1.0 + cr * (p_old - pref))
        
        sw_conn = 0.12
        so = jnp.clip(1.0 - sw_conn - sg, 0.0, 1.0)
        so_old = jnp.clip(1.0 - sw_conn - sg_old, 0.0, 1.0)
        
        bo, mu_o = self.model.fluid.get_oil_props(p, rs)
        bo_old, _ = self.model.fluid.get_oil_props(p_old, rs_old)
        
        bg, mu_g = self.model.fluid.get_gas_props(p)
        bg_old, _ = self.model.fluid.get_gas_props(p_old)
        
        if hasattr(self.model.rock, 'sgof') and self.model.rock.sgof is not None:
            krg = jnp.interp(sg, jnp.array(self.model.rock.sgof['sg']), jnp.array(self.model.rock.sgof['krg']))
            kro = jnp.interp(sg, jnp.array(self.model.rock.sgof['sg']), jnp.array(self.model.rock.sgof['krog']))
        else:
            kro = jnp.clip(so / (1.0 - sw_conn), 0.0, 1.0)**2
            krg = jnp.clip(sg / (1.0 - sw_conn), 0.0, 1.0)**2
        
        lam_o = kro / (mu_o * bo)
        lam_g = krg / (mu_g * bg)
        
        R_o = (1.0 / dt) * (vp * so / bo - vp_old * so_old / bo_old)
        R_g = (1.0 / dt) * (vp * (sg / bg + rs * so / bo) - vp_old * (sg_old / bg_old + rs_old * so_old / bo_old))
        
        p_3d = p.reshape(grid.dimensions, order='F')
        def van_leer(a, b):
            # Limiter phi(r) where r = b/a. Simplified to phi(a, b) = (a*|b| + |a|*b) / (|a| + |b|)
            # Handles zero-gradient cases safely.
            abs_a, abs_b = jnp.abs(a), jnp.abs(b)
            return jnp.where(abs_a + abs_b > 1e-20, (a * abs_b + abs_a * b) / (abs_a + abs_b), 0.0)

        # Potentials and Fluxes (F-order Reshape Bitwise Sync)
        p_3d = p.reshape(grid.dimensions, order='F')
        lam_o_3d = lam_o.reshape(grid.dimensions, order='F')
        lam_g_3d = lam_g.reshape(grid.dimensions, order='F')
        rs_3d = rs.reshape(grid.dimensions, order='F')
        z_3d = jnp.array(grid.z_centers).reshape(grid.dimensions, order='F')
        
        rho_o_surf = getattr(self.model.fluid, 'density_oil', 53.66)
        rho_g_surf = getattr(self.model.fluid, 'density_gas', 0.0533)
        # Unit conversion: 1 MSCF = 1000 SCF, 1 RB = 5.61458 ft3 -> 1000/5.61458 = 178.1076
        f_surf = 1000.0 / 5.61458
        gam_o_3d = ((rho_o_surf + rs * rho_g_surf * f_surf) / bo * 0.006944).reshape(grid.dimensions, order='F')
        gam_g_3d = (rho_g_surf * f_surf / bg * 0.006944).reshape(grid.dimensions, order='F')
        
        flow_o = jnp.zeros(grid.dimensions)
        flow_g = jnp.zeros(grid.dimensions)
        
        Tx = jnp.array(self.Tx)
        Ty = jnp.array(self.Ty)
        Tz = jnp.array(self.Tz)
        
        if nx > 1:
            dp_x = p_3d[:-1,:,:] - p_3d[1:,:,:]
            dz_x = z_3d[:-1,:,:] - z_3d[1:,:,:]
            g_o_x = 0.5 * (gam_o_3d[:-1,:,:] + gam_o_3d[1:,:,:])
            g_g_x = 0.5 * (gam_g_3d[:-1,:,:] + gam_g_3d[1:,:,:])
            dPhi_o_x = dp_x - g_o_x * dz_x
            dPhi_g_x = dp_x - g_g_x * dz_x
            up_o, up_g = dPhi_o_x >= 0, dPhi_g_x >= 0
            
            # TVD Reconstruction (X)
            def get_tvd(val_3d, up_mask):
                v_up = jnp.where(up_mask, val_3d[:-1,:,:], val_3d[1:,:,:])
                v_down = jnp.where(up_mask, val_3d[1:,:,:], val_3d[:-1,:,:])
                # Up-upstream depends on direction and boundaries
                v_ups_p = jnp.concatenate([jnp.expand_dims(val_3d[0,:,:], 0), val_3d[:-2,:,:]], axis=0) # for up=True
                v_ups_m = jnp.concatenate([val_3d[2:,:,:], jnp.expand_dims(val_3d[-1,:,:], 0)], axis=0) # for up=False
                v_ups = jnp.where(up_mask, v_ups_p, v_ups_m)
                
                # Revert to standard Upwind for diagnostic
                return v_up

            lo_face = get_tvd(lam_o_3d, up_o)
            lg_face = get_tvd(lam_g_3d, up_g)
            rs_face = get_tvd(rs_3d, up_o)
            
            flux_o = Tx * lo_face * dPhi_o_x
            flux_g = Tx * (lg_face * dPhi_g_x + rs_face * lo_face * dPhi_o_x)
            
            flow_o = flow_o.at[:-1,:,:].add(flux_o); flow_o = flow_o.at[1:,:,:].add(-flux_o)
            flow_g = flow_g.at[:-1,:,:].add(flux_g); flow_g = flow_g.at[1:,:,:].add(-flux_g)
            
        if ny > 1:
            dp_y = p_3d[:,:-1,:] - p_3d[:,1:,:]
            dz_y = z_3d[:,:-1,:] - z_3d[:,1:,:]
            g_o_y = 0.5 * (gam_o_3d[:,:-1,:] + gam_o_3d[:,1:,:])
            g_g_y = 0.5 * (gam_g_3d[:,:-1,:] + gam_g_3d[:,1:,:])
            dPhi_o_y = dp_y - g_o_y * dz_y
            dPhi_g_y = dp_y - g_g_y * dz_y
            up_o, up_g = dPhi_o_y >= 0, dPhi_g_y >= 0
            
            def get_tvd_y(val_3d, up_mask):
                v_up = jnp.where(up_mask, val_3d[:,:-1,:], val_3d[:,1:,:])
                # Revert to standard Upwind for diagnostic
                return v_up

            lo_face = get_tvd_y(lam_o_3d, up_o)
            lg_face = get_tvd_y(lam_g_3d, up_g)
            rs_face = get_tvd_y(rs_3d, up_o)
            
            flux_o = Ty * lo_face * dPhi_o_y
            flux_g = Ty * (lg_face * dPhi_g_y + rs_face * lo_face * dPhi_o_y)
            
            flow_o = flow_o.at[:,:-1,:].add(flux_o); flow_o = flow_o.at[:,1:,:].add(-flux_o)
            flow_g = flow_g.at[:,:-1,:].add(flux_g); flow_g = flow_g.at[:,1:,:].add(-flux_g)
            
        if nz > 1:
            dp_z = p_3d[:,:,:-1] - p_3d[:,:,1:]
            dz_z = z_3d[:,:,:-1] - z_3d[:,:,1:]
            g_o_z = 0.5 * (gam_o_3d[:,:,:-1] + gam_o_3d[:,:,1:])
            g_g_z = 0.5 * (gam_g_3d[:,:,:-1] + gam_g_3d[:,:,1:])
            dPhi_o_z = dp_z - g_o_z * dz_z
            dPhi_g_z = dp_z - g_g_z * dz_z
            up_o, up_g = dPhi_o_z >= 0, dPhi_g_z >= 0
            
            def get_tvd_z(val_3d, up_mask):
                v_up = jnp.where(up_mask, val_3d[:,:,:-1], val_3d[:,:,1:])
                # Revert to standard Upwind for diagnostic
                return v_up

            lo_face = get_tvd_z(lam_o_3d, up_o)
            lg_face = get_tvd_z(lam_g_3d, up_g)
            rs_face = get_tvd_z(rs_3d, up_o)
            
            flux_o = Tz * lo_face * dPhi_o_z
            flux_g = Tz * (lg_face * dPhi_g_z + rs_face * lo_face * dPhi_o_z)
            
            flow_o = flow_o.at[:,:,:-1].add(flux_o); flow_o = flow_o.at[:,:,1:].add(-flux_o)
            flow_g = flow_g.at[:,:,:-1].add(flux_g); flow_g = flow_g.at[:,:,1:].add(-flux_g)
            
        R_o = R_o + flow_o.flatten(order='F')
        R_g = R_g + flow_g.flatten(order='F')
        
        for well in self.model.wells:
            w_idx = well.location[0] + well.location[1]*nx + well.location[2]*nx*ny
            wi = float(self.well_indices[well.name])
            
            lo_w = lam_o[w_idx]; lg_w = lam_g[w_idx]; lt_w = jnp.maximum(lo_w + lg_w, 1e-12)
            
            is_prod = (well.rate < 0) if well.rate else ('PROD' in well.name.upper())
            req_rate = abs(well.rate) if well.rate else 0.0
            
            # Producer Logic: decoupled phase rates
            # lo_w = kro / (mu_o * Bo), so wi * lo_w * dp is already in STB/day
            q_o_pot_stb = wi * lo_w * (p[w_idx] - well.bhp) if well.bhp is not None else 1e12
            # lg_w = krg / (mu_g * Bg), so wi * lg_w * dp is already in MSCF/day
            q_g_pot_mscf = wi * lg_w * (p[w_idx] - well.bhp) if well.bhp is not None else 0.0
            
            # Rate limited if orat (req_rate) is exceeded
            is_limited = (req_rate > 0) & (q_o_pot_stb > req_rate) & (q_o_pot_stb > 1e-6)
            scaling = jnp.where(is_limited, req_rate / jnp.maximum(q_o_pot_stb, 1e-9), 1.0)
            
            # Final phase rates for R calculation
            q_o_prod = jnp.where(is_prod, q_o_pot_stb * scaling, 0.0)
            q_g_prod = jnp.where(is_prod, q_g_pot_mscf * scaling, 0.0)
            
            # Injector Logic (if not is_prod)
            # Use endpoint mobility for injector (krg=1.0)
            mob_g_inj = 1.0 / (mu_g[w_idx] * bg[w_idx])
            q_pot_g_inj = wi * mob_g_inj * (well.bhp - p[w_idx]) if well.bhp is not None else 1e12
            q_g_inj_scaled = jnp.clip(q_pot_g_inj, 0, req_rate) if well.rate else q_pot_g_inj
            
            q_o = q_o_prod
            q_free_g = jnp.where(is_prod, q_g_prod, -q_g_inj_scaled)
            
            q_g = q_free_g + rs[w_idx] * q_o
            
            R_o = R_o.at[w_idx].add(q_o)
            R_g = R_g.at[w_idx].add(q_g)
            
        act = jnp.array(grid.actnum.flatten(order='F'))
        R_o = jnp.where(act == 0, 0.0, R_o)
        R_g = jnp.where(act == 0, 0.0, R_g)
        
        R = jnp.empty(2*N)
        R = R.at[0::2].set(R_o)
        R = R.at[1::2].set(R_g)
        
        # CFL Calculation: f_cfl = sum(|q|) / Vp
        # q is flux (RB/day), Vp is pore volume (RB)
        q_sum = (jnp.abs(flow_o) + jnp.abs(flow_g))
        throughput = q_sum.flatten(order='F') / self.pore_volume.flatten(order='F')
        cfl_max = jnp.max(throughput) * dt
        
        return R, cfl_max
        
    def _build_jacobian_fim(self, p, Y, is_sat, p_old, sg_old, rs_old, dt):
        import jax.numpy as jnp
        N = len(p)
        
        p_jax = jnp.array(p, dtype=jnp.float64)
        Y_jax = jnp.array(Y, dtype=jnp.float64)
        is_sat_jax = jnp.array(is_sat, dtype=jnp.bool_)
        p_old_jax = jnp.array(p_old, dtype=jnp.float64)
        sg_old_jax = jnp.array(sg_old, dtype=jnp.float64)
        rs_old_jax = jnp.array(rs_old, dtype=jnp.float64)
        dt_jax = jnp.float64(dt)
        
        # Execute the pre-compiled XLA cache bypassing python interpreting limits inherently
        (J_p, J_y), cfl_max = self._jitted_jacobian(p_jax, Y_jax, is_sat_jax, p_old_jax, sg_old_jax, rs_old_jax, dt_jax)
        R_base, _ = self._jitted_residual(p_jax, Y_jax, is_sat_jax, p_old_jax, sg_old_jax, rs_old_jax, dt_jax)
        
        # Interleave analytic Jacobian segments into explicit 2N block structure
        J = np.zeros((2*N, 2*N))
        J[:, 0::2] = np.array(J_p)
        J[:, 1::2] = np.array(J_y)
        
        act = self.model.grid.actnum.flatten(order='F')
        inactive = np.where(act == 0)[0]
        if len(inactive) > 0:
            J[2*inactive, :] = 0.0
            J[2*inactive+1, :] = 0.0
            J[2*inactive, 2*inactive] = 1.0
            J[2*inactive+1, 2*inactive+1] = 1.0
            
        return J, np.array(R_base), float(cfl_max)

    def step_fim(self, dt: float, max_iter: int = 15, tol: float = 1.0, report_writer=None) -> np.ndarray:
        p_old = self.model.pressure.flatten(order='F')
        sg_old = self.model.sgas.flatten(order='F')
        rs_old = self.model.rs.flatten(order='F')
        
        p = p_old.copy()
        
        # Determine initial Variable Substitution state
        rsat_init = self.model.fluid.get_rsat(p)
        is_sat = sg_old > 0
        to_sat = (~is_sat) & (rs_old >= rsat_init)
        is_sat[to_sat] = True
        
        # Y is our secondary variable array
        Y = np.where(is_sat, sg_old, rs_old)
        
        act = self.model.grid.actnum.flatten(order='F')
        
        if report_writer:
            report_writer.log_newton_outer()
            
        for iteration in range(max_iter):
            J, R, cfl_val = self._build_jacobian_fim(p, Y, is_sat, p_old, sg_old, rs_old, dt)
            if iteration == 0:
                print(f"   [CFL Check] Current CFL = {cfl_val:.4f} (Target < 1.0)")
            
            active_R = np.empty(0)
            if np.any(act == 1):
                active_R = R[np.repeat(act == 1, 2)]
            
            if report_writer:
                mb_o = np.max(np.abs(active_R[0::2])) if len(active_R) > 0 else 0.0
                mb_g = np.max(np.abs(active_R[1::2])) if len(active_R) > 0 else 0.0
                report_writer.log_newton_iter(iteration, mb_o, 0.0, mb_g, mb_o*0.1, 0.0, mb_g*0.1)
                
            error = np.max(np.abs(active_R)) if len(active_R) > 0 else 0.0
            if error < tol:
                if report_writer:
                    report_writer.log_newton_summary(iteration + 1, dt)
                self.last_iterations = iteration + 1
                break
                
            r_max = np.abs(J).max(axis=1)
            r_max[r_max < 1e-12] = 1.0
            J_scaled = J / r_max[:, None]
            R_scaled = R / r_max
            
            c_max = np.abs(J_scaled).max(axis=0)
            c_max[c_max < 1e-12] = 1.0
            J_scaled = J_scaled / c_max
            
            np.fill_diagonal(J_scaled, J_scaled.diagonal() + 1e-12)
            
            try:
                import scipy.sparse as sp
                import scipy.sparse.linalg as spla
                # Convert explicitly to Compressed Sparse Row format eliminating mathematical blocks of absolute zero
                J_sparse = sp.csr_matrix(J_scaled)
                dX_scaled = spla.spsolve(J_sparse, -R_scaled)
                dX = dX_scaled / c_max
            except Exception as e:
                print(f"Jacobian sparse inversion failed: {e}")
                break
                
            dp = dX[0::2]
            dy = dX[1::2]
            
            # Appleyard Step Damping (Smooth limiting to prevent overshoots)
            dp = dp / (1.0 + np.abs(dp) / 500.0)
            dy = dy / (1.0 + np.abs(dy) / 0.10)
            
            p[act == 1] += dp[act == 1]
            Y[act == 1] += dy[act == 1]
            
            # Post-iteration State Switching Evaluation
            rsat = self.model.fluid.get_rsat(p)
            
            # Undersaturated dropping below bubble point?
            to_sat = (~is_sat) & (Y >= rsat) & (act == 1)
            if np.any(to_sat):
                is_sat[to_sat] = True
                Y[to_sat] = 1e-4
                
            # Saturated gas vanishes?
            to_und = is_sat & (Y < 0.0) & (act == 1)
            if np.any(to_und):
                is_sat[to_und] = False
                Y[to_und] = rsat[to_und]
                
            # Clip bounds logically
            Y[is_sat] = np.clip(Y[is_sat], 0.0, 0.88)
            Y[~is_sat] = np.clip(Y[~is_sat], 0.0, rsat[~is_sat])
            
        if error >= tol:
            raise RuntimeError(f"FIM Newton Solver divergence after {max_iter} iterations (Residual: {error:.4f})")
            
        rsat_final = self.model.fluid.get_rsat(p)
        self.model.pressure = p.reshape(self.model.grid.dimensions, order='F')
        self.model.sgas = np.where(is_sat, Y, 0.0).reshape(self.model.grid.dimensions, order='F')
        self.model.rs = np.where(is_sat, rsat_final, Y).reshape(self.model.grid.dimensions, order='F')
        
        self.time += dt
        return self.model.pressure

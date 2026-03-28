import numpy as np
import jax.numpy as jnp

class PVDGTable:
    def __init__(self, data):
        # data is [p1, bg1, mu1, p2, bg2, mu2...]
        self.p = jnp.array(data[0::3])
        self.bg = jnp.array(data[1::3])
        self.mu = jnp.array(data[2::3])
        
    def evaluate(self, p):
        # Linearly interpolate Expansion Factor Eg = 1/Bg which is more linear with P
        eg = jnp.interp(p, self.p, 1.0 / jnp.maximum(self.bg, 1e-12))
        bg = 1.0 / jnp.maximum(eg, 1e-12)
        mu = jnp.interp(p, self.p, self.mu)
        return bg, mu

class PVTOTable:
    def __init__(self, rs_nodes):
        # rs_nodes is a list: [{'rs': rs, 'data': [pbub, bo_bub, mu_bub, p1, bo1, mu1...]}]
        self.rs_nodes = sorted(rs_nodes, key=lambda n: n['rs'])
        self.rs_arr = jnp.array([n['rs'] for n in self.rs_nodes])
        self.pbub_arr = jnp.array([n['data'][0] for n in self.rs_nodes])
        
        self.bub_bo = jnp.array([n['data'][1] for n in self.rs_nodes])
        self.bub_mu = jnp.array([n['data'][2] for n in self.rs_nodes])
        
        # Store undersaturated lines and precompute slopes
        self.under_lines = []
        slopes_bo = np.zeros(len(self.rs_nodes))
        slopes_mu = np.zeros(len(self.rs_nodes))
        
        for i, n in enumerate(self.rs_nodes):
            d = n['data']
            p_arr = jnp.array(d[0::3])
            bo_arr = jnp.array(d[1::3])
            mu_arr = jnp.array(d[2::3])
            self.under_lines.append({'p': p_arr, 'bo': bo_arr, 'mu': mu_arr})
            
            if len(p_arr) > 1:
                dp_val = p_arr[-1] - p_arr[0]
                slopes_bo[i] = (bo_arr[-1] - bo_arr[0]) / dp_val if dp_val > 0 else 0
                slopes_mu[i] = (mu_arr[-1] - mu_arr[0]) / dp_val if dp_val > 0 else 0
                
        self.slopes_bo = jnp.array(slopes_bo)
        self.slopes_mu = jnp.array(slopes_mu)
            
    def get_rsat(self, p):
        return jnp.interp(p, self.pbub_arr, self.rs_arr)
        
    def evaluate(self, p, rs):
        pbub = jnp.interp(rs, self.rs_arr, self.pbub_arr)
        bo_bub = jnp.interp(rs, self.rs_arr, self.bub_bo)
        mu_bub = jnp.interp(rs, self.rs_arr, self.bub_mu)
        
        # JAX arrays for slopes
        slope_bo_rs = jnp.interp(rs, self.rs_arr, self.slopes_bo)
        slope_mu_rs = jnp.interp(rs, self.rs_arr, self.slopes_mu)
        
        # Undersaturated correction (Exponential/Log-linear matching OPM)
        dp_under = jnp.maximum(0.0, p - pbub)
        
        # bo = bo_bub * exp(c_o * dp_under) where c_o = slope / bo_bub
        # For small x, exp(x) ~ 1 + x, matching linear if needed, but more accurate for large dp
        bo = bo_bub * jnp.exp((slope_bo_rs / jnp.maximum(bo_bub, 1e-12)) * dp_under)
        mu = mu_bub * jnp.exp((slope_mu_rs / jnp.maximum(mu_bub, 1e-12)) * dp_under)
        
        return bo, mu

class Fluid:
    """Represents the multi-phase fluid thermodynamic properties."""
    def __init__(self, pvto_table=None, pvdg_table=None, density_oil=53.66, density_gas=0.0533, compressibility=1e-5):
        self.pvto = pvto_table
        self.pvdg = pvdg_table
        self.density_oil = density_oil
        self.density_gas = density_gas
        self.compressibility = compressibility

    def get_oil_props(self, p, rs):
        if self.pvto:
            return self.pvto.evaluate(p, rs)
        return 1.2, 1.0 # fallback Bo=1.2, Mu=1.0

    def get_gas_props(self, p):
        if self.pvdg:
            return self.pvdg.evaluate(p)
        bg = 14.7 / jnp.clip(p, 14.7, 10000.0)
        return bg, 0.02

    def get_rsat(self, p):
        if self.pvto:
            return self.pvto.get_rsat(p)
        return jnp.zeros_like(p)

import numpy as np
import os
from opm.io import Parser
from ..core import Grid, Rock, Fluid, ReservoirModel, Well

class EclipseParser:
    """Parses Eclipse .DATA files into ReservoirModel objects using opm.io."""
    
    def __init__(self, filename: str):
        self.filename = filename
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Eclipse deck file not found: {filename}")
            
        # Basic check for common Eclipse extensions
        valid_exts = ('.DATA', '.data', '.INC', '.inc')
        if not filename.lower().endswith(valid_exts):
            print(f"Warning: File '{filename}' does not have a standard Eclipse extension (.DATA or .INC).")

        self.parser = Parser()
        try:
            self.deck = self.parser.parse(filename)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to parse Eclipse deck '{filename}'.\n"
                f"Error from OPM Parser: {e}\n"
                "Please ensure the file is a valid Eclipse .DATA deck."
            ) from e
        
    def build_model(self) -> ReservoirModel:
        """Constructs a ReservoirModel from the parsed deck."""
        grid = self._parse_grid()
        rock = self._parse_rock(grid.dimensions)
        fluid = self._parse_fluid()
        
        # Initial pressure from SOLUTION section
        pressure = self._parse_pressure(grid, fluid)
        rs = self._parse_rs(grid)
        
        # Start Date from SCHEDULE section
        start_date = self._parse_start_date()

        # Wells from SCHEDULE section
        wells = self._parse_wells()
        
        return ReservoirModel(grid, rock, fluid, pressure, wells=wells, rs=rs, start_date=start_date)

    def _parse_start_date(self) -> str:
        """Parses the START keyword to get simulation origin date."""
        if 'START' in self.deck:
            start_rec = self.deck['START'][0]
            day = self._get_val(start_rec[0])
            month = start_rec[1].get_str(0)
            year = self._get_val(start_rec[2])
            # Return standardized format: DD-MMM-YYYY
            return f"{int(day):02d}-{month.upper()}-{int(year)}"
        return "01-JAN-2015"

    def _get_val(self, item, index=0):
        """Helper to get value from DeckItem regardless of it being int, double, or UDA."""
        if item.is_int():
            return item.get_int(index)
        if hasattr(item, 'is_uda') and item.is_uda():
            # UDA objects have get_double()
            return item.get_uda(index).get_double()
        # get_raw(index) seems to work for doubles
        return item.get_raw(index)

    def _parse_grid(self) -> Grid:
        """Parses DIMENS/SPECGRID and DX/DY/DZ arrays."""
        # Get dimensions
        if 'SPECGRID' in self.deck:
            spec = self.deck['SPECGRID'][0]
            nx = self._get_val(spec[0])
            ny = self._get_val(spec[1])
            nz = self._get_val(spec[2])
        elif 'DIMENS' in self.deck:
            dim_kw = self.deck['DIMENS']
            nx = self._get_val(dim_kw[0][0])
            ny = self._get_val(dim_kw[0][1])
            nz = self._get_val(dim_kw[0][2])
        else:
            raise ValueError("No DIMENS or SPECGRID keyword found in deck.")
        
        # Get spacing
        dx_data = self.deck['DX'][0][0].get_raw_data_list()
        dy_data = self.deck['DY'][0][0].get_raw_data_list()
        dz_data = self.deck['DZ'][0][0].get_raw_data_list()
        
        # Get ACTNUM if present
        actnum = None
        if 'ACTNUM' in self.deck:
            act_data = self.deck['ACTNUM'][0][0].get_raw_data_list()
            actnum = np.array(act_data, dtype=np.int32).reshape((nx, ny, nz), order='F')
            
        # Get TOPS if present
        top_depth = None
        if 'TOPS' in self.deck:
            tops_data = np.array(self.deck['TOPS'][0][0].get_raw_data_list(), dtype=np.float32)
            # Standard SPE1 has TOPS for top layer only.
            # We must accumulate DZ to get top depths of all cells.
            dz_3d = np.array(dz_data).reshape((nx, ny, nz), order='F')
            top_3d = np.zeros((nx, ny, nz), dtype=np.float32)
            
            # Initial top from TOPS keyword (first layer)
            top_3d[:, :, 0] = tops_data.reshape((nx, ny), order='F')
            
            # Layers 2...N: top[k] = top[k-1] + dz[k-1]
            for k in range(1, nz):
                top_3d[:, :, k] = top_3d[:, :, k-1] + dz_3d[:, :, k-1]
            
            top_depth = top_3d
        
        return Grid(nx, ny, nz, dx_data, dy_data, dz_data, actnum=actnum, top_depth=top_depth)

    def _parse_rock(self, dims) -> Rock:
        """Parses PORO, PERMX, PERMY, PERMZ."""
        poro = np.array(self.deck['PORO'][0][0].get_raw_data_list()).reshape(dims, order='F')
        permx = np.array(self.deck['PERMX'][0][0].get_raw_data_list()).reshape(dims, order='F')
        permy = np.array(self.deck['PERMY'][0][0].get_raw_data_list()).reshape(dims, order='F')
        permz = np.array(self.deck['PERMZ'][0][0].get_raw_data_list()).reshape(dims, order='F')
        
        # Compressibility is often in PROPS/ROCK or here in this simplified parser we might hardcode or look for ROCK
        # For now, if ROCK keyword is missing, default to 3e-6
        compressibility = 3e-6
        if 'ROCK' in self.deck:
            # ROCK usually: Pref Compressibility
            compressibility = self._get_val(self.deck['ROCK'][0][1])
            
        # Parse SGOF
        sgof_table = None
        krg_max = 1.0 # Default fallback
        if 'SGOF' in self.deck:
            sgof_data = self.deck['SGOF'][0][0].get_raw_data_list()
            sgof_table = {
                'sg': np.array(sgof_data[0::4]),
                'krg': np.array(sgof_data[1::4]),
                'krog': np.array(sgof_data[2::4])
            }
            krg_max = sgof_table['krg'][-1]
        
        # Parse SWOF
        sw_conn = 0.12 # Default
        if 'SWOF' in self.deck:
            swof_data = self.deck['SWOF'][0][0].get_raw_data_list()
            sw_conn = swof_data[0] # First value is Swc
            
        # Handle MULTZ (Transmissibility Multipliers in Z direction)
        # In this simplified implementation, we modify PERMZ directly to emulate MULTZ.
        if 'MULTZ' in self.deck:
            for row in self.deck['MULTZ']:
                try:
                    # MULTZ: I1 I2 J1 J2 K1 K2 MULT
                    # OPM Parser might group all items into the first item's data list
                    items = row[0].get_raw_data_list()
                    if len(items) < 7:
                        print(f"Warning: MULTZ record has too few items ({len(items)})")
                        continue
                    
                    i1 = int(items[0]) - 1
                    i2 = int(items[1]) - 1
                    j1 = int(items[2]) - 1
                    j2 = int(items[3]) - 1
                    k1 = int(items[4]) - 1
                    k2 = int(items[5]) - 1
                    mult = float(items[6])
                    
                    # Apply multiplier to the specified slice
                    permz[i1:i2+1, j1:j2+1, k1:k2+1] *= mult
                    print(f"Applied MULTZ {mult} to region [{i1+1}:{i2+1}, {j1+1}:{j2+1}, {k1+1}:{k2+1}]")
                except Exception as e:
                    print(f"Warning: Failed to parse MULTZ record: {e}")

        rock = Rock(poro, permx, permy, permz, compressibility)
        rock.sgof = sgof_table
        rock.sw_conn = sw_conn
        rock.krg_max = krg_max
        return rock

    def _parse_fluid(self) -> Fluid:
        """Parses DENSITY, PVTO, and PVDG tables."""
        dens_kw = self.deck['DENSITY']
        oil_density = self._get_val(dens_kw[0][0])
        gas_density = self._get_val(dens_kw[0][2]) if len(dens_kw[0]) > 2 else 0.0533
        
        from ..core.fluid import PVTOTable, PVDGTable, Fluid
        
        pvto_table = None
        if 'PVTO' in self.deck:
            pvto_nodes = []
            for row in self.deck['PVTO']:
                rs = row[0].get_raw(0)
                data = row[1].get_raw_data_list()
                pvto_nodes.append({'rs': rs, 'data': data})
            pvto_table = PVTOTable(pvto_nodes)
            
        pvdg_table = None
        if 'PVDG' in self.deck:
            pvdg_data = self.deck['PVDG'][0][0].get_raw_data_list()
            pvdg_table = PVDGTable(pvdg_data)
            
        # Simplified compressibility logic if needed elsewhere
        compressibility = 3e-6
            
        return Fluid(pvto_table, pvdg_table, oil_density, gas_density, compressibility)

    def _parse_pressure(self, grid, fluid) -> np.ndarray:
        """Parses PRESSURE from SOLUTION section, with EQUIL fallback."""
        dims = grid.dimensions
        if 'PRESSURE' in self.deck:
            p_data = self.deck['PRESSURE'][0][0].get_raw_data_list()
            return np.array(p_data).reshape(dims, order='F')
            
        if 'EQUIL' in self.deck:
            # Item 1: Datum Depth, Item 2: Datum Pressure
            z_datum = self._get_val(self.deck['EQUIL'][0][0])
            p_datum = self._get_val(self.deck['EQUIL'][0][1])
            
            # Use consistent density-gradient physics with simulator.py
            # SPE1 datum is at 8400 ft, P=4800 psi. Saturated with Rs=1.27.
            rs_val = 1.27 # Standard for SPE1
            bo_val = 1.60 # Standard for SPE1
            
            # lb/RB density for saturated oil
            rho_o_surf = fluid.density_oil # 52.8
            rho_g_surf = fluid.density_gas # 0.0702
            
            # Consistent with simulator.py units: 5.61458 ft3/bbl, 1000 scf/mscf
            rho_res = (rho_o_surf * 5.61458 + rs_val * 1000.0 * rho_g_surf) / bo_val
            oil_grad = rho_res / (5.61458 * 144.0)
            
            print(f"Initializing Equilibrium with oil gradient: {oil_grad:.4f} psi/ft (from rho={rho_res:.4f} lb/RB)")
            
            # Calculate pressure for each cell based on its center depth
            z_centers = grid.z_centers
            p_init = p_datum + (z_centers - z_datum) * oil_grad
            return p_init
            
        return np.full(dims, 3000.0)

    def _parse_rs(self, grid) -> np.ndarray:
        """Parses RS from SOLUTION section, with RSVD fallback."""
        dims = grid.dimensions
        if 'RS' in self.deck:
            rs_data = self.deck['RS'][0][0].get_raw_data_list()
            return np.array(rs_data).reshape(dims, order='F')
            
        if 'RSVD' in self.deck:
            # RSVD table: each row [Depth, Rs]
            rsvd_raw = self.deck['RSVD'][0][0].get_raw_data_list()
            # Convert flat list [z1, rs1, z2, rs2, ...] to depth and rs arrays
            z_rsvd = np.array(rsvd_raw[0::2])
            rs_rsvd = np.array(rsvd_raw[1::2])
            
            z_centers = grid.z_centers
            # Interpolate Rs based on cell center depth
            return np.interp(z_centers, z_rsvd, rs_rsvd)
            
        return np.zeros(dims)

    def _parse_wells(self) -> list:
        """Parses WELSPECS, COMPDAT, WCONPROD, and WCONINJE."""
        wells = []
        if 'WELSPECS' not in self.deck:
            return wells
            
        # 1. Basic well specs (I, J and reference depth)
        specs = {}
        for row in self.deck['WELSPECS']:
            name = row[0].get_str(0)
            i = int(self._get_val(row[2])) - 1
            j = int(self._get_val(row[3])) - 1
            # row[4] is Z-datum, ignore for K-index
            specs[name] = [i, j, 0] # Default K=0
            
        # 2. Connection data (COMPDAT) - overwrites K from SPECS
        if 'COMPDAT' in self.deck:
            for row in self.deck['COMPDAT']:
                name = row[0].get_str(0)
                if name in specs:
                    # SPE1: PROD at 10 10 3 3 ... -> i=9, j=9, k_top=2
                    k_top = int(self._get_val(row[3])) - 1
                    specs[name][2] = k_top
                    
                    # Read SKIN if available (Item 9, column 8)
                    try:
                        if len(row) > 8:
                            skin_val = float(self._get_val(row[8]))
                            # Store skin in specs to pass to Well constructor
                            if len(specs[name]) < 4:
                                specs[name].append(skin_val)
                            else:
                                specs[name][3] = skin_val
                    except Exception:
                        pass

        # 3. Production control (WCONPROD)
        # WCONPROD columns: Name, Status, Control, ORAT, WRAT, GRAT, LRAT, RESV, BHP
        if 'WCONPROD' in self.deck:
            for row in self.deck['WCONPROD']:
                name = row[0].get_str(0)
                if name not in specs:
                    continue
                status = row[1].get_str(0)
                if status == 'SHUT':
                    continue

                control = row[2].get_str(0)
                orat = None
                grat = None
                bhp_min = None

                # Always read the ORAT target (col 3) if non-zero
                try:
                    orat_val = self._get_val(row[3])
                    if orat_val > 0:
                        orat = orat_val
                except Exception:
                    pass

                # Read BHP minimum constraint (col 8)
                try:
                    if len(row) > 8:
                        bhp_val = self._get_val(row[8])
                        if bhp_val > 0:
                            bhp_min = bhp_val
                except Exception:
                    pass

                # The `rate` in Well is the negative ORAT (for legacy backward compat)
                rate = -orat if orat else 0.0

                # Read skin if stored in specs
                skin = specs[name][3] if len(specs[name]) > 3 else 0.0
                
                wells.append(Well(name, tuple(specs[name][:3]),
                                  rate=rate, bhp=bhp_min,
                                  orat=orat, grat=grat, skin=skin))

        # WCONINJE: Name, Type, Status, Control, Rate, ResV, BHP
        if 'WCONINJE' in self.deck:
            for row in self.deck['WCONINJE']:
                name = row[0].get_str(0)
                if name not in specs:
                    continue
                status = row[2].get_str(0)
                if status == 'SHUT':
                    continue

                control = row[3].get_str(0)
                gir = None
                rate = 0.0
                bhp_max = None

                # Gas injection rate target (col 4) — MSCF/day
                try:
                    rate_val = self._get_val(row[4])
                    if rate_val > 0:
                        gir = rate_val
                        rate = gir   # positive = injection
                except Exception:
                    pass

                # BHP maximum constraint (col 6)
                try:
                    if len(row) > 6:
                        bhp_val = self._get_val(row[6])
                        if bhp_val > 0:
                            bhp_max = bhp_val
                except Exception:
                    pass

                # Read skin if stored in specs
                skin = specs[name][3] if len(specs[name]) > 3 else 0.0

                wells.append(Well(name, tuple(specs[name][:3]),
                                  rate=rate, bhp=bhp_max, gir=gir, skin=skin))

        return wells

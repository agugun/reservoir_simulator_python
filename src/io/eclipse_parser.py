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
        pressure = self._parse_pressure(grid.dimensions)
        rs = self._parse_rs(grid.dimensions)
        
        # Wells from SCHEDULE section
        wells = self._parse_wells()
        
        return ReservoirModel(grid, rock, fluid, pressure, wells=wells, rs=rs)

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
            tops_data = self.deck['TOPS'][0][0].get_raw_data_list()
            # TOPS is usually only for the top layer (nx*ny) or all cells (nx*ny*nz)
            if len(tops_data) == nx * ny:
                top_depth = np.array(tops_data, dtype=np.float32).reshape((nx, ny, 1), order='F')
            else:
                top_depth = np.array(tops_data, dtype=np.float32).reshape((nx, ny, nz), order='F')
        
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
        if 'SGOF' in self.deck:
            sgof_data = self.deck['SGOF'][0][0].get_raw_data_list()
            sgof_table = {
                'sg': np.array(sgof_data[0::4]),
                'krg': np.array(sgof_data[1::4]),
                'krog': np.array(sgof_data[2::4])
            }
            
        rock = Rock(poro, permx, permy, permz, compressibility)
        rock.sgof = sgof_table
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

    def _parse_pressure(self, dims: tuple) -> np.ndarray:
        """Parses PRESSURE from SOLUTION section, with EQUIL fallback."""
        if 'PRESSURE' in self.deck:
            p_data = self.deck['PRESSURE'][0][0].get_raw_data_list()
            return np.array(p_data).reshape(dims, order='F')
            
        if 'EQUIL' in self.deck:
            # Item 2 is datum pressure
            p_datum = self._get_val(self.deck['EQUIL'][0][1])
            return np.full(dims, p_datum)
            
        return np.full(dims, 3000.0)

    def _parse_rs(self, dims: tuple) -> np.ndarray:
        """Parses RS from SOLUTION section, with RSVD fallback."""
        if 'RS' in self.deck:
            rs_data = self.deck['RS'][0][0].get_raw_data_list()
            return np.array(rs_data).reshape(dims, order='F')
            
        if 'RSVD' in self.deck:
            rs_val = float(self.deck['RSVD'][0][0].get_raw_data_list()[1])
            return np.full(dims, rs_val)
            
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

        # 3. Production control (WCONPROD)
        if 'WCONPROD' in self.deck:
            for row in self.deck['WCONPROD']:
                name = row[0].get_str(0)
                if name in specs:
                    status = row[1].get_str(0)
                    if status == 'SHUT':
                        continue
                    
                    control = row[2].get_str(0)
                    rate = 0.0
                    bhp = None
                    
                    if control == 'ORAT':
                        rate = -self._get_val(row[3])
                    elif control == 'BHP':
                        bhp = self._get_val(row[8])
                    else:
                        rate = -self._get_val(row[3])
                    
                    if bhp is None and len(row) > 8:
                        bhp = self._get_val(row[8])
                        
                    wells.append(Well(name, tuple(specs[name]), rate=rate, bhp=bhp))

        # WCONINJE: Name, Type, Status, Control, Rate, ResV, BHP
        if 'WCONINJE' in self.deck:
            for row in self.deck['WCONINJE']:
                name = row[0].get_str(0)
                if name in specs:
                    status = row[2].get_str(0)
                    if status == 'SHUT':
                        continue
                        
                    control = row[3].get_str(0)
                    rate = 0.0
                    bhp = None
                    
                    if control == 'RATE':
                        rate = self._get_val(row[4]) # Injection is positive
                    elif control == 'BHP':
                        bhp = self._get_val(row[6])
                        
                    if bhp is None and len(row) > 6:
                        bhp = self._get_val(row[6])
                        
                    wells.append(Well(name, tuple(specs[name]), rate=rate, bhp=bhp))
                    
        return wells

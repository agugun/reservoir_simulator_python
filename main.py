import numpy as np
import os
from src.core import Simulator
from src.io import EclipseParser

def run_simulation(data_file: str = None):
    """
    Main entry point for the reservoir simulation.
    Can load from an Eclipse .DATA file or use a hardcoded model.
    """
    print("="*40)
    print("   RESERVOIR SIMULATOR (PYTHON)    ")
    print("="*40)
    
    if data_file and os.path.exists(data_file):
        print(f"Loading model from: {data_file}")
        try:
            parser = EclipseParser(data_file)
            model = parser.build_model()
        except Exception as e:
            print(f"\n[ERROR] Failed to load reservoir model: {e}")
            print("\nPlease ensure the focused file in VS Code is a valid .DATA deck.")
            return
    else:
        print("Using default synthetic model...")
        from src.core import Grid, Rock, Fluid, ReservoirModel, Well
        # Fallback to the original hardcoded model if no file provided
        grid = Grid(nx=5, ny=5, nz=3, dx=100.0, dy=100.0, dz=20.0)
        rock = Rock.homogeneous(grid.dimensions, poro=0.25, perm_mD=50.0)
        fluid = Fluid(viscosity=1.0, density=62.4, compressibility=1e-5, fvf=1.0)
        p_init = np.full(grid.dimensions, 4000.0)
        wells = [Well("PROD-1", (2, 2, 0), rate=-500.0)]
        model = ReservoirModel(grid, rock, fluid, p_init, wells=wells)

    print(f"Grid setup complete. Total cells: {model.grid.nx * model.grid.ny * model.grid.nz}")
    print(f"Wells defined: {[w.name for w in model.wells]}")
    
    # Initialize Simulator
    print("Initializing Simulator and Transmissibility...")
    
    # 3.5 Hydrostatic Initialization (OPM Parity)
    # Datum: 8400 ft, 4800 psia. Oil density: 53.66 lb/ft3 -> 0.3726 psi/ft
    # TOPS[i,j,k] is the top of the cell. Center is TOPS + DZ/2.
    z_centers = model.grid.top_depth + model.grid.dz/2.0
    model.pressure = 4800.0 + (z_centers - 8400.0) * (53.66 / 144.0)
    
    sim = Simulator(model)
    
    # Setup Exporter
    writer = None
    if data_file:
        base_name = os.path.splitext(data_file)[0]
        from src.io import EclipseWriter
        writer = EclipseWriter(model, base_name)
        writer.write_egrid()
        writer.write_init() # Export grid properties
        writer.write_summary_spec() # Create the .SMSPEC file
        # Initial step (Time 0)
        writer.write_restart(0.0, model.pressure, 0, model.swat, model.sgas, dt=0.0)
    
    # Simulation Parameters
    well_totals = {w.name: {'oil': 0.0, 'gas': 0.0} for w in model.wells}
    
    print(f"\nStarting Final Parity Simulation (Adaptive 120 target steps):")
    
    all_summary_results = []
    
    target_times = [1.0, 4.0, 11.60956, 31.0] + [31.0 + i*30.4375 for i in range(1, 120)]
    report_step = 1
    
    total_time = 0.0
    dt_current = 2.0 # Initial small step
    
    for target in target_times:
        step_dt_accum = 0.0
        while total_time < target - 1e-4:
            dt_step = min(dt_current, target - total_time)
            try:
                # FIM Step Native Matrix Solve
                model.pressure = sim.step_fim(dt_step)
                total_time += dt_step
                step_dt_accum += dt_step
                
                # Successful convergence: attempt to magnify stepsize smoothly
                dt_current = min(dt_current * 1.5, 30.4375)
                
            except RuntimeError as e:
                # Newton diverged! Chop the time step
                print(f"[{total_time:.2f} days] {e} - Chopping dt from {dt_step:.4f} to {dt_step*0.25:.4f}")
                dt_current *= 0.25
                if dt_current < 1e-5:
                    raise BaseException("Timestep collapsed due to severe physical non-linearities.")
                continue # Retry step
                
        # Target reached precisely. Extrapolate outputs and write.
        avg_p = np.mean(model.pressure)
        
        # Collect data Snapshot
        well_data = {}
        field_rates = {'oil': 0.0, 'gas': 0.0}
        
        for well in model.wells:
            wi = sim.calculate_well_index(well)
            w_idx = well.location
            p_cell = model.pressure[w_idx]
            rs_cell = model.rs[w_idx]
            
            bo, visc = model.fluid.get_oil_props(p_cell, rs_cell)
            mob = 1.0 / (visc * bo if visc * bo > 1e-6 else 1e-6)
            
            q_pot = wi * mob * (p_cell - well.bhp) if well.bhp is not None else -well.rate
            is_prod = "PROD" in well.name.upper()
            
            q_well = max(0, q_pot) if is_prod else min(0, q_pot)
            
            well_totals[well.name]['oil'] += max(0, q_well) * step_dt_accum if is_prod else 0.0
            well_totals[well.name]['gas'] += max(0, q_well) * step_dt_accum if not is_prod else 0.0
            
            well_data[well.name] = {
                'bhp': well.bhp if well.bhp is not None else p_cell,
                'oil': max(0, q_well) if is_prod else 0.0,
                'gas': max(0, q_well) if not is_prod else 0.0,
                'oil_total': well_totals[well.name]['oil'],
                'gas_total': well_totals[well.name]['gas']
            }
            if is_prod: field_rates['oil'] += abs(q_well)
            else: field_rates['gas'] += abs(q_well)
            
        writer.write_restart(total_time, model.pressure, report_step, model.swat, model.sgas, dt=step_dt_accum)
        vals = writer.write_summary_data(total_time, model.pressure, model.swat, model.sgas, field_rates, well_data, report_step, write_seqhdr=(report_step==1))
        all_summary_results.append(vals)
        
        if report_step == 4 or report_step % 20 == 0:
            print(f"--- Step {report_step} (Month {report_step if report_step < 4 else report_step - 3}) ---")
            print(f"Time: {total_time:.1f} days | Avg Pressure: {avg_p:.2f} psi | dt: {dt_current:.2f}")
            
        report_step += 1

    # Final Summary File (.ESMRY)
    writer.write_esmry(all_summary_results)
    
    print("\nSimulation Run Complete.")
    print("="*40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Reservoir Simulation")
    parser.add_argument(
        "--input-file", "-i", 
        type=str, 
        help="Path to the Eclipse .DATA file"
    )
    
    args = parser.parse_args()
    
    # Priority: 1. Command-line argument, 2. Default sample file
    data_file = args.input_file
    if not data_file:
        sample_path = "data/sample/sample_model.DATA"
        if os.path.exists(sample_path):
            data_file = sample_path
            
    run_simulation(data_file)

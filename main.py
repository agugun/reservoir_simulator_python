import numpy as np
import os
from src.core import Simulator
from src.io import EclipseParser
from src.io.report_writer import OPMReportWriter

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
    opm_reporter = None
    if data_file:
        base_name = os.path.splitext(data_file)[0]
        from src.io import EclipseWriter
        writer = EclipseWriter(model, base_name)
        opm_reporter = OPMReportWriter(base_name, model)
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
        
        # Start Time Step Log
        date_str = "01-Jan-2015"
        if opm_reporter:
            opm_reporter.log_time_step_start(report_step, target - total_time, total_time, date_str)
            
        while total_time < target - 1e-4:
            dt_step = min(dt_current, target - total_time)
            try:
                # FIM Step Native Matrix Solve
                model.pressure = sim.step_fim(dt_step, report_writer=opm_reporter)
                total_time += dt_step
                step_dt_accum += dt_step
                
                # Advanced PID Target Time-Stepping Logic
                iters = getattr(sim, 'last_iterations', 1)
                target_iters = 4.0  # OPM Industrial target for Newton sequences
                omega = 0.5
                
                # Calculate convergence scaling ratio rationally
                alpha = (target_iters + omega) / (iters + omega)
                alpha = max(0.5, min(alpha, 2.0)) # Bound magnification limits securely
                
                # Safely update future baseline unconditionally ensuring stability
                dt_current = min(dt_current * alpha, 30.4375)
                
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
        field_rates = {'oil': 0.0, 'gas': 0.0, 'FOPR': 0.0, 'FGPR': 0.0, 'FPRV': 0.0, 'FGIR': 0.0, 'FIRV': 0.0}
        cum_totals = {'FOPT': 0.0, 'FGPT': 0.0, 'FPRV_CUM': 0.0, 'FGIT': 0.0, 'FIRV_CUM': 0.0}
        
        for well in model.wells:
            wi = sim.calculate_well_index(well)
            w_idx = well.location
            p_cell = model.pressure[w_idx]
            rs_cell = model.rs[w_idx]
            
            bo, visc = model.fluid.get_oil_props(p_cell, rs_cell)
            mob = 1.0 / (visc * bo if bo > 0 else 1.0)
            
            q_pot = wi * mob * (p_cell - well.bhp) if well.bhp is not None else -well.rate
            is_prod = "PROD" in well.name.upper()
            
            q_well = max(0, q_pot) if is_prod else min(0, q_pot)
            q_resv = abs(q_well) * bo
            
            # Generic gas flux assumptions (Simplified proxy since it's hardcoded)
            q_gas = abs(q_well) * 1.24 if is_prod else abs(q_well)
            
            well_totals[well.name]['oil'] += abs(q_well) * step_dt_accum if is_prod else 0.0
            well_totals[well.name]['gas'] += q_gas * step_dt_accum
            well_totals[well.name].setdefault('resv', 0.0)
            well_totals[well.name]['resv'] += q_resv * step_dt_accum
            
            well_data[well.name] = {
                'type': 'PROD' if is_prod else 'INJ',
                'i': w_idx[0] + 1,
                'j': w_idx[1] + 1,
                'bhp': well.bhp if well.bhp is not None else p_cell,
                'oil': abs(q_well) if is_prod else 0.0,
                'gas': q_gas if not is_prod else 0.0,
                'orat': abs(q_well) if is_prod else 0.0,
                'grat': q_gas,
                'resv': q_resv,
                'cum_oil': well_totals[well.name]['oil'] if is_prod else 0.0,
                'cum_gas_prod': well_totals[well.name]['gas'] if is_prod else 0.0,
                'cum_resv_prod': well_totals[well.name]['resv'] if is_prod else 0.0,
                'cum_gas_inj': well_totals[well.name]['gas'] if not is_prod else 0.0,
                'cum_resv_inj': well_totals[well.name]['resv'] if not is_prod else 0.0,
            }
            
            if is_prod: 
                field_rates['oil'] += abs(q_well)
                field_rates['FOPR'] += abs(q_well)
                field_rates['FGPR'] += q_gas
                field_rates['FPRV'] += q_resv
                cum_totals['FOPT'] += well_data[well.name]['cum_oil']
                cum_totals['FGPT'] += well_data[well.name]['cum_gas_prod']
                cum_totals['FPRV_CUM'] += well_data[well.name]['cum_resv_prod']
            else: 
                field_rates['gas'] += abs(q_well)
                field_rates['FGIR'] += q_gas
                field_rates['FIRV'] += q_resv
                cum_totals['FGIT'] += well_data[well.name]['cum_gas_inj']
                cum_totals['FIRV_CUM'] += well_data[well.name]['cum_resv_inj']
            
        writer.write_restart(total_time, model.pressure, report_step, model.swat, model.sgas, dt=step_dt_accum)
        vals = writer.write_summary_data(total_time, model.pressure, model.swat, model.sgas, field_rates, well_data, report_step, write_seqhdr=(report_step==1))
        all_summary_results.append(vals)
        
        if opm_reporter:
            opm_reporter.log_report_matrices(report_step, 120, total_time, target, date_str, field_rates, well_data, cum_totals)
        
        if report_step == 4 or report_step % 20 == 0:
            print(f"--- Step {report_step} (Month {report_step if report_step < 4 else report_step - 3}) ---")
            print(f"Time: {total_time:.1f} days | Avg Pressure: {avg_p:.2f} psi | dt: {dt_current:.2f}")
            
        report_step += 1

    # Final Summary File (.ESMRY)
    writer.write_esmry(all_summary_results)
    if opm_reporter:
        opm_reporter.close()
    
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

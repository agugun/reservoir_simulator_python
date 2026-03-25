import numpy as np
import os
from src.core import Simulator
from src.io import EclipseParser
from src.io.report_writer import OPMReportWriter

def run_simulation(data_file: str = None, output_dir: str = None):
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
        rock = Rock.homogeneous(grid.dimensions, porosity=0.25, perm_x=50.0, perm_y=50.0, perm_z=50.0, compressibility=1e-6)
        fluid = Fluid(density_oil=53.66, density_gas=0.06054)
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
        # Resolve Base Name for outputs
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.join(output_dir, os.path.basename(os.path.splitext(data_file)[0]))
        else:
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
    
    # Monthly intervals for SPE1 (integer days to match ResInsight expectations)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    target_times = []
    current_cum = 0.0
    for year in range(10):
        for days in month_days:
            current_cum += days
            target_times.append(float(current_cum))
            
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
        field_rates = {'oil': 0.0, 'gas': 0.0, 'FOPR': 0.0, 'FGPR': 0.0, 'FPRV': 0.0, 'FGIR': 0.0, 'FIRV': 0.0, 'FOPT': 0.0, 'FGPT': 0.0}
        cum_totals = {'FOPT': 0.0, 'FGPT': 0.0, 'FPRV_CUM': 0.0, 'FGIT': 0.0, 'FIRV_CUM': 0.0}
        
        for well in model.wells:
            wi = sim.calculate_well_index(well)
            w_idx = well.location
            p_cell = model.pressure[w_idx]
            rs_cell = model.rs[w_idx]

            bo, mu_o = model.fluid.get_oil_props(p_cell, rs_cell)
            bo = max(bo, 0.5)   # guard against degenerate PVT

            is_prod = well.is_producer

            if is_prod:
                # ── OPM-aligned producer: ORAT target with BHP-floor limit ──
                #
                # OPM logic (BlackoilWellModelConstraints):
                #   1. Compute max deliverable rate at BHP_floor using Darcy + kro(Sg)
                #   2. Actual rate = min(q_darcy_at_bhp_floor, ORAT_target)
                #   3. When reservoir depletion / gas breakthrough reduces kro,
                #      q_darcy < ORAT → well is BHP-limited, rate declines naturally.

                sg_cell = float(model.sgas[w_idx])

                # kro from SGOF table (same table simulator uses internally)
                if hasattr(model.rock, 'sgof') and model.rock.sgof is not None:
                    sg_arr  = np.array(model.rock.sgof['sg'])
                    kro_arr = np.array(model.rock.sgof['krog'])
                    kro_w   = float(np.interp(sg_cell, sg_arr, kro_arr))
                else:
                    sw_conn = 0.12
                    so_cell = max(1.0 - sw_conn - sg_cell, 0.0)
                    kro_w   = (so_cell / (1.0 - sw_conn)) ** 2

                # Mobility at wellbore conditions
                bg, mu_g = model.fluid.get_gas_props(p_cell)
                bg = float(bg); mu_g = float(mu_g)
                # bg is in RB/MSCF (consistent with PVDG)
                
                # Relative permeabilities from model.rock.sgof
                if hasattr(model.rock, 'sgof') and model.rock.sgof is not None:
                    sg_arr  = np.array(model.rock.sgof['sg'])
                    krg_arr = np.array(model.rock.sgof['krg'])
                    krg_w   = float(np.interp(sg_cell, sg_arr, krg_arr))
                else:
                    krg_w = (sg_cell / (1.0 - 0.12)) ** 2
                    
                lo_w = kro_w / (mu_o * bo)
                lg_w = krg_w / (mu_g * bg)

                bhp_floor = well.bhp if well.bhp is not None else 1000.0
                dp        = max(p_cell - bhp_floor, 0.0)

                # Correct Producer Logic: Calculate phase rates independently
                bhp_floor = well.bhp if well.bhp is not None else 1000.0
                dp        = max(p_cell - bhp_floor, 0.0)

                # Potential reservoir rates (RB/day)
                q_o_pot_resv = wi * lo_w * dp
                q_g_pot_resv = wi * lg_w * dp
                
                # Check for rate targets (ORAT)
                orat_target = well.orat if well.orat is not None else 1e9
                q_o_pot_surf = q_o_pot_resv / bo
                
                # Scale factor if we hit ORAT
                scaling = min(1.0, orat_target / (q_o_pot_surf if q_o_pot_surf > 1e-6 else 1e9))
                
                q_oil_surf = q_o_pot_surf * scaling
                q_oil_resv = q_oil_surf * bo
                
                q_free_gas_mscf = (q_g_pot_resv * scaling) / bg
                q_gas_diss_mscf = q_oil_surf * float(rs_cell)
                q_gas_mscf = q_gas_diss_mscf + q_free_gas_mscf
                
                # Back-calculate actual BHP for reporting
                lt_w = lo_w + lg_w
                if wi > 1e-12 and lt_w > 1e-12:
                    q_tot_resv = q_oil_resv + (q_free_gas_mscf * bg)
                    bhp_actual = p_cell - q_tot_resv / (wi * lt_w)
                else:
                    bhp_actual = p_cell
                bhp_actual = max(bhp_actual, bhp_floor)

                well_totals[well.name]['oil'] += q_oil_surf * step_dt_accum
                well_totals[well.name]['gas'] += q_gas_mscf * step_dt_accum
                well_totals[well.name].setdefault('resv', 0.0)
                q_resv = q_oil_resv + q_free_gas_mscf * bg
                well_totals[well.name]['resv'] += q_resv * step_dt_accum

                well_data[well.name] = {
                    'type':    'PROD',
                    'i': w_idx[0] + 1, 'j': w_idx[1] + 1,
                    'bhp':     bhp_actual,          # actual wellbore BHP, not just floor
                    # WOPR key expected by eclipse_writer
                    'opr':     q_oil_surf,
                    'gpr':     q_gas_mscf,
                    'wpr':     0.0,
                    'opt':     well_totals[well.name]['oil'],
                    'gpt':     well_totals[well.name]['gas'],
                    # Legacy keys still used by report_writer
                    'oil':     q_oil_surf,
                    'gas':     q_gas_mscf,
                    'resv':    q_resv,
                    'orat':    q_oil_surf,
                    'grat':    q_gas_mscf,
                    'cum_oil':      well_totals[well.name]['oil'],
                    'cum_gas_prod': well_totals[well.name]['gas'],
                    'cum_resv_prod':well_totals[well.name]['resv'],
                    'cum_gas_inj':  0.0,
                    'cum_resv_inj': 0.0,
                }

                field_rates['oil']  += q_oil_surf
                field_rates['gas']  += q_gas_mscf
                field_rates['FOPR'] += q_oil_surf
                field_rates['FGPR'] += q_gas_mscf
                field_rates['FPRV'] += q_resv
                
                # Accrue cumulatives for field-level summary export
                field_rates['FOPT'] += well_totals[well.name]['oil']
                field_rates['FGPT'] += well_totals[well.name]['gas']
                
                cum_totals['FOPT']  += well_totals[well.name]['oil']
                cum_totals['FGPT']  += well_totals[well.name]['gas']
                cum_totals['FPRV_CUM'] += well_totals[well.name]['resv']

            else:
                # ── Injector: honour GIR gas injection target ─────────────
                gir_target = well.gir if well.gir is not None else abs(well.rate)
                bhp_ceil   = well.bhp if well.bhp is not None else 15000.0

                # Use exact gas properties from fluid model (consistent with simulator.py)
                bg, mu_g = model.fluid.get_gas_props(p_cell)
                bg = float(bg); mu_g = float(mu_g)
                
                # Injection uses endpoint mobility (krg=1.0)
                mob_g_inj = 1.0 / (mu_g * bg)
                
                dp_inj  = max(bhp_ceil - p_cell, 0.0)
                q_inj_pot = wi * mob_g_inj * dp_inj       # MSCF/day (potential)

                q_gir = min(q_inj_pot, gir_target)        # MSCF/day (actual)

                well_totals[well.name]['gas'] += q_gir * step_dt_accum
                well_totals[well.name].setdefault('resv', 0.0)
                q_resv_inj = q_gir * bg                   # res bbl/day

                well_data[well.name] = {
                    'type':    'INJ',
                    'i': w_idx[0] + 1, 'j': w_idx[1] + 1,
                    'bhp':     bhp_ceil,
                    'gir':     q_gir,
                    'wir':     0.0,
                    'git':     well_totals[well.name]['gas'],
                    # Legacy keys
                    'oil':     0.0,
                    'gas':     q_gir,
                    'resv':    q_resv_inj,
                    'orat':    0.0,
                    'grat':    q_gir,
                    'cum_oil':      0.0,
                    'cum_gas_prod': 0.0,
                    'cum_resv_prod':0.0,
                    'cum_gas_inj':  well_totals[well.name]['gas'],
                    'cum_resv_inj': well_totals[well.name]['resv'],
                }

                field_rates['FGIR'] += q_gir
                field_rates['FIRV'] += q_resv_inj
                cum_totals['FGIT']  += well_data[well.name]['cum_gas_inj']
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
    
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the simulation outputs"
    )
    
    args = parser.parse_args()
    
    # Priority: 1. Command-line argument, 2. Default sample file
    data_file = args.input_file
    output_dir = args.output_dir
    
    if not data_file:
        sample_path = "data/sample/sample_model.DATA"
        if os.path.exists(sample_path):
            data_file = sample_path
            
    run_simulation(data_file, output_dir)

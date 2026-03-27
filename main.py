import numpy as np
import os
from src.core import Simulator, Grid, Rock, Fluid, ReservoirModel, Well
from src.io import EclipseParser
from src.io.report_writer import OPMReportWriter
import sys
import jax

def calculate_snapshot(model, sim, well_totals, step_dt_accum):
    well_data = {}
    field_rates = {'oil': 0.0, 'gas': 0.0, 'FOPR': 0.0, 'FGPR': 0.0, 'FPRV': 0.0, 'FGIR': 0.0, 'FIRV': 0.0, 'FOPT': 0.0, 'FGPT': 0.0}
    cum_totals = {'FOPT': 0.0, 'FGPT': 0.0, 'FPRV_CUM': 0.0, 'FGIT': 0.0, 'FIRV_CUM': 0.0}
    
    for well in model.wells:
        wi = sim.calculate_well_index(well)
        w_idx = well.location
        p_cell = model.pressure[w_idx]
        rs_cell = model.rs[w_idx]

        bo, mu_o = model.fluid.get_oil_props(p_cell, rs_cell)
        bo = max(bo, 0.5)

        is_prod = well.is_producer

        if is_prod:
            sg_cell = float(model.sgas[w_idx])
            if hasattr(model.rock, 'sgof') and model.rock.sgof is not None:
                sg_arr = np.array(model.rock.sgof['sg'])
                kro_arr = np.array(model.rock.sgof['krog'])
                kro_w = float(np.interp(sg_cell, sg_arr, kro_arr))
            else:
                sw_conn = 0.12
                so_cell = max(1.0 - sw_conn - sg_cell, 0.0)
                kro_w = (so_cell / (1.0 - sw_conn)) ** 2

            bg, mu_g = model.fluid.get_gas_props(p_cell)
            bg = float(bg); mu_g = float(mu_g)
            
            if hasattr(model.rock, 'sgof') and model.rock.sgof is not None:
                sg_arr = np.array(model.rock.sgof['sg'])
                krg_arr = np.array(model.rock.sgof['krg'])
                krg_w = float(np.interp(sg_cell, sg_arr, krg_arr))
            else:
                krg_w = (sg_cell / (1.0 - 0.12)) ** 2
                
            lo_w = kro_w / (mu_o * bo)
            lg_w = krg_w / (mu_g * bg)

            bhp_floor = well.bhp if well.bhp is not None else 1000.0
            dp = max(p_cell - bhp_floor, 0.0)

            q_o_pot_stb = wi * lo_w * dp
            q_g_pot_mscf = wi * lg_w * dp
            orat_target = well.orat if well.orat is not None else 1e9
            scaling = min(1.0, orat_target / (q_o_pot_stb if q_o_pot_stb > 1e-6 else 1e9))
            
            q_oil_surf = q_o_pot_stb * scaling
            q_oil_resv = q_oil_surf * bo
            q_gas_mscf = q_g_pot_mscf * scaling + q_oil_surf * float(rs_cell)
            
            if wi > 1e-12 and (lo_w + lg_w) > 1e-12:
                q_res = (q_o_pot_stb * bo + q_g_pot_mscf * bg) * scaling
                bhp_actual = p_cell - q_res / (wi * (kro_w / mu_o + krg_w / mu_g))
            else:
                bhp_actual = p_cell
            bhp_actual = max(bhp_actual, bhp_floor)

            # Trapezoidal Integration Logic
            if step_dt_accum > 0:
                q_o_avg = 0.5 * (well_totals[well.name].get('last_q_oil', q_oil_surf) + q_oil_surf)
                q_g_avg = 0.5 * (well_totals[well.name].get('last_q_gas', q_gas_mscf) + q_gas_mscf)
                well_totals[well.name]['oil'] += q_o_avg * step_dt_accum
                well_totals[well.name]['gas'] += q_g_avg * step_dt_accum
            
            well_totals[well.name]['last_q_oil'] = q_oil_surf
            well_totals[well.name]['last_q_gas'] = q_gas_mscf
            
            well_totals[well.name].setdefault('resv', 0.0)
            q_free_gas_mscf = q_gas_mscf - q_oil_surf * float(rs_cell)
            q_resv = q_oil_resv + q_free_gas_mscf * bg
            
            if step_dt_accum > 0:
                q_r_avg = 0.5 * (well_totals[well.name].get('last_q_resv', q_resv) + q_resv)
                well_totals[well.name]['resv'] += q_r_avg * step_dt_accum
            well_totals[well.name]['last_q_resv'] = q_resv

            well_data[well.name] = {
                'type': 'PROD', 'i': w_idx[0] + 1, 'j': w_idx[1] + 1, 'bhp': bhp_actual,
                'opr': q_oil_surf, 'gpr': q_gas_mscf, 'wpr': 0.0,
                'opt': well_totals[well.name]['oil'], 'gpt': well_totals[well.name]['gas'],
                'oil': q_oil_surf, 'gas': q_gas_mscf, 'resv': q_resv,
                'orat': q_oil_surf, 'grat': q_gas_mscf,
                'cum_oil': well_totals[well.name]['oil'], 'cum_gas_prod': well_totals[well.name]['gas'],
                'cum_resv_prod': well_totals[well.name]['resv'], 'cum_gas_inj': 0.0, 'cum_resv_inj': 0.0,
            }
            field_rates['oil'] += q_oil_surf
            field_rates['gas'] += q_gas_mscf
            field_rates['FOPR'] += q_oil_surf
            field_rates['FGPR'] += q_gas_mscf
            field_rates['FPRV'] += q_resv
            field_rates['FOPT'] += well_totals[well.name]['oil']
            field_rates['FGPT'] += well_totals[well.name]['gas']
            cum_totals['FOPT'] += well_totals[well.name]['oil']
            cum_totals['FGPT'] += well_totals[well.name]['gas']
            cum_totals['FPRV_CUM'] += well_totals[well.name]['resv']

        else:
            gir_target = well.gir if well.gir is not None else abs(well.rate)
            bhp_ceil = well.bhp if well.bhp is not None else 15000.0
            bg, mu_g = model.fluid.get_gas_props(p_cell)
            bg = float(bg); mu_g = float(mu_g)
            mob_g_inj = 1.0 / (mu_g * bg)
            dp_inj = max(bhp_ceil - p_cell, 0.0)
            
            # Injector logic refinement
            q_gir = min(wi * mob_g_inj * dp_inj, gir_target)

            if step_dt_accum > 0:
                q_g_avg = 0.5 * (well_totals[well.name].get('last_q_gas', q_gir) + q_gir)
                well_totals[well.name]['gas'] += q_g_avg * step_dt_accum
            well_totals[well.name]['last_q_gas'] = q_gir
            
            well_totals[well.name].setdefault('resv', 0.0)
            q_resv_inj = q_gir * bg
            if step_dt_accum > 0:
                q_r_avg = 0.5 * (well_totals[well.name].get('last_q_resv', q_resv_inj) + q_resv_inj)
                well_totals[well.name]['resv'] += q_r_avg * step_dt_accum
            well_totals[well.name]['last_q_resv'] = q_resv_inj

            well_data[well.name] = {
                'type': 'INJ', 'i': w_idx[0] + 1, 'j': w_idx[1] + 1, 'bhp': bhp_ceil,
                'gir': q_gir, 'wir': 0.0, 'git': well_totals[well.name]['gas'],
                'oil': 0.0, 'gas': q_gir, 'resv': q_resv_inj, 'orat': 0.0, 'grat': q_gir,
                'cum_oil': 0.0, 'cum_gas_prod': 0.0, 'cum_resv_prod': 0.0,
                'cum_gas_inj': well_totals[well.name]['gas'], 'cum_resv_inj': well_totals[well.name]['resv'],
            }
            field_rates['FGIR'] += q_gir
            field_rates['FIRV'] += q_resv_inj
            cum_totals['FGIT'] += well_data[well.name]['cum_gas_inj']
            cum_totals['FIRV_CUM'] += well_data[well.name]['cum_resv_inj']

    return well_data, field_rates, cum_totals


def run_simulation(data_file: str = None, output_dir: str = None, refine: bool = False, compare: bool = False, ref_dir: str = None, num_steps: int = 120):
    """
    Main entry point for the reservoir simulation.
    Can load from an Eclipse .DATA file or use a hardcoded model.
    """
    print("="*40)
    print("   RESERVOIR SIMULATOR (PYTHON)    ")
    print("="*40)
    
    if not data_file or not os.path.exists(data_file):
        print(f"\n[ERROR] No valid Eclipse .DATA file provided or file not found: '{data_file}'")
        print("Usage: python3 main.py --input-file path/to/your/deck.DATA")
        return

    print(f"Loading model from: {data_file}")
    try:
        parser = EclipseParser(data_file)
        model = parser.build_model()
    except Exception as e:
        print(f"\n[ERROR] Failed to load reservoir model: {e}")
        print("\nPlease ensure the provided file is a valid .DATA deck.")
        return

    print(f"Grid setup complete. Total cells: {model.grid.nx * model.grid.ny * model.grid.nz}")
    print(f"Wells defined: {[w.name for w in model.wells]}")
    
    # 3. Grid Management
    nx, ny, nz = model.grid.nx, model.grid.ny, model.grid.nz
    
    if refine:
        print(f"Applying Strategic Grid Refinement for {nx}x{ny}x{nz} grid...")
        
        def get_refined_spacing(n, total_length, ratio=1.5):
            """Creates a symmetric parabolic distribution focusing resolution in the center."""
            if n < 3: return np.full(n, total_length / n)
            center = (n - 1) / 2.0
            i = np.arange(n)
            weights = 1.0 + ratio * ((i - center) / center)**2
            return weights * (total_length / np.sum(weights))

        # Extract total lengths from original grid
        total_x = np.sum(model.grid.dx[:, 0, 0])
        total_y = np.sum(model.grid.dy[0, :, 0])
        total_z = np.sum(model.grid.dz[0, 0, :])
        
        dx_refined = get_refined_spacing(nx, total_x)
        dy_refined = get_refined_spacing(ny, total_y)
        dz_refined = model.grid.dz[0, 0, :] 
        
        dx_3d = np.zeros((nx, ny, nz))
        dy_3d = np.zeros((nx, ny, nz))
        dz_3d = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    dx_3d[i,j,k] = dx_refined[i]
                    dy_3d[i,j,k] = dy_refined[j]
                    dz_3d[i,j,k] = dz_refined[k]

        model.grid = Grid(nx, ny, nz, dx_3d, dy_3d, dz_3d, actnum=model.grid.actnum, top_depth=model.grid.top_depth)
    else:
        print(f"Using deck grid directly ({nx}x{ny}x{nz}).")
    
    # Initialize Simulator
    print("\nWELL CONFIGURATION CHECK:")
    for well in model.wells:
        print(f"Well: {well.name} | Type: {'PROD' if well.is_producer else 'INJ'} | Location: {well.location}")
    
    print("Initializing Simulator and Transmissibility...")
    
    # 3.5 Simulator Initialization
    # model.pressure and model.rs are now automatically handled by EclipseParser
    # via EQUIL/RSVD or PRESSURE/RS keywords.
    
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
        
        # Simulation Parameters with rate history for Trapezoidal integration
        well_totals = {w.name: {
            'oil': 0.0, 'gas': 0.0, 'resv': 0.0,
            'last_q_oil': 0.0, 'last_q_gas': 0.0, 'last_q_resv': 0.0
        } for w in model.wells}

        # Write initial summary record (T=0) for proper interpolation
        # This call populates well_totals[...]['last_q_...'] for subsequent steps
        init_well_data, init_field_rates, _ = calculate_snapshot(model, sim, well_totals, 0.0)
        writer.write_summary_data(0.0, model.pressure, model.swat, model.sgas, init_field_rates, init_well_data, 0, write_seqhdr=True)
    
    print(f"\nStarting Final Parity Simulation (Adaptive {num_steps} target steps):")
    
    all_summary_results = []
    
    # Target times with Refined Start-up (Warm-up steps matching OPM behavior)
    warm_up_times = [1.0, 4.0, 11.0, 20.0]
    target_times = [t for t in warm_up_times]

    # Monthly intervals for SPE1 (integer days to match ResInsight expectations)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Calculate steps per month based on num_steps
    # Total nominal steps is 120 (10 years * 12 months)
    steps_per_month = max(1, num_steps // 120)
    
    current_cum = 0.0
    for year in range(10):
        for days in month_days:
            # Sub-divide each month into steps_per_month segments
            for k in range(1, steps_per_month + 1):
                sub_step_time = current_cum + (days * k / steps_per_month)
                # Only add if we haven't already covered this time in warm-up
                if sub_step_time > target_times[-1]:
                    target_times.append(float(sub_step_time))
            current_cum += days
            
    report_step = 1

    
    total_time = 0.0
    dt_current = 2.0 # Initial small step
    step_idx = 0
    
    for target in target_times:
        step_dt_accum = 0.0
        
        # Start Time Step Log
        # Report Date from deck
        date_str = model.start_date 
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
        
        # Collect data Snapshot using helper function
        well_data, field_rates, cum_totals = calculate_snapshot(model, sim, well_totals, step_dt_accum)

        writer.write_restart(total_time, model.pressure, report_step, model.swat, model.sgas, dt=step_dt_accum)
        vals = writer.write_summary_data(total_time, model.pressure, model.swat, model.sgas, field_rates, well_data, report_step, write_seqhdr=(report_step==1))
        all_summary_results.append(vals)
        
        if opm_reporter:
            opm_reporter.log_report_matrices(report_step, len(target_times), total_time, target, date_str, field_rates, well_data, cum_totals)
        
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
    
    # Automated Comparison Report
    if compare:
        import subprocess
        print("\nGenerating Comparison Report...")
        py_dir = output_dir if output_dir else "output"
        opm_dir = ref_dir if ref_dir else "comparison/opm_run/"
        fig_dir = os.path.join("comparison", "figures_auto")
        
        cmd = [
            sys.executable, "tools/compare_results.py",
            "--opm-dir", opm_dir,
            "--py-dir", py_dir,
            "--output-dir", fig_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=False, check=True)
            print(f"\n[SUCCESS] Comparison report generated in: {fig_dir}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Comparison tool failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Reservoir Simulation")
    parser.add_argument(
        "--input-file", "-i", 
        type=str, 
        help="Path to the Eclipse .DATA file"
    )
    
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario name (e.g., spe1) to automatically resolve data, output, and ref paths"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "gpu"],
        default="cpu",
        help="JAX execution device (cpu, cuda, or gpu)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the simulation outputs"
    )
    
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Apply strategic grid refinement to focus resolution in the center"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Execute tools/compare_results.py after simulation"
    )
    
    parser.add_argument(
        "--ref-dir",
        default="tests/ref/",
        help="Default reference OPM directory for comparison"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Target number of simulation steps (default 120 for 10 years monthly)"
    )
    
    args = parser.parse_args()
    
    # Configure JAX Device
    if args.device:
        device_name = "gpu" if args.device == "cuda" else args.device
        jax.config.update("jax_platform_name", device_name)
        print(f"JAX configured for hardware acceleration on: {device_name.upper()}")
    
    # Resolve Scenario Paths
    data_file = args.input_file
    output_dir = args.output_dir
    ref_dir = args.ref_dir
    
    if args.scenario:
        scenario = args.scenario.lower()
        print(f"Scenario-Based Execution Mode: '{scenario}'")
        
        # 1. Resolve Data File (Strictly from tests/run/)
        data_dir = f"tests/run/{scenario}/"
        found_data = False
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if scenario in f.lower() and f.endswith(".DATA"):
                    data_file = os.path.join(data_dir, f)
                    found_data = True
                    break
        
        if not found_data:
            print(f"  ⚠  Could not find matching .DATA file for scenario '{scenario}' in {data_dir}")
            
        # 2. Resolve Paths
        output_dir = f"tests/run/{scenario}/"
        ref_dir = f"tests/ref/{scenario}/"
        print(f"  → Loading model: {data_file}")
        print(f"  → Output: {output_dir}")
        print(f"  → Reference: {ref_dir}")
        
    # Validation
    if not data_file:
        raise FileNotFoundError("Simulation Error: No .DATA model provided or scenario resolution failed.")
            
    run_simulation(data_file, output_dir, refine=args.refine, compare=args.compare, ref_dir=ref_dir, num_steps=args.steps)


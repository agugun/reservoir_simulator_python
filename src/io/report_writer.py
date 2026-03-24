import os
import datetime
import numpy as np

class OPMReportWriter:
    def __init__(self, file_prefix, model):
        self.prt_path = f"{file_prefix}.PRT"
        self.dbg_path = f"{file_prefix}.DBG"
        self.model = model
        self.start_time = datetime.datetime.now()
        
        self.prt_file = open(self.prt_path, "w")
        self.dbg_file = open(self.dbg_path, "w")
        
        self._write_header()

    def _write_header(self):
        # Fake 150-line header matching the general structure of OPM PRT
        header = f"""

 ########  #          ######   #           #
 #         #         #      #   #         # 
 #####     #         #      #    #   #   #  
 #         #         #      #     # # # #   
 #         #######    ######       #   #    

Flow is a simulator for fully implicit three-phase black-oil flow, and is part of OPM.
For more information visit: https://opm-project.org 

Flow Version     =  2026.04-pre (Python JAX Engine Parity)
Machine name     =  Python-VM
Operating system =  Linux x86_64
Simulation started on {self.start_time.strftime('%d-%m-%Y at %H:%M:%S')} hrs
Using 1 Python processes with JAX Automatic Differentiation backend
Parameters used by Flow:
# [known parameters which were specified at run-time]
EclDeckFileName="{self.model.file_path if hasattr(self.model, 'file_path') else 'Unknown.DATA'}"
# [parameters which were specified at compile-time]
EnableEclOutput="1"
EnableGravity="1"
EnableTerminalOutput="1"

Reading deck file '{self.model.file_path if hasattr(self.model, 'file_path') else 'Unknown.DATA'}'
3 fluid phases are active

Processing grid
Total number of active cells: {self.model.grid.nx * self.model.grid.ny * self.model.grid.nz}

================ Starting main simulation loop ===============

"""
        self.prt_file.write(header)
        self.prt_file.flush()
        self.dbg_file.write(header)
        self.dbg_file.flush()
        
    def log_time_step_start(self, step_idx, step_size, current_time, date_str):
        msg = f"\nStarting time step {step_idx}, stepsize {step_size:g} days, at day {current_time:g}/3653, date = {date_str}\n"
        self.prt_file.write(msg)
        self.dbg_file.write(msg)
        
    def log_newton_outer(self):
        msg = "Iter    MB(O)      MB(W)      MB(G)      CNV(O)     CNV(W)     CNV(G) \n"
        self.dbg_file.write(msg)

    def log_newton_iter(self, iter_idx, mb_o, mb_w, mb_g, cnv_o, cnv_w, cnv_g):
        msg = f"   {iter_idx}  {mb_o:.3e}  {mb_w:.3e}  {mb_g:.3e}  {cnv_o:.3e}  {cnv_w:.3e}  {cnv_g:.3e}\n"
        self.dbg_file.write(msg)
        
    def log_newton_summary(self, newton_its, dt_step):
        # Time step took 0.0sec
        msg = f" Newton its= {newton_its}, linearizations= {newton_its+1} (0.0sec), linear its=  {newton_its} (0.0sec)\n"
        self.prt_file.write(msg)
        self.dbg_file.write(msg)
        
    def log_report_matrices(self, report_idx, total_reports, elapsed_days, target_days, date_str, field_rates, well_rates, cum_totals):
        # Format the big tables exactly like OPM PRT.
        
        # Report Step Header
        header = f"""
                              **************************************************************************
  WELLS    AT   {int(target_days):5d}  DAYS *                 SPE1 - CASE 1                                          *
  REPORT {report_idx:4d}    {date_str}  *                                             Flow  version 2026.04-pre  *
                              **************************************************************************

"""
        self.prt_file.write(header)
        self.dbg_file.write(header)
        
        # PRODUCTION REPORT
        p_table = f"""
======================================================= PRODUCTION REPORT =======================================================
 :  WELL  : LOCATION  :CTRL:    OIL    :   WATER   :    GAS    :   FLUID   :   WATER   : GAS/OIL  :  WAT/GAS   : BHP OR : THP OR :
 :  NAME  :  (I,J,K)  :MODE:   RATE    :   RATE    :   RATE    : RES.VOL.  :    CUT    :  RATIO   :   RATIO    :CON.PR. :BLK.PR. :
 :        :           :    :  STB/DAY  :  STB/DAY  : MSCF/DAY  :  RB/DAY   :           : MSCF/STB :  STB/MSCF  :  PSIA  :  PSIA  :
=================================================================================================================================
"""
        # Write Field
        fo_rate = field_rates.get('FOPR', 0.0)
        fg_rate = field_rates.get('FGPR', 0.0)
        frv = field_rates.get('FPRV', 0.0)
        fgor = fg_rate / fo_rate if fo_rate > 0 else 0.0
        p_table += f":FIELD   :           :    :{fo_rate:11.1f}:        0.0:{fg_rate:11.1f}:{frv:11.1f}:      0.000:{fgor:10.2f}:      0.0000:        :        :\n"
        p_table += f":G1      :           :    :{fo_rate:11.1f}:        0.0:{fg_rate:11.1f}:{frv:11.1f}:      0.000:{fgor:10.2f}:      0.0000:        :        :\n"
        
        for w_name, w_data in well_rates.items():
            if w_data['type'] == 'PROD':
                loc = f"{w_data['i']:2d}, {w_data['j']:2d}"
                orat = w_data['orat']
                grat = w_data['grat']
                resv = w_data['resv']
                bhp = w_data['bhp']
                gor = grat / orat if orat > 0 else 0.0
                p_table += f":{w_name[:7]:7s} : {loc:10s}:ORAT:{orat:11.1f}:       -0.0:{grat:11.1f}:{resv:11.1f}:     -0.000:{gor:10.2f}:      0.0000:{bhp:8.1f}:     0.0:\n"
        p_table += " :--------:-----------:----:-----------:-----------:-----------:-----------:-----------:----------:------------:--------:--------:\n"
        
        self.prt_file.write(p_table)
        self.dbg_file.write(p_table)
        
        # INJECTION REPORT
        i_table = f"""
============================================= INJECTION REPORT ==============================================
 :  WELL  : LOCATION  : CTRL : CTRL : CTRL :    OIL    :   WATER   :    GAS    :   FLUID   : BHP OR : THP OR :
 :  NAME  :  (I,J,K)  : MODE : MODE : MODE :   RATE    :   RATE    :   RATE    : RES.VOL.  :CON.PR. :BLK.PR. :
 :        :           : OIL  : WAT  : GAS  :  STB/DAY  :  STB/DAY  : MSCF/DAY  :  RB/DAY   :  PSIA  :  PSIA  :
=============================================================================================================
"""
        fg_inj = field_rates.get('FGIR', 0.0)
        firv = field_rates.get('FIRV', 0.0)
        i_table += f":FIELD   :           :      :      :      :        0.0:        0.0:{fg_inj:11.1f}:{firv:11.1f}:        :        :\n"
        i_table += f":G1      :           :      :      :      :        0.0:        0.0:{fg_inj:11.1f}:{firv:11.1f}:        :        :\n"
        
        for w_name, w_data in well_rates.items():
            if w_data['type'] == 'INJ':
                loc = f"{w_data['i']:2d}, {w_data['j']:2d}"
                grat = w_data['grat']
                resv = w_data['resv']
                bhp = w_data['bhp']
                i_table += f":{w_name[:7]:7s} : {loc:10s}:      :      :  GRAT:        0.0:        0.0:{grat:11.1f}:{resv:11.1f}:{bhp:8.1f}:     0.0:\n"
        i_table += " :--------:-----------:------:------:------:-----------:-----------:-----------:-----------:--------:--------:\n"
        
        self.prt_file.write(i_table)
        self.dbg_file.write(i_table)
        
        # CUMULATIVE TOTALS
        c_table = f"""
============================================== CUMULATIVE PRODUCTION/INJECTION TOTALS ==============================================
 :  WELL  : LOCATION  :  WELL  :CTRL:    OIL    :   WATER   :    GAS    :   Prod    :    OIL    :   WATER   :    GAS    :    INJ    :
 :  NAME  :  (I,J,K)  :  TYPE  :MODE:   PROD    :   PROD    :   PROD    : RES.VOL.  :    INJ    :    INJ    :    INJ    : RES.VOL.  :
 :        :           :        :    :   MSTB    :   MSTB    :   MMSCF   :    MRB    :   MSTB    :   MSTB    :   MMSCF   :    MRB    :
====================================================================================================================================
"""
        cop = cum_totals.get('FOPT', 0.0) / 1e3
        cgp = cum_totals.get('FGPT', 0.0) / 1e3
        cpv = cum_totals.get('FPRV_CUM', 0.0) / 1e3
        cgi = cum_totals.get('FGIT', 0.0) / 1e3
        civ = cum_totals.get('FIRV_CUM', 0.0) / 1e3
        
        c_table += f":FIELD   :           :        :    :{cop:11.1f}:        0.0:{cgp:11.1f}:{cpv:11.1f}:        0.0:        0.0:{cgi:11.1f}:{civ:11.1f}:\n"
        c_table += f":G1      :           :        :    :{cop:11.1f}:        0.0:{cgp:11.1f}:{cpv:11.1f}:        0.0:        0.0:{cgi:11.1f}:{civ:11.1f}:\n"
        
        for w_name, w_data in well_rates.items():
            loc = f"{w_data['i']:2d}, {w_data['j']:2d}"
            wcop = w_data['cum_oil'] / 1e3
            wcgp = w_data['cum_gas_prod'] / 1e3
            wcpv = w_data['cum_resv_prod'] / 1e3
            wcgi = w_data['cum_gas_inj'] / 1e3
            wciv = w_data['cum_resv_inj'] / 1e3
            
            if w_data['type'] == 'PROD':
                c_table += f":{w_name[:7]:7s} : {loc:10s}:    PROD:ORAT:{wcop:11.1f}:        0.0:{wcgp:11.1f}:{wcpv:11.1f}:        0.0:        0.0:        0.0:        0.0:\n"
            else:
                c_table += f":{w_name[:7]:7s} : {loc:10s}:     INJ:GRAT:        0.0:        0.0:        0.0:        0.0:        0.0:        0.0:{wcgi:11.1f}:{wciv:11.1f}:\n"
                
        c_table += " :--------:-----------:--------:----:-----------:-----------:-----------:-----------:-----------:-----------:-----------:-----------:\n"
        
        self.prt_file.write(c_table)
        self.dbg_file.write(c_table)
        
        self.prt_file.flush()
        self.dbg_file.flush()

    def close(self):
        self.prt_file.close()
        self.dbg_file.close()

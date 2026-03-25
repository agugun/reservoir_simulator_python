#!/usr/bin/env python3
"""
OPM Flow vs. Python Reservoir Simulator — Comparison & Visualization Tool
==========================================================================
Usage:
    python3 tools/compare_results.py [--opm-dir DIR] [--py-dir DIR] [--output-dir DIR]

Defaults:
    --opm-dir   comparison/opm_run/
    --py-dir    data/sample/
    --output-dir tools/output/

Produces 3 saved figures:
    fig1_time_series.png   — Field summary quantities (FOPR, FGPR, FOPT, BHP)
    fig2_grid_snapshot.png — Cross-section heatmaps (PRESSURE, SGAS) at final step
    fig3_error_map.png     — Absolute difference |P_OPM - P_Python| spatial map
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── colour palette ─────────────────────────────────────────────────────────────
OPM_COLOR    = "#E63946"   # vibrant red
PY_COLOR     = "#457B9D"   # steel blue
BG_COLOR     = "#0D1117"   # dark background
PANEL_COLOR  = "#161B22"
GRID_COLOR   = "#30363D"
TEXT_COLOR   = "#C9D1D9"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   PANEL_COLOR,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "legend.framealpha": 0.2,
    "legend.edgecolor": GRID_COLOR,
})


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_case(directory: str) -> str:
    """Return the base path (no extension) of the .SMSPEC file in *directory*."""
    for f in os.listdir(directory):
        if f.upper().endswith(".SMSPEC"):
            base = f[:-6]  # strip exactly the last 6 chars (.SMSPEC)
            return os.path.join(directory, base)
    raise FileNotFoundError(f"No .SMSPEC found in {directory}")


def load_summary(directory: str) -> dict:
    """Load all available time-series vectors from a SMSPEC+UNSMRY pair."""
    from resdata.summary import Summary
    case = _find_case(directory)
    sm = Summary(case)

    vectors = {}
    all_keys = list(sm.keys())

    # Load all field and well scalars using the non-deprecated API
    field_keys = [k for k in all_keys if not k.startswith(("B", "R", "S")) or k.startswith(("WBHP", "WOPR", "WGPR", "WGIR", "WGOR"))]
    for key in field_keys:
        try:
            vectors[key] = sm.numpy_vector(key)
        except Exception:
            pass

    # Time axis – sm.days returns plain floats in this version
    try:
        days = np.asarray(sm.days, dtype=float)
    except Exception:
        days = np.array([])
    vectors["TIME"] = days

    # Derive cumulative FOPT by integrating FOPR over time
    if "FOPT" not in vectors and "FOPR" in vectors and len(days) > 1:
        fopr = vectors["FOPR"]
        vectors["FOPT"] = np.array([np.trapezoid(fopr[:i+1], days[:i+1]) for i in range(len(days))])

    # Derive cumulative FGPT from FGOR * FOPR
    if "FGPR" not in vectors and "FGOR" in vectors and "FOPR" in vectors:
        vectors["FGPR"] = vectors["FOPR"] * vectors["FGOR"]
    if "FGPT" not in vectors and "FGPR" in vectors and len(days) > 1:
        fgpr = vectors["FGPR"]
        vectors["FGPT"] = np.array([np.trapezoid(fgpr[:i+1], days[:i+1]) for i in range(len(days))])

    return vectors


def load_grid_restart(directory: str, step: int = -1) -> dict:
    """Load 3D arrays from the last restart step by scanning UNRST sequentially."""
    from resdata.resfile import ResdataFile
    from resdata.grid import Grid

    # Locate files by scanning the folder (case-insensitive)
    egrid_path = None
    unrst_path = None
    for f in os.listdir(directory):
        fu = f.upper()
        if fu.endswith(".EGRID"): egrid_path = os.path.join(directory, f)
        if fu.endswith(".UNRST"): unrst_path = os.path.join(directory, f)

    if not egrid_path:
        raise FileNotFoundError(f"Missing EGRID in {directory}")
    if not unrst_path:
        raise FileNotFoundError(f"Missing UNRST in {directory}")

    grid = Grid(egrid_path)
    nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
    N = nx * ny * nz

    rst = ResdataFile(unrst_path)
    data = {"nx": nx, "ny": ny, "nz": nz}

    # Collect keyword names present in the file; get the LAST occurrence of each
    last_kw = {}   # kw_name -> numpy array
    for kw in rst:
        name = kw.getName()
        if name in ("PRESSURE", "SGAS", "SOIL", "SWAT") and len(kw) == N:
            last_kw[name] = kw.numpy_copy().reshape((nz, ny, nx))

    data.update(last_kw)
    return data


def _ax_style(ax, title: str):
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=8)
    ax.grid(True)
    ax.tick_params(labelsize=9)


# ── Figure 1 — Time-Series Dashboard ──────────────────────────────────────────

def figure1_time_series(opm: dict, py: dict, out_path: str):
    panels = [
        ("FOPR",  "Field Oil Production Rate",  "STB/day"),
        ("FGPR",  "Field Gas Production Rate",  "MSCF/day"),
        ("FOPT",  "Cum. Oil Production",         "STB"),
        ("FGPT",  "Cum. Gas Production",         "MSCF"),
    ]

    # pick well-BHP key (prioritize producer)
    bhp_key = next((k for k in opm if "WBHP:PROD" in k.upper()), None)
    if not bhp_key:
        bhp_key = next((k for k in opm if "WBHP" in k.upper()), None)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("OPM Flow  vs  Python Simulator — Field Summary",
                 fontsize=15, color=TEXT_COLOR, y=0.98, fontweight="bold")

    ncols = 3 if bhp_key else 2
    nrows = 2
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.42, wspace=0.35)

    def _plot(ax, key, title, unit):
        t_o = opm.get("TIME", [])
        t_p = py.get("TIME", [])
        if key in opm:
            ax.plot(t_o, opm[key], color=OPM_COLOR,  lw=2, label="OPM Flow")
        if key in py:
            ax.plot(t_p, py[key],  color=PY_COLOR,   lw=2, label="Python Sim", ls="--")
        _ax_style(ax, title)
        ax.set_xlabel("Time  (days)", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        ax.legend(fontsize=8)

    positions = [(0,0),(0,1),(1,0),(1,1)]
    for (r, c), (key, title, unit) in zip(positions, panels):
        ax = fig.add_subplot(gs[r, c])
        _plot(ax, key, title, unit)

    if bhp_key:
        ax_bhp = fig.add_subplot(gs[:, 2])
        t_o = opm.get("TIME", [])
        t_p = py.get("TIME", [])
        if bhp_key in opm:
            ax_bhp.plot(t_o, opm[bhp_key], color=OPM_COLOR, lw=2, label="OPM Flow")
        py_bhp = next((k for k in py if "WBHP" in k), None)
        if py_bhp:
            ax_bhp.plot(t_p, py[py_bhp], color=PY_COLOR, lw=2, label="Python Sim", ls="--")
        _ax_style(ax_bhp, f"Producer BHP  ({bhp_key})")
        ax_bhp.set_xlabel("Time  (days)", fontsize=9)
        ax_bhp.set_ylabel("psia", fontsize=9)
        ax_bhp.legend(fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ── Figure 2 — Grid Snapshot Heatmaps ─────────────────────────────────────────

def figure2_grid_snapshot(opm: dict, py: dict, out_path: str, layer: int = 0):
    vars_ = [("PRESSURE", "Pressure  (psia)", "plasma"),
             ("SGAS",     "Gas Saturation",    "viridis")]

    ncols = 4
    fig, axes = plt.subplots(len(vars_), ncols,
                             figsize=(20, 6 * len(vars_)),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.35})
    fig.suptitle(f"OPM Flow  vs  Python Simulator — Grid Snapshot (Layer {layer+1}, Final Step)",
                 fontsize=14, color=TEXT_COLOR, y=1.01, fontweight="bold")

    for row, (kw, label, cmap) in enumerate(vars_):
        opm_slice = opm.get(kw, None)
        py_slice  = py.get(kw, None)

        # shared colour scale
        vals = []
        if opm_slice is not None: vals.append(opm_slice[layer])
        if py_slice  is not None: vals.append(py_slice[layer])
        vmin = np.nanmin([v.min() for v in vals]) if vals else 0
        vmax = np.nanmax([v.max() for v in vals]) if vals else 1
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        titles = ["OPM Flow", "Python Sim", "Difference  |OPM − Py|", "% Relative Diff"]
        ax_row = axes[row] if len(vars_) > 1 else axes

        # OPM
        im0 = ax_row[0].imshow(opm_slice[layer] if opm_slice is not None else np.zeros((1,1)),
                                cmap=cmap, norm=norm, origin="lower", aspect="auto")
        ax_row[0].set_title(f"{label}\n{titles[0]}", fontsize=10, color=TEXT_COLOR)

        # Python
        im1 = ax_row[1].imshow(py_slice[layer] if py_slice is not None else np.zeros((1,1)),
                                cmap=cmap, norm=norm, origin="lower", aspect="auto")
        ax_row[1].set_title(f"{label}\n{titles[1]}", fontsize=10, color=TEXT_COLOR)

        # Absolute difference
        if opm_slice is not None and py_slice is not None:
            diff = np.abs(opm_slice[layer] - py_slice[layer])
            im2 = ax_row[2].imshow(diff, cmap="hot", origin="lower", aspect="auto")
            fig.colorbar(im2, ax=ax_row[2], fraction=0.046)
        ax_row[2].set_title(f"{label}\n{titles[2]}", fontsize=10, color=TEXT_COLOR)

        # Relative % difference
        if opm_slice is not None and py_slice is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(np.abs(opm_slice[layer]) > 1e-6,
                               100.0 * np.abs(opm_slice[layer] - py_slice[layer]) / np.abs(opm_slice[layer]),
                               0.0)
            im3 = ax_row[3].imshow(rel, cmap="RdYlGn_r", vmin=0, vmax=10,
                                    origin="lower", aspect="auto")
            fig.colorbar(im3, ax=ax_row[3], fraction=0.046, label="%")
        ax_row[3].set_title(f"{label}\n{titles[3]}", fontsize=10, color=TEXT_COLOR)

        fig.colorbar(im1, ax=ax_row[1], fraction=0.046)

        for ax in ax_row:
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            ax.set_facecolor(PANEL_COLOR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ── Figure 3 — Error Statistics Bar Chart ─────────────────────────────────────

def figure3_error_stats(opm: dict, py: dict, out_path: str):
    """Bar chart comparing mean/max absolute error across all layers for PRESSURE and SGAS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("OPM Flow  vs  Python Simulator — Spatial Error Statistics",
                 fontsize=14, color=TEXT_COLOR, y=1.01, fontweight="bold")

    for ax, kw, unit in zip(axes, ["PRESSURE", "SGAS"], ["psia", "fraction"]):
        o = opm.get(kw)
        p = py.get(kw)
        if o is None or p is None:
            ax.text(0.5, 0.5, f"No {kw} data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color=TEXT_COLOR)
            continue

        nz = o.shape[0]
        means, maxs = [], []
        for k in range(nz):
            diff = np.abs(o[k] - p[k])
            means.append(diff.mean())
            maxs.append(diff.max())

        x = np.arange(nz) + 1
        w = 0.38
        bars_mean = ax.bar(x - w/2, means, w, color=OPM_COLOR, alpha=0.85, label="Mean |error|")
        bars_max  = ax.bar(x + w/2, maxs,  w, color=PY_COLOR,  alpha=0.85, label="Max |error|")

        _ax_style(ax, f"{kw} Error by Layer")
        ax.set_xlabel("Layer  (k)", fontsize=10)
        ax.set_ylabel(f"Absolute Error  ({unit})", fontsize=10)
        ax.set_xticks(x)
        ax.legend(fontsize=9)
        ax.grid(axis="y")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ── Figure 4 — Time-series % divergence ───────────────────────────────────────

def figure4_divergence(opm: dict, py: dict, out_path: str):
    """Shows how OPM and Python cumulative production diverge over time as %."""
    keys = [("FOPT", "Cum. Oil Production"), ("FGPT", "Cum. Gas Production")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("OPM Flow  vs  Python Simulator — Cumulative Production % Divergence",
                 fontsize=13, color=TEXT_COLOR, y=1.01, fontweight="bold")

    for ax, (key, label) in zip(axes, keys):
        t_o = np.array(opm.get("TIME", []))
        t_p = np.array(py.get("TIME", []))
        o_v = np.array(opm.get(key, []))
        p_v = np.array(py.get(key, []))

        # Interpolate Python onto OPM time axis
        if len(t_o) > 0 and len(t_p) > 0 and len(o_v) > 0 and len(p_v) > 0:
            p_interp = np.interp(t_o, t_p, p_v)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(o_v > 1e-6, 100.0 * (p_interp - o_v) / o_v, 0.0)
            ax.fill_between(t_o, pct, 0,
                            where=(pct >= 0), color=PY_COLOR, alpha=0.4, label="Python > OPM")
            ax.fill_between(t_o, pct, 0,
                            where=(pct < 0),  color=OPM_COLOR, alpha=0.4, label="Python < OPM")
            ax.plot(t_o, pct, color=TEXT_COLOR, lw=1.2)
            ax.axhline(0, color=GRID_COLOR, lw=1)

        _ax_style(ax, f"{label}  —  % Divergence  (Py − OPM) / OPM")
        ax.set_xlabel("Time  (days)", fontsize=9)
        ax.set_ylabel("Relative error  (%)", fontsize=9)
        ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OPM vs Python Sim Comparison")
    parser.add_argument("--opm-dir",    default="data/spe1/",
                        help="Benchmark OPM results directory")
    parser.add_argument("--py-dir",     default="output/spe1/",
                        help="Python simulator results directory")
    parser.add_argument("--output-dir", default="tools/output/",
                        help="Directory to save generated figures")
    parser.add_argument("--layer",      type=int, default=0,
                        help="Grid layer (0-indexed) for spatial snapshot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  OPM Flow vs Python Reservoir Simulator — Comparison")
    print("="*60)

    # ── Load summary time-series ──────────────────────────────────
    print("\n[1/4]  Loading summary time-series...")
    try:
        opm_sum = load_summary(args.opm_dir)
        print(f"       OPM: {len(opm_sum.get('TIME',[]))} time steps  |  "
              f"keys: {[k for k in opm_sum if k != 'TIME']}")
    except Exception as e:
        print(f"       ⚠  OPM summary load failed: {e}"); opm_sum = {}

    try:
        py_sum = load_summary(args.py_dir)
        print(f"       Py : {len(py_sum.get('TIME',[]))} time steps  |  "
              f"keys: {[k for k in py_sum if k != 'TIME']}")
    except Exception as e:
        print(f"       ⚠  Python summary load failed: {e}"); py_sum = {}

    # ── Load grid restart data ────────────────────────────────────
    print("\n[2/4]  Loading restart grid data (final step)...")
    try:
        opm_grid = load_grid_restart(args.opm_dir)
        print(f"       OPM grid: {opm_grid['nx']}×{opm_grid['ny']}×{opm_grid['nz']}  |  "
              f"arrays: {[k for k in opm_grid if k not in ('nx','ny','nz')]}")
    except Exception as e:
        print(f"       ⚠  OPM grid load failed: {e}"); opm_grid = {}

    try:
        py_grid = load_grid_restart(args.py_dir)
        print(f"       Py  grid: {py_grid['nx']}×{py_grid['ny']}×{py_grid['nz']}  |  "
              f"arrays: {[k for k in py_grid if k not in ('nx','ny','nz')]}")
    except Exception as e:
        print(f"       ⚠  Python grid load failed: {e}"); py_grid = {}

    # ── Generate figures ──────────────────────────────────────────
    print("\n[3/4]  Generating figures...")

    figure1_time_series(opm_sum, py_sum,
                        os.path.join(args.output_dir, "fig1_time_series.png"))

    figure2_grid_snapshot(opm_grid, py_grid,
                          os.path.join(args.output_dir, "fig2_grid_snapshot.png"),
                          layer=args.layer)

    figure3_error_stats(opm_grid, py_grid,
                        os.path.join(args.output_dir, "fig3_error_stats.png"))

    figure4_divergence(opm_sum, py_sum,
                       os.path.join(args.output_dir, "fig4_divergence.png"))

    print(f"\n[4/4]  All figures saved to: {os.path.abspath(args.output_dir)}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

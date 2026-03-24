# Simulator Software Architecture

The Python reservoir simulator is modularly decoupled across four fundamental layers, maintaining strict structural responsibilities matching industrial flow protocols.

## 1. Ingestion Layer (`src/io/eclipse_parser.py`)

*   **Deck Processor**: Rapidly reads and tokenizes legacy Fortran-based `.DATA` Eclipse formats natively.
*   **Keyword Router**: Identifies arrays (`PORO`, `PERMX`, `ZCOR`), structures (`RUNSPEC`, `GRID`), fluid interpolators (`PVTO`, `SGWFN`), and historical mappings (`WCONPROD`, `WCONINJE`).
*   **Vector Construction**: Resolves compressed multiplier traces (`10*1.0`) structurally unrolling arrays matching the strict global grid bounds perfectly.

## 2. Physics Core Layer (`src/core/`)

*   **`grid.py`**: Controls global dimensions, $X/Y/Z$ geometric Cartesian spacing depths, cell active bounds (`ACTNUM`), and generic Transmissibility $T_{ij}$ constant connections map.
*   **`rock.py`**: Resolves instantaneous Pore Volumes dynamically integrating $c_r$ static rock compressibilities tracking $P$.
*   **`fluid.py`**: Executes $O(1)$ interpolations matching thermodynamic states identically. Exclusively utilizes pure mathematical non-mutant representations directly bridging Google JAX operations.
*   **`simulator.py`**: The overarching algebraic supervisor. Combines internal physics, upwinding logic, fluid-flow, matrix Jacobian compilations ($J$), tracking adaptive time-stepping iterations explicitly continuously.

## 3. Top-Level Execution Engine (`main.py`)

The overarching dynamic controller explicitly mapping the temporal span of the simulated projection.

1.  Determines initial **Hydrostatic Equilibration** balancing gravity logic perfectly against depth coordinates structurally.
2.  Forces strict discrete target reporting steps synchronizing explicitly to exact day endpoints statically requested by schedule tracking.
3.  Automatically chops the discrete target bounds ($\Delta t$) significantly avoiding extreme physical non-linearities causing structural Newton divergences conditionally.

## 4. Reporting & Exporter Pipeline (`src/io/`)

*   **`eclipse_writer.py`**: Output structural binary geometries tracking raw industrial legacy formats natively. Directly outputs `.EGRID`, `.INIT`, `.UNRST`, `.SMSPEC`, and `.ESMRY` mapping the exact binary footprint required perfectly driving 3rd-party renderers exactly (e.g. `ResInsight`).
*   **`report_writer.py`**: Formats visually elegant, highly exact generic execution traces explicitly inside identical native representations. Generates natively `.PRT` string matrix accumulations and generic `.DBG` step convergence iterations structurally compliant exactly limits matched directly towards OPM-Flow legacy compatibility arrays.

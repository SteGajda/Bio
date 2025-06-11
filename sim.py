#!/usr/bin/env python3
"""sim.py
===========
Reusable OpenMM driver for a coarse‑grained 1‑D polymer with optional
Coulomb interaction defined by an external vector of charges *and* an
optional on‑the‑fly Ising update.

Typical notebook usage
----------------------
```python
import numpy as np
from sim import run_simulation

# prepare charges (here: ±1 spins from some Ising model you calculated earlier)
charges = np.random.choice([-1, 1], size=100)

# run MD with *fixed* charges
data = run_simulation(charges,
                      kc=1.0,          # prefactor in kJ·nm/mol
                      n_blocks=500,
                      steps_per_block=2000,
                      dynamic=False,   # ← no Ising updates inside MD
                      out_prefix='ising_fixed')

# run MD where the same vector acts as *initial* spins that keep evolving
run_simulation(charges,
               kc=1.0,
               n_blocks=500,
               steps_per_block=2000,
               dynamic=True,    # ← enable Metropolis sweeps
               out_prefix='ising_dynamic')
```
The routine returns a dictionary with the measured end‑to‑end distance array
and names of the generated trajectory/data files so you can post‑process them
from the notebook.
"""

from __future__ import annotations

# ────────── standard library ──────────
import os
from sys import stdout

# ────────── third‑party ──────────
import numpy as np
from tqdm import tqdm

import openmm as mm
import openmm.unit as u
from openmm.app import (
    PDBxFile,
    ForceField,
    Simulation,
    DCDReporter,
    StateDataReporter,
)
from mdtraj.reporters import HDF5Reporter

# ────────── local helper routines (provided in earlier tasks) ──────────
from utils import line, write_mmcif, generate_psf  # type: ignore

# ──────────── constants ────────────
N_BEADS = 100                # chain length (keep in sync with utils helpers!)
DEFAULT_TEMP = 310 * u.kelvin
TIME_STEP = 100 * u.femtosecond
INIT_CIF = "init_struct.cif"

# ────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────

def _prepare_initial_structure(n_beads: int = N_BEADS) -> PDBxFile:  # type: ignore[name‑defined]
    """Write (if necessary) and cache a straight chain as mmCIF, return PDBxFile."""
    if not os.path.exists(INIT_CIF):
        pts = line(n_beads)                      # (nm) coordinates along x axis
        write_mmcif(pts, INIT_CIF)
        generate_psf(n_beads, "LE_init_struct.psf")  # purely for viz; not needed by OpenMM
    return PDBxFile(INIT_CIF)


def _build_system(charges: np.ndarray, kc: float):
    """Create OpenMM System, Integrator and return also Coulomb force handle."""
    pdb = _prepare_initial_structure()

    # classic excluded‑volume from earlier XML; no electrostatics there
    ff = ForceField("forcefields/classic_sm_ff.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=mm.app.NoCutoff)

    # 1. Harmonic bonds (successive + loop)
    bond_force = mm.HarmonicBondForce()
    system.addForce(bond_force)
    for i in range(N_BEADS - 1):
        bond_force.addBond(
            i,
            i + 1,
            0.1 * u.nanometer,
            300_000.0 * u.kilojoule_per_mole / u.nanometer**2,
        )
    bond_force.addBond(10, 70, 0.1 * u.nanometer, 300_000.0 * u.kilojoule_per_mole / u.nanometer**2)

    # 2. Bending rigidity via harmonic angle
    angle_force = mm.HarmonicAngleForce()
    system.addForce(angle_force)
    for i in range(N_BEADS - 2):
        angle_force.addAngle(
            i,
            i + 1,
            i + 2,
            np.pi * u.radian,
            500 * u.kilojoule_per_mole / u.radian**2,
        )

    # 3. User‑defined Coulomb term (signed kc)
       # --- tworzenie siły Coulomba ---
    coulomb = mm.CustomNonbondedForce("kc*q1*q2/r")
    coulomb.addPerParticleParameter("q")     # jeden parametr na cząstkę
    coulomb.addGlobalParameter("kc", kc)
    # wykluczamy pary 1-2 (bondy)
    coulomb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    coulomb.setCutoffDistance(1.0*u.nanometer)
    system.addForce(coulomb)
    
    # --- pierwsze ustawienie ładunków ---
    for i, q in enumerate(initial_charges):
        coulomb.addParticle([float(q)])


    # Langevin integrator
    integrator = mm.LangevinIntegrator(DEFAULT_TEMP, 0.05 / u.picosecond, TIME_STEP)

    return system, integrator, pdb, coulomb


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────

def run_simulation(
    charges: np.ndarray | list,
    kc: float = 1.0,
    *,
    n_blocks: int = 1000,
    steps_per_block: int = 1000,
    dynamic: bool = False,
    seed: int | None = 1234,
    out_prefix: str = "run",
):
    """Run coarse‑grained polymer MD with user‑supplied charges / spins.

    Parameters
    ----------
    charges
        1‑D iterable of length *N_BEADS* (100 by default).  Values are used
        verbatim as `qi` in the potential.  When `dynamic=True`, the vector must
        consist solely of ±1 and is treated as an *initial* Ising configuration
        that evolves during the simulation.
    kc
        Prefactor in energy units kJ·nm/mol.  Positive → like‑sign attraction
        (opposite of electrostatics); negative → conventional electrostatics.
    n_blocks, steps_per_block
        Simulation length = ``n_blocks * steps_per_block`` MD steps.
    dynamic
        If *True* perform a single‑spin Metropolis sweep (1‑D nearest‑neighbour
        Ising, J=1, k_B T = 1) after every block and update charges in situ.
        If *False* the supplied `charges` remain fixed.
    seed
        RNG seed used only for the internal Ising updates when `dynamic=True`.
    out_prefix
        Basename for output files: ``{out_prefix}.dcd`` (trajectory),
        ``{out_prefix}.h5`` (mdtraj‑HDF5), ``{out_prefix}.npz`` (end‑to‑end data).

    Returns
    -------
    dict
        Keys: ``end2end`` (np.ndarray [n_blocks]), ``trajectory`` (str path).
    """

    charges = np.asarray(charges, dtype=float)
    if charges.shape != (N_BEADS,):
        raise ValueError(f"charges must be length {N_BEADS}, got shape {charges.shape}")

    if dynamic and not np.all(np.isin(charges, [-1, 1])):
        raise ValueError("When dynamic=True, initial charges must be ±1 spins.")

    system, integrator, pdb, coulomb = _build_system(charges, kc)

    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(tolerance=1e-3)

    # reporters
    simulation.reporters.append(StateDataReporter(stdout, steps_per_block, step=True, potentialEnergy=True, temperature=True))
    simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", steps_per_block))
    simulation.reporters.append(HDF5Reporter(f"{out_prefix}.h5", steps_per_block, positions=True, time=True))

    # storage arrays
    end2end: list[float] = []

    # Ising apparatus (only if dynamic=True)
    rng = np.random.default_rng(seed)
    spins = charges.astype(int).copy()  # copied even if dynamic=False for uniformity

    def metropolis_single_flip() -> None:
        i = rng.integers(0, N_BEADS)
        left, right = (i - 1) % N_BEADS, (i + 1) % N_BEADS
        delta_E = 2 * spins[i] * (spins[left] + spins[right])  # J = 1, k_B T = 1
        if delta_E <= 0 or rng.random() < np.exp(-delta_E):
            spins[i] *= -1

    # ───────── main loop ─────────
    for _ in tqdm(range(n_blocks), unit="blk", desc="Blocks"):
        simulation.step(steps_per_block)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        end2end.append(float(np.linalg.norm(pos[-1] - pos[0])))

        if dynamic:
            # perform ~one full sweep
            for _ in range(N_BEADS):
                metropolis_single_flip()
            # update charges in force & context
            for idx, s in enumerate(spins):
                coulomb.setParticleParameters(idx, [float(s)])
            coulomb.updateParametersInContext(simulation.context)

    # ───────── save observables ─────────
    np.savez(f"{out_prefix}.npz", end2end=np.asarray(end2end))

    return {"end2end": np.asarray(end2end), "trajectory": f"{out_prefix}.dcd"}


# ────────────────── quick CLI test ──────────────────
if __name__ == "__main__":
    # fixed positive charges as sanity‑check, tiny run
    run_simulation(np.ones(N_BEADS), kc=1.0, n_blocks=10, steps_per_block=1000, dynamic=False, out_prefix="test")

# ─────────────────────────────────────────────────────────────
# External spin trajectory → OpenMM
#   spin_seq.shape == (n_frames, N_BEADS)
#   Każda klatka trafia do Coulomba po jednym bloku MD.
# ─────────────────────────────────────────────────────────────
def run_simulation_with_spin_sequence(
    spin_seq: np.ndarray,
    kc: float = 1.0,
    *,
    steps_per_frame: int = 1000,
    out_prefix: str = "ising_ext",
):
    spin_seq = np.asarray(spin_seq, dtype=float)
    n_frames, n_beads = spin_seq.shape
    if n_beads != N_BEADS:
        raise ValueError(f"spin_seq columns ({n_beads}) ≠ N_BEADS ({N_BEADS})")

    system, integrator, pdb, coulomb = _build_system(spin_seq[0], kc)
    sim = Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(tolerance=1e-3)

    sim.reporters.append(DCDReporter(f"{out_prefix}.dcd", steps_per_frame))
    sim.reporters.append(HDF5Reporter(f"{out_prefix}.h5", steps_per_frame, positions=True))

    end2end = []

    for frame_idx in range(n_frames):
        if frame_idx:                # od 2-giej klatki aktualizujemy ładunki
            for i, q in enumerate(spin_seq[frame_idx]):
                coulomb.setParticleParameters(i, [float(q)])
            coulomb.updateParametersInContext(sim.context)

        sim.step(steps_per_frame)

        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        end2end.append(float(np.linalg.norm(pos[-1] - pos[0])))

    np.savez(f"{out_prefix}.npz", end2end=np.asarray(end2end))
    return {"end2end": np.asarray(end2end), "trajectory": f"{out_prefix}.dcd"}


#!/usr/bin/env python3
"""
sim.py  –  OpenMM driver for coarse-grained 1-D polymer
=======================================================

* Umożliwia MD z: (1) wiązaniami harm., (2) sztywnością kątową,
  (3) excluded-volume z pliku XML, (4) dowolnym wektorem ładunków
  w potencjale U = kc · q1 · q2 / r.
* Tryb `dynamic=True` → po każdym bloku MD wykonywany jest pełny sweep
  Metropolisa (1-D Ising, J=1, kBT=1) i ładunki aktualizują się w locie.
* Funkcja `run_simulation_with_spin_sequence` pozwala zamiast tego
  podać **gotową trajektorię spinów** (np. przygotowaną w innym kodzie).
"""

from __future__ import annotations
# ────────── stdlib ──────────
import os
from sys import stdout
# ────────── third-party ──────────
import numpy as np
from tqdm import tqdm
import openmm as mm
import openmm.unit as u
from openmm.app import (
    PDBxFile, ForceField, Simulation,
    DCDReporter, StateDataReporter
)
from mdtraj.reporters import HDF5Reporter
# ────────── local helpers (z wcześniejszych zadań) ──────────
from utils import line, write_mmcif, generate_psf   # type: ignore

# ───────────── stałe ─────────────
N_BEADS      = 100
DEFAULT_TEMP = 310 * u.kelvin
TIME_STEP    = 100 * u.femtosecond
INIT_CIF     = "init_struct.cif"

# ─────────────────────────────────────────────────────────────
#  przygotowanie struktury startowej
# ─────────────────────────────────────────────────────────────
def _prepare_initial_structure(n_beads: int = N_BEADS) -> PDBxFile:  # type: ignore[name-defined]
    """Zapisuje (jednorazowo) prosty łańcuch jako mmCIF i zwraca PDBxFile."""
    if not os.path.exists(INIT_CIF):
        coords = line(n_beads)                # prosta linia po osi x
        write_mmcif(coords, INIT_CIF)
        generate_psf(n_beads, "LE_init_struct.psf")  # opcjonalne – tylko do VMD
    return PDBxFile(INIT_CIF)

# ─────────────────────────────────────────────────────────────
#  budowa systemu z podanym wektorem ładunków
# ─────────────────────────────────────────────────────────────
def _build_system(charges: np.ndarray, kc: float):
    pdb = _prepare_initial_structure()

    # excluded-volume z pliku XML (bez elektrostatyki)
    ff = ForceField("forcefields/classic_sm_ff.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=mm.app.NoCutoff)

    # 1) Harmonijne wiązania
    bond = mm.HarmonicBondForce();  system.addForce(bond)
    for i in range(N_BEADS - 1):
        bond.addBond(i, i + 1, 0.1 * u.nanometer,
                     300_000.0 * u.kilojoule_per_mole / u.nanometer**2)
    bond.addBond(10, 70, 0.1 * u.nanometer,
                 300_000.0 * u.kilojoule_per_mole / u.nanometer**2)

    # 2) Sztywność kątowa
    angle = mm.HarmonicAngleForce();  system.addForce(angle)
    for i in range(N_BEADS - 2):
        angle.addAngle(i, i + 1, i + 2,
                       np.pi * u.radian,
                       500 * u.kilojoule_per_mole / u.radian**2)

    # 3) Coulomb  kc * q1 * q2 / r
    coulomb = mm.CustomNonbondedForce("kc*q1*q2/r")
    coulomb.addPerParticleParameter("q")     # nazwa parametru
    coulomb.addGlobalParameter("kc", kc)
    coulomb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    coulomb.setCutoffDistance(1.0 * u.nanometer)
    system.addForce(coulomb)

    #  inicjalizacja parametrów cząstek
    for q in charges:
        coulomb.addParticle([float(q)])

    integrator = mm.LangevinIntegrator(
        DEFAULT_TEMP, 0.05 / u.picosecond, TIME_STEP
    )
    return system, integrator, pdb, coulomb

# ─────────────────────────────────────────────────────────────
#  główna funkcja – MD + opcjonalny Ising
# ─────────────────────────────────────────────────────────────
def run_simulation(
    charges, *, kc=1.0,
    n_blocks=1000, steps_per_block=1000,
    dynamic=False, seed: int|None=1234,
    out_prefix="run"
):
    """MD z opcjonalną ewolucją ładunków metodą Isinga."""
    charges = np.asarray(charges, dtype=float)
    if charges.shape != (N_BEADS,):
        raise ValueError(f"charges must be length {N_BEADS}, got {charges.shape}")
    if dynamic and not np.all(np.isin(charges, [-1, 1])):
        raise ValueError("dynamic=True wymaga spinów ±1")

    system, integrator, pdb, coulomb = _build_system(charges, kc)
    sim = Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(tolerance=1e-3)

    sim.reporters += [
        StateDataReporter(stdout, steps_per_block, step=True,
                          potentialEnergy=True, temperature=True),
        DCDReporter(f"{out_prefix}.dcd", steps_per_block),
        HDF5Reporter(f"{out_prefix}.h5", steps_per_block,
                     positions=True, time=True)
    ]

    spins = charges.astype(int).copy()
    rng = np.random.default_rng(seed)
    def metropolis_flip():
        i = rng.integers(0, N_BEADS)
        dE = 2 * spins[i] * (spins[(i-1)%N_BEADS] + spins[(i+1)%N_BEADS])
        if dE <= 0 or rng.random() < np.exp(-dE):
            spins[i] *= -1

    end2end = []
    for _ in tqdm(range(n_blocks), unit="blk", desc="Blocks"):
        sim.step(steps_per_block)
        pos = sim.context.getState(getPositions=True)\
                  .getPositions(asNumpy=True).value_in_unit(u.nanometer)
        end2end.append(float(np.linalg.norm(pos[-1] - pos[0])))

        if dynamic:
            for _ in range(N_BEADS):  metropolis_flip()
            for i, s in enumerate(spins):
                coulomb.setParticleParameters(i, [float(s)])
            coulomb.updateParametersInContext(sim.context)

    np.savez(f"{out_prefix}.npz", end2end=np.asarray(end2end))
    return {"end2end": np.asarray(end2end),
            "trajectory": f"{out_prefix}.dcd"}

# ─────────────────────────────────────────────────────────────
#  alternatywa: gotowa trajektoria spinów
# ─────────────────────────────────────────────────────────────
def run_simulation_with_spin_sequence(
    spin_seq: np.ndarray, *, kc=1.0,
    steps_per_frame=1000, out_prefix="ising_ext"
):
    spin_seq = np.asarray(spin_seq, dtype=float)
    n_frames, n_cols = spin_seq.shape
    if n_cols != N_BEADS:
        raise ValueError(f"spin_seq columns {n_cols} != N_BEADS {N_BEADS}")

    system, integrator, pdb, coulomb = _build_system(spin_seq[0], kc)
    sim = Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(tolerance=1e-3)

    sim.reporters += [
        DCDReporter(f"{out_prefix}.dcd", steps_per_frame),
        HDF5Reporter(f"{out_prefix}.h5", steps_per_frame, positions=True)
    ]

    end2end = []
    for frame in range(n_frames):
        if frame:                       # od drugiej klatki aktualizujemy q
            for i, q in enumerate(spin_seq[frame]):
                coulomb.setParticleParameters(i, [float(q)])
            coulomb.updateParametersInContext(sim.context)

        sim.step(steps_per_frame)
        pos = sim.context.getState(getPositions=True)\
                  .getPositions(asNumpy=True).value_in_unit(u.nanometer)
        end2end.append(float(np.linalg.norm(pos[-1] - pos[0])))

    np.savez(f"{out_prefix}.npz", end2end=np.asarray(end2end))
    return {"end2end": np.asarray(end2end),
            "trajectory": f"{out_prefix}.dcd"}

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":        # szybki sanity-check
    run_simulation(np.ones(N_BEADS),
                   kc=1.0, n_blocks=5, steps_per_block=500,
                   dynamic=False, out_prefix="test")

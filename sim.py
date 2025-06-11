#!/usr/bin/env python3
import os, numpy as np, openmm as mm, openmm.unit as u
from openmm.app import PDBxFile, ForceField, Simulation, DCDReporter, StateDataReporter
from mdtraj.reporters import HDF5Reporter
from tqdm import tqdm
from utils import line, write_mmcif, generate_psf               # noqa: F401

N_BEADS      = 100
DEFAULT_TEMP = 310 * u.kelvin
TIME_STEP    = 100 * u.femtosecond
INIT_CIF     = "init_struct.cif"

def _prepare_initial_structure():
    if not os.path.exists(INIT_CIF):
        write_mmcif(line(N_BEADS), INIT_CIF)
        generate_psf(N_BEADS, "LE_init_struct.psf")
    return PDBxFile(INIT_CIF)

def _build_system(charges, kc):
    pdb     = _prepare_initial_structure()
    ff      = ForceField("forcefields/classic_sm_ff.xml")
    system  = ff.createSystem(pdb.topology, nonbondedMethod=mm.app.NoCutoff)

    bond = mm.HarmonicBondForce(); system.addForce(bond)
    for i in range(N_BEADS - 1):
        bond.addBond(i, i + 1, 0.1*u.nanometer, 300_000.0*u.kilojoule_per_mole/u.nanometer**2)
    bond.addBond(10, 70, 0.1*u.nanometer, 300_000.0*u.kilojoule_per_mole/u.nanometer**2)

    angle = mm.HarmonicAngleForce(); system.addForce(angle)
    for i in range(N_BEADS - 2):
        angle.addAngle(i, i + 1, i + 2, np.pi*u.radian, 500*u.kilojoule_per_mole/u.radian**2)

    coulomb = mm.CustomNonbondedForce("kc*q1*q2/r")
    coulomb.addPerParticleParameter("q")
    coulomb.addGlobalParameter("kc", kc)
    coulomb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    coulomb.setCutoffDistance(1.0*u.nanometer)
    for q in charges: coulomb.addParticle([float(q)])
    system.addForce(coulomb)

    integrator = mm.LangevinIntegrator(DEFAULT_TEMP, 0.05/u.picosecond, TIME_STEP)
    return system, integrator, pdb, coulomb

def run_simulation(charges, *, kc=1.0, n_blocks=1000, steps_per_block=1000,
                   dynamic=False, seed=1234, out_prefix="run"):
    charges = np.asarray(charges, float)
    if charges.shape != (N_BEADS,): raise ValueError
    if dynamic and not np.all(np.isin(charges, [-1, 1])): raise ValueError

    system, integrator, pdb, coulomb = _build_system(charges, kc)
    sim = Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(tolerance=1e-3)

    sim.reporters += [
        StateDataReporter(open(os.devnull, "w"), steps_per_block, step=True),
        DCDReporter(f"{out_prefix}.dcd", steps_per_block),
        HDF5Reporter(f"{out_prefix}.h5", steps_per_block)
    ]

    spins = charges.astype(int).copy()
    rng   = np.random.default_rng(seed)

    end2end = []
    for _ in tqdm(range(n_blocks), unit="blk"):
        sim.step(steps_per_block)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        end2end.append(float(np.linalg.norm(pos[-1]-pos[0])))

        if dynamic:
            for _ in range(N_BEADS):
                i = rng.integers(0, N_BEADS)
                dE = 2*spins[i]*(spins[(i-1)%N_BEADS]+spins[(i+1)%N_BEADS])
                if dE<=0 or rng.random()<np.exp(-dE): spins[i]*=-1
            for i, s in enumerate(spins): coulomb.setParticleParameters(i, [float(s)])
            coulomb.updateParametersInContext(sim.context)

    np.savez(f"{out_prefix}.npz", end2end=np.asarray(end2end))
    return {"end2end": np.asarray(end2end), "trajectory": f"{out_prefix}.dcd"}

def run_simulation_with_spin_sequence(spin_seq, *, kc=1.0,
                                      steps_per_frame=1000, out_prefix="ising"):
    spin_seq = np.asarray(spin_seq, float)
    if spin_seq.shape[1] != N_BEADS: raise ValueError
    system, integrator, pdb, coulomb = _build_system(spin_seq[0], kc)
    sim = Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(tolerance=1e-3)

    sim.reporters += [
        DCDReporter(f"{out_prefix}.dcd", steps_per_frame),
        HDF5Reporter(f"{out_prefix}.h5", steps_per_frame)
    ]

    end2end = []
    for frame, vec in enumerate(spin_seq):
        if frame:
            for i, q in enumerate(vec): coulomb.setParticleParameters(i, [float(q)])
            coulomb.updateParametersInContext(sim.context)
        sim.step(steps_per_frame)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        end2end.append(float(np.linalg.norm(pos[-1]-pos[0])))

    np.savez(f"{out_prefix}.npz", end2end=np.asarray(end2end))
    return {"end2end": np.asarray(end2end), "trajectory": f"{out_prefix}.dcd"}

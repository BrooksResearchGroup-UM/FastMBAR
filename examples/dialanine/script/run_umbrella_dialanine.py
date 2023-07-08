import numpy as np
import mdtraj as md
import simtk.openmm.app as app
import simtk.openmm
import simtk.unit as unit
from tqdm import tqdm
import math
import sys
import os

# generating data with umbrella sampling
# some code adapted from: https://fastmbar.readthedocs.io/en/latest/dialanine_PMF.html
psf = app.CharmmPsfFile('examples/dialanine/data/dialanine.psf')
pdb = app.PDBFile('examples/dialanine/data/dialanine.pdb')
charmm_toppar = app.CharmmParameterSet('examples/dialanine/data/top_all36_prot.rtf', 'examples/dialanine/data/par_all36_prot.prm')
system = psf.createSystem(charmm_toppar, nonbondedMethod=app.NoCutoff)
bias_torsion_psi = simtk.openmm.CustomTorsionForce("0.5*k_psi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - psi)")
bias_torsion_psi.addGlobalParameter("pi", math.pi)
bias_torsion_psi.addGlobalParameter("k_psi", 1.0)
bias_torsion_psi.addGlobalParameter("psi", 0.0)
bias_torsion_psi.addTorsion(4, 6, 8, 14) # psi

bias_torsion_phi = simtk.openmm.CustomTorsionForce("0.5*k_phi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - phi)")
bias_torsion_phi.addGlobalParameter("pi", math.pi)
bias_torsion_phi.addGlobalParameter("k_phi", 1.0)
bias_torsion_phi.addGlobalParameter("phi", 0.0)
bias_torsion_phi.addTorsion(6, 8, 14, 16) # phi

system.addForce(bias_torsion_psi)
system.addForce(bias_torsion_phi)

platform = simtk.openmm.Platform.getPlatformByName('CPU')
T = 298.15*unit.kelvin  ## temperature
fricCoef = 10/unit.picoseconds ## friction coefficient
stepsize = 1*unit.femtoseconds ## integration step size
integrator = simtk.openmm.LangevinIntegrator(T, fricCoef, stepsize)
context = simtk.openmm.Context(system, integrator, platform)
k_psi = 100
k_phi = 100
context.setParameter("k_psi", k_psi)
context.setParameter("k_phi", k_phi)
m = 25
M = m*m
psi_bins = np.linspace(-math.pi, math.pi, m, endpoint=False)
phi_bins = np.linspace(-math.pi, math.pi, m, endpoint=False)

if not os.path.exists('test-dialanine/simulation-data'):
    os.makedirs('test-dialanine/simulation-data')

for i in range(M):
    psi_i = psi_bins[i//m]
    phi_i = phi_bins[i%m]
    print(f'psi and phi umbrella centers are {psi_i}, {phi_i}')
    sys.stdout.flush()
    context.setParameter("psi", psi_i)
    context.setParameter("phi", phi_i)
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    for j in range(50):
        simtk.openmm.LocalEnergyMinimizer_minimize(context, 1, 20)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
    integrator.step(5000) # relax the simulation system
    file_handle = open(f'test-dialanine/simulation-data/traj_psi_{i//m}_phi_{i%m}.dcd', 'bw')
    dcd_file = app.dcdfile.DCDFile(file_handle, psf.topology, dt=stepsize)
    for j in range(100):
        integrator.step(100)
        state = context.getState(getPositions=True)
        positions = state.getPositions()
        dcd_file.writeModel(positions)
    file_handle.close()



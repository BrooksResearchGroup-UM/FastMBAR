"""
We run umbrella sampling for the butane dihedral (atom indices: 3-6-9-13).
The dihedral is split into multiple windows and in each window, the dihedral 
is restrainted around a center using a harmonic biasing potential. In this
script, we run simulations in each window sequentially, but they can be run in
parallel you have a computer cluster with multiple nodes.
"""

import openmm.app  as omm_app
import openmm as omm
import openmm.unit as unit
import math
import os
import numpy as np
from tqdm import tqdm
from sys import exit

## read the OpenMM system of butane
with open("./output/system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

## read psf and pdb file of butane
psf = omm_app.CharmmPsfFile("./data/butane.psf")
pdb = omm_app.PDBFile('./data/butane.pdb')
    
## setup an OpenMM context
platform = omm.Platform.getPlatformByName('Reference')
T = 298.15 * unit.kelvin
fricCoef = 10/unit.picoseconds
stepsize = 1 * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)
context = omm.Context(system, integrator, platform)

## set equilibrium theta0 for the biasing potential
K = 50
context.setParameter("K", K)

## M centers of dihedral windows are used in umbrella sampling
M = 30
theta0 = np.linspace(-math.pi, math.pi, M, endpoint = False)
np.savetxt("./output/theta0.csv", theta0, delimiter = ",")

## the main loop to run umbrella sampling window by window
for theta0_index in range(M):
    print(f"sampling at theta0 index: {theta0_index} out of {M}")

    ## set the center of the biasing potential
    context.setParameter("theta0", theta0[theta0_index])

    ## minimize 
    context.setPositions(pdb.positions)    
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()
    for i in range(50):
        omm.LocalEnergyMinimizer_minimize(context, 1, 20)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()

    ## initial equilibrium
    integrator.step(5000)

    ## sampling production. trajectories are saved in dcd files
    file_handle = open(f"./output/traj/traj_{theta0_index}.dcd", 'bw')
    dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)
    for i in tqdm(range(1000)):
        integrator.step(100)
        state = context.getState(getPositions = True)
        positions = state.getPositions()
        dcd_file.writeModel(positions)    
    file_handle.close()

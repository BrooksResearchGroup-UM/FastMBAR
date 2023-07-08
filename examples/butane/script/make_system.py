''' 
Because we are using OpenMM as our MD engine, we need to setup the 
MD molecular system in the format required by OpenMM. The format/object 
used by OpenMM for a molecular system happens to be a class called System. 
Therefore, we will prepare our MD molecular system as an OpenMM System.
When we prepare the OpenMM system, we will add a CustomTorsionForce so
that we can add biasing potentials to the system in the following umbrella
sampling.
'''

import openmm.app  as omm_app
import openmm as omm
import openmm.unit as unit
import math
import os
import numpy as np
from sys import exit

## read psf and pdb file of butane
psf = omm_app.CharmmPsfFile('./data/butane.psf')
pdb = omm_app.PDBFile('./data/butane.pdb')

## read CHARMM force field for butane
params = omm_app.CharmmParameterSet('./data/top_all35_ethers.rtf',
                                    './data/par_all35_ethers.prm')

## create a OpenMM system based on psf of butane and CHARMM force field
system = psf.createSystem(params, nonbondedMethod=omm_app.NoCutoff)

## add a harmonic biasing potential on butane dihedral to the OpenMM system
bias_torsion = omm.CustomTorsionForce("0.5*K*dtheta^2; dtheta = min(diff, 2*Pi-diff); diff = abs(theta - theta0)")
bias_torsion.addGlobalParameter("Pi", math.pi)
bias_torsion.addGlobalParameter("K", 1.0)
bias_torsion.addGlobalParameter("theta0", 0.0)
## 3, 6, 9, 13 are indices of the four carton atoms in butane, between which
## the dihedral angle is biased.
bias_torsion.addTorsion(3, 6, 9, 13)  
system.addForce(bias_torsion)

## save the OpenMM system of butane
with open("./output/system.xml", 'w') as file_handle:
    file_handle.write(omm.XmlSerializer.serialize(system))

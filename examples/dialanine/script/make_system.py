''' 
Because we are using OpenMM as our MD engine, we need to setup the 
MD molecular system in the format required by OpenMM. The format/object 
used by OpenMM for a molecular system happens to be a class called System. 
Therefore, we will prepare our MD molecular system as an OpenMM System.
When we prepare the OpenMM system, we will add two CustomTorsionForces so
that we can add biasing potentials to the system in the following umbrella
sampling.
'''

import simtk.openmm.app  as omm_app
import simtk.openmm as omm
import simtk.unit as unit
import math

## read CHARMM force field for proteins
charmm_toppar = omm_app.CharmmParameterSet("./data/top_all36_prot.rtf",
                                           "./data/par_all36_prot.prm")

## read psf and pdb file of dialanine
psf = omm_app.CharmmPsfFile("./output/dialanine.psf")
pdb = omm_app.PDBFile('./data/dialanine.pdb')

## create a OpenMM system based on psf of dialanine and CHARMM force field
system = psf.createSystem(charmm_toppar, nonbondedMethod = omm_app.NoCutoff)

## add harmonic biasing potentials on two dihedrals of dialanine (psi, phi) in the OpenMM system
## for dihedral psi
bias_torsion_1 = omm.CustomTorsionForce("0.5*k_psi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - psi)")
bias_torsion_1.addGlobalParameter("pi", math.pi)
bias_torsion_1.addGlobalParameter("k_psi", 1.0)
bias_torsion_1.addGlobalParameter("psi", 0.0)
## 4, 6, 8, 14 are indices of the atoms of the torsion psi
bias_torsion_1.addTorsion(4, 6, 8, 14)

## for dihedral phi
bias_torsion_2 = omm.CustomTorsionForce("0.5*k_phi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - phi)")
bias_torsion_2.addGlobalParameter("pi", math.pi)
bias_torsion_2.addGlobalParameter("k_phi", 1.0)
bias_torsion_2.addGlobalParameter("phi", 0.0)
## 6, 8, 14, 16 are indices of the atoms of the torsion phi
bias_torsion_2.addTorsion(6, 8, 14, 16)

system.addForce(bias_torsion_1)
system.addForce(bias_torsion_2)

## save the OpenMM system of dialanine
xml = omm.XmlSerializer.serialize(system)
f = open("./output/system.xml", 'w')
f.write(xml)
f.close()

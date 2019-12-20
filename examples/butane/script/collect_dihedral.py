"""
Based on trajectories from umbrella sampling, we compute the butane dihedral and
save them in .csv files. 
"""

import numpy as np
import mdtraj
import math
from sys import exit

topology = mdtraj.load_psf("./data/butane.psf")
M = 20
for theta0_index in range(M):
    print(theta0_index)
    traj = mdtraj.load_dcd(f"./output/traj/traj_{theta0_index}.dcd", topology)
    theta = mdtraj.compute_dihedrals(traj, [[3, 6, 9, 13]])
    np.savetxt(f"./output/dihedral/dihedral_{theta0_index}.csv", theta, fmt = "%.5f", delimiter = ",")    

__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/10/05 02:25:36"

import numpy as np
import matplotlib.pyplot as plt
import mdtraj
import math
import simtk.unit as unit
import sys
#sys.path.insert(0, "/home/xqding/course/projectsOnGitHub/FastMBAR/FastMBAR/")
from FastMBAR import *
from sys import exit
import pickle

topology = mdtraj.load_psf("./output/dialanine.psf")
K = 100

M = 25
psi = np.linspace(-math.pi, math.pi, M, endpoint = False)
phi = np.linspace(-math.pi, math.pi, M, endpoint = False)

psis = []
phis = []
for psi_index in range(M):
    for phi_index in range(M):
        traj = mdtraj.load_dcd(f"./output/traj/traj_psi_{psi_index}_phi_{phi_index}.dcd", topology)
        psis.append(mdtraj.compute_dihedrals(traj, [[4, 6, 8, 14]]))
        phis.append(mdtraj.compute_dihedrals(traj, [[6, 8, 14, 16]]))

K = 100
T = 298.15 * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * 298.15 * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

psi_array = np.squeeze(np.stack(psis))
phi_array = np.squeeze(np.stack(phis))

n = psi_array.shape[1]
energy_matrix = np.zeros((M**2, n*M**2))                         

psi_array = np.reshape(psi_array, (-1,))
phi_array = np.reshape(phi_array, (-1,))

for index in range(M**2):
    psi_index = index // M
    phi_index = index % M
    
    psi_c = psi[psi_index]
    phi_c = phi[phi_index]

    psi_diff = np.abs(psi_array - psi_c)
    psi_diff = np.minimum(psi_diff, 2*math.pi-psi_diff)

    phi_diff = np.abs(phi_array - phi_c)
    phi_diff = np.minimum(phi_diff, 2*math.pi-phi_diff)
    
    energy_matrix[index, :] = 0.5*K*(psi_diff**2 + phi_diff**2)/kbT
    
M_PMF = 25
psi_PMF = np.linspace(-math.pi, math.pi, M_PMF, endpoint = False)
phi_PMF = np.linspace(-math.pi, math.pi, M_PMF, endpoint = False)
width = 2*math.pi / M_PMF

energy_PMF = np.zeros((M_PMF**2, energy_matrix.shape[1]))

for index in range(M_PMF**2):
    print(index)
    psi_index = index // M_PMF
    phi_index = index % M_PMF
    psi_c_PMF = psi_PMF[psi_index]
    phi_c_PMF = phi_PMF[phi_index]

    psi_low = psi_c_PMF - 0.5*width
    psi_high = psi_c_PMF + 0.5*width

    phi_low = phi_c_PMF - 0.5*width
    phi_high = phi_c_PMF + 0.5*width

    psi_indicator = ((psi_array > psi_low) & (psi_array <= psi_high)) | \
                     ((psi_array + 2*math.pi > psi_low) & (psi_array + 2*math.pi <= psi_high)) | \
                     ((psi_array - 2*math.pi > psi_low) & (psi_array - 2*math.pi <= psi_high))

    phi_indicator = ((phi_array > phi_low) & (phi_array <= phi_high)) | \
                     ((phi_array + 2*math.pi > phi_low) & (phi_array + 2*math.pi <= phi_high)) | \
                     ((phi_array - 2*math.pi > phi_low) & (phi_array - 2*math.pi <= phi_high))
        
    indicator = psi_indicator & phi_indicator
    energy_PMF[index, ~indicator] = np.inf
    
#energy_matrix = np.vstack([energy_matrix, energy_PMF])
num_conf_all = np.array([n for i in range(M**2)])
fastmbar = FastMBAR(energy = energy_matrix, num_conf = num_conf_all, cuda = True, verbose = True)
PMF, _ = fastmbar.calculate_free_energies_of_perturbed_states(energy_PMF)

with open("./output/PMF_fast_mbar.pkl", 'wb') as file_handle:
    pickle.dump(PMF, file_handle)

fig = plt.figure(0)
fig.clf()
plt.imshow(np.flipud(PMF.reshape((M_PMF, M_PMF)).T), extent = (-180, 180, -180, 180))
plt.colorbar()
plt.savefig("./output/PMF_fast_mbar.pdf")

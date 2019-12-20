__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/10/05 02:25:36"

import numpy as np
import matplotlib.pyplot as plt
import math
import simtk.unit as unit
from FastMBAR import *
from sys import exit
import pickle

M = 20
thetas = []
num_conf = []
for theta0_index in range(M):
    theta = np.loadtxt(f"./output/dihedral/dihedral_{theta0_index}.csv", delimiter = ",")
    thetas.append(theta)
    num_conf.append(len(theta))
thetas = np.concatenate(thetas)
num_conf = np.array(num_conf).astype(np.float64)
N = len(thetas)

reduced_energy_matrix = np.zeros((M, N))
K = 100
T = 298.15 * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * 298.15 * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

theta0 = np.loadtxt("./output/theta0.csv", delimiter = ",")

for theta0_index in range(M):
    current_theta0 = theta0[theta0_index]
    diff = np.abs(thetas - current_theta0)
    diff = np.minimum(diff, 2*math.pi-diff)
    reduced_energy_matrix[theta0_index, :] = 0.5*K*diff**2/kbT

fastmbar = FastMBAR(energy = reduced_energy_matrix, num_conf = num_conf, cuda=False, verbose = True)
print(fastmbar.F)

M_PMF = 25
theta_PMF = np.linspace(-math.pi, math.pi, M_PMF, endpoint = False)
width = 2*math.pi / M_PMF
reduced_energy_PMF = np.zeros((M_PMF, N))

for i in range(M_PMF):
    print(i)
    theta_center = theta_PMF[i]
    theta_low = theta_center - 0.5*width
    theta_high = theta_center + 0.5*width

    indicator = ((thetas > theta_low) & (thetas <= theta_high)) | \
                 ((thetas + 2*math.pi > theta_low) & (thetas + 2*math.pi <= theta_high)) | \
                 ((thetas - 2*math.pi > theta_low) & (thetas - 2*math.pi <= theta_high))
    
    reduced_energy_PMF[i, ~indicator] = np.inf
    
PMF, _ = fastmbar.calculate_free_energies_of_perturbed_states(reduced_energy_PMF)
    
fig = plt.figure(0)
fig.clf()
plt.plot(theta_PMF, PMF, '-o')
plt.savefig("./output/PMF_fast_mbar.pdf")

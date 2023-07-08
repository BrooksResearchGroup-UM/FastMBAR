import fastmbartools
import numpy as np
import mdtraj as md
import simtk.openmm.app as app
import simtk.openmm
import simtk.unit as unit
from tqdm import tqdm
import math
import sys
import os
import seaborn as sns
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# draw the plot
if not os.path.exists('pictures'):
    os.makedirs('pictures')

# compute PMF with fastmbartools

# set parameters
# in openmm, the default unit for energy is kj/mol
T = 298.15*unit.kelvin
temp = T.value_in_unit(unit.kelvin)
kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
kB = kB.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
k_psi = 100
k_phi = 100
m = 25
M = m*m
psi_bins = np.linspace(-math.pi, math.pi, m, endpoint=False)
phi_bins = np.linspace(-math.pi, math.pi, m, endpoint=False)
topology = md.load_psf('examples/dialanine/data/dialanine.psf')
traj_bias_cv = []
bias_param = []

# collect data
for i in range(M):
    psi_i = psi_bins[i//m]
    phi_i = phi_bins[i%m]
    dcd_i = f'test-dialanine/simulation-data/traj_psi_{i//m}_phi_{i%m}.dcd'
    if os.path.exists(dcd_i):
        #print(f'Loading dcd file {dcd_i}')
        traj_i = md.load_dcd(dcd_i, topology)
        dihedrals = md.compute_dihedrals(traj_i, [[4, 6, 8, 14], [6, 8, 14, 16]])
        traj_bias_cv.append(dihedrals)
        bias_param.append(np.array([k_psi, psi_i, k_phi, phi_i]))
        plt.scatter(dihedrals[:, 0], dihedrals[:, 1], s=0.1)
    else:
        pass
        #print(f'{dcd_i} does not exist')
    
plt.xlabel(r'$\psi$')
plt.ylabel(r'$\phi$')
plt.savefig('pictures/dialanine_distribution.pdf')
plt.close()

# define bias function
# because dihedrals are periodic, so we cannot simply use harmoinc bias
def dihedral_umbrella_bias(x, param):
    # param should be np.array([kappa1, x0_1, kappa2, x0_2, ...])
    param = np.reshape(param, (-1, 2))
    kappa = param[:, 0]
    x0 = param[:, 1]
    delta_x = np.absolute(x - x0)
    delta_x = np.minimum(delta_x, 2*np.pi - delta_x)
    bias = 0.5*np.sum(kappa*(delta_x**2), axis=1)
    return bias

# as we keep working on the same temperature for all the simulations and target PMF, we do not need to input unbiased energies
bias_func = dihedral_umbrella_bias
fastmbarsolver1 = fastmbartools.FastMBARSolver(temp, traj_bias_cv, kB, bias_param, bias_func)
fastmbarsolver1.solve()
traj_target_cv = traj_bias_cv
target_temp = temp
theta_bins = np.linspace(-math.pi, math.pi, m + 1)
fastmbarsolver1.computePMF(traj_target_cv, target_temp, [theta_bins, theta_bins])
fastmbarsolver1.writePMF('test-dialanine/pmf1.txt')
x = fastmbarsolver1.output[:, :2]
f = fastmbarsolver1.output[:, 2]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
my_cmap = ListedColormap(sns.color_palette("coolwarm", 20).as_hex())
grid_psi, grid_phi = np.meshgrid(theta_bins, theta_bins)
grid_f = griddata(x, f, (grid_psi, grid_phi), method='linear')
levels = np.arange(0, 15, 2)
ax1.contour(grid_psi, grid_phi, grid_f, colors='black', levels=levels)
heatmap = ax1.contourf(grid_psi, grid_phi, grid_f, vmin=None, vmax=None, levels=levels, cmap=my_cmap)
ax1.set(xlabel=r'$\psi$', ylabel=r'$\phi$')
cbar = fig1.colorbar(heatmap)
fig1.savefig('pictures/pmf1.pdf')
plt.close(fig1)

# draw the with similar format as the one shown in manual example website
fig = plt.figure(0)
fig.clf()
plt.imshow(np.flipud(fastmbarsolver1.PMF.reshape((m, m)).T), extent = (-180, 180, -180, 180))
plt.xlabel(r"$\psi$")
plt.ylabel(r"$\phi$")
plt.colorbar()
plt.savefig("pictures/PMF_fastmbar.pdf")
plt.close(fig)

# next, let's test if method add_traj can work properly
# we first initialize with a fraction of the data, then use add_traj to load the rest 
traj_bias_cv_part1 = traj_bias_cv[:40]
traj_bias_cv_part2 = traj_bias_cv[40:]
bias_param_part1 = bias_param[:40]
bias_param_part2 = bias_param[40:]
fastmbarsolver2 = fastmbartools.FastMBARSolver(temp, traj_bias_cv_part1, kB, bias_param_part1, bias_func)
fastmbarsolver2.add_traj(temp, traj_bias_cv_part2, bias_param_part2, bias_func)
fastmbarsolver2.solve()
fastmbarsolver2.computePMF(traj_target_cv, target_temp, [theta_bins, theta_bins])
fastmbarsolver2.writePMF('test-dialanine/pmf2.txt')
x = fastmbarsolver2.output[:, :2]
f = fastmbarsolver2.output[:, 2]

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
my_cmap = ListedColormap(sns.color_palette("coolwarm", 20).as_hex())
grid_psi, grid_phi = np.meshgrid(theta_bins, theta_bins)
grid_f = griddata(x, f, (grid_psi, grid_phi), method='linear')
levels = np.arange(0, 15, 2)
ax2.contour(grid_psi, grid_phi, grid_f, colors='black', levels=levels)
heatmap = ax2.contourf(grid_psi, grid_phi, grid_f, vmin=None, vmax=None, levels=levels, cmap=my_cmap)
ax2.set(xlabel=r'$\psi$', ylabel=r'$\phi$')
cbar = fig2.colorbar(heatmap)
fig2.savefig('pictures/pmf2.pdf')
plt.close(fig2)



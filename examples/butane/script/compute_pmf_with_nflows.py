__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/07/11 22:54:45"

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/nflows")
from nflows import transforms, distributions
import simtk.unit as unit
from sys import exit
import math
import torch.optim as optim

M = 30
thetas = []
num_conf = []
for theta0_index in range(M):
    theta = np.loadtxt(f"./output/dihedral/dihedral_{theta0_index}.csv", delimiter = ",")
    thetas.append(theta)
    num_conf.append(len(theta))

theta0 = np.loadtxt("./output/theta0.csv", delimiter = ",")

K = 50
T = 298.15 * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * 298.15 * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

base_distribution = distributions.BoxUniform(low = torch.tensor([-math.pi]),
                                             high = torch.tensor([math.pi]))

num_bins = 4
unnormalized_widths = torch.randn(num_bins, requires_grad = True)
unnormalized_heights = torch.randn(num_bins, requires_grad = True)
unnormalized_internal_derivatives = torch.randn(num_bins-1, requires_grad = True)
unnormalized_edge_derivatives = torch.randn(1, requires_grad = True)
unnormalized_derivatives = torch.cat([unnormalized_edge_derivatives,
                                      unnormalized_internal_derivatives,
                                      unnormalized_edge_derivatives])

optimizer = optim.Adam([unnormalized_widths,
                        unnormalized_heights,
                        unnormalized_internal_derivatives,
                        unnormalized_edge_derivatives],
                       lr = 0.001)

for idx_step in range(50):
    log_prob_list = []
    log_Q_list = []
    for idx in range(M):
        theta = torch.from_numpy(thetas[idx])
        batch_size = theta.shape[0]

        z, logabsdet = transforms.splines.rational_quadratic_spline(
            theta,
            unnormalized_widths.repeat(batch_size, 1),
            unnormalized_heights.repeat(batch_size, 1),
            unnormalized_derivatives.repeat(batch_size, 1),
            inverse = False,
            left = -math.pi, right = math.pi,
            bottom = -math.pi, top = math.pi)
        z = z[:, None]
        log_prob = base_distribution.log_prob(z) + logabsdet 

        sample_z = base_distribution.sample((batch_size,))[:,0]
        N = sample_z.shape[0]
        sample_theta, logabsdet = transforms.splines.rational_quadratic_spline(
            sample_z,
            unnormalized_widths.repeat(N, 1),
            unnormalized_heights.repeat(N, 1),
            unnormalized_derivatives.repeat(N, 1),
            inverse = True,
            left = -math.pi, right = math.pi,
            bottom = -math.pi, top = math.pi)

        diff = torch.abs(sample_theta - theta0[idx])
        diff_candidate = 2*math.pi - diff

        flag = diff_candidate < diff
        diff[flag] = diff_candidate[flag]
        
        Q = torch.mean(torch.exp(-0.5*K*diff**2/kbT))
        log_prob = log_prob - torch.log(Q)
        
        log_prob_list.append(log_prob)
        log_Q_list.append(torch.log(Q))

    log_prob = torch.cat(log_prob_list)
    loss = -torch.mean(log_prob)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("idx_step: {}, loss: {:.2f}".format(idx_step, loss.item()))
    
theta = torch.linspace(-math.pi, math.pi)
N = theta.shape[0]
with torch.no_grad():
    z, logabsdet = transforms.splines.rational_quadratic_spline(
        theta,
        unnormalized_widths.repeat(N, 1),
        unnormalized_heights.repeat(N, 1),
        unnormalized_derivatives.repeat(N, 1),
        inverse = False,
        left = -math.pi, right = math.pi,
        bottom = -math.pi, top = math.pi)
z = z[:, None]
log_prob = base_distribution.log_prob(z) + logabsdet 
F = -log_prob

fig = plt.figure(0)
fig.clf()
plt.plot(theta.numpy(), F, "o-")
plt.savefig("./output/PMF_nflows.pdf")


    




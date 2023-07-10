import pytest
from pytest import approx
import torch
import torch.distributions as dist
import numpy as np
from sys import exit
import math
import os, sys
from FastMBAR import FastMBAR

@pytest.fixture
def setup_data():
    num_states = 20
    num_conf = torch.randint(500, 1000, (num_states,))

    mu = dist.Normal(0.0, 2.0).sample((num_states,))
    sigma = dist.Uniform(1, 3).sample((num_states,))  ## sqrt of 1/k

    ## draw samples from each state and
    ## calculate energies of each sample in all states
    Xs = []
    for i in range(num_states):
        Xs.append(dist.Normal(mu[i], sigma[i]).sample((num_conf[i],)))
    Xs = torch.cat(Xs)
    energy = torch.zeros((num_states, len(Xs)))
    for i in range(num_states):
        energy[i, :] = 0.5 * ((Xs - mu[i]) / sigma[i]) ** 2
    
    F_ref = -torch.log(sigma)
    pi = num_conf / num_conf.sum()
    F_ref = F_ref - torch.sum(pi * F_ref)

    return energy, num_conf, F_ref

@pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_FastMBAR_cpus(setup_data, method, bootstrap):
    energy, num_conf, F_ref = setup_data
    fastmbar = FastMBAR(
        energy,
        num_conf,
        cuda=False,
        cuda_batch_mode=False,
        bootstrap=bootstrap,
        verbose=False,
        method=method,
    )
    # print(fastmbar.F)
    # print(F_ref)
    assert fastmbar.F == approx(F_ref, abs = 1e-1)

@pytest.mark.skipif(torch.cuda.is_available() is False, reason="CUDA is not avaible")
@pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("cuda_batch_mode", [False, True])
def test_FastMBAR_gpus(setup_data, method, bootstrap, cuda_batch_mode):
    energy, num_conf, F_ref = setup_data
    fastmbar = FastMBAR(
        energy,
        num_conf,
        cuda=True,
        cuda_batch_mode=cuda_batch_mode,
        bootstrap=bootstrap,
        verbose=False,
        method=method,
    )
    assert fastmbar.F == approx(F_ref, abs = 1e-1)

# energy, num_conf, F_ref = setup_data()
# fastmbar = FastMBAR(
#     energy,
#     num_conf,
#     cuda=False,
#     cuda_batch_mode=False,
#     bootstrap=True,
#     verbose=True,
#     method="Newton",
# )

# res = fastmbar.calculate_free_energies_of_perturbed_states(energy)

# mbar = pymbar.MBAR(energy.numpy(), num_conf.numpy())
# mbar_results = mbar.compute_free_energy_differences(return_theta=True)
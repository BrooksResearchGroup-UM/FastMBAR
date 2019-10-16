import sys
sys.path.insert(0, "/home/xqding/course/projectsOnGitHub/FastMBAR/FastMBAR/")
from FastMBAR import *
import torch
import numpy as np
from sys import exit

def test_FastMBAR():
    import numpy as np

    ## draw samples from multiple states, each of which is a harmonic
    ## osicillator
    #np.random.seed(0)
    num_states = 20 ## num of states with nonzero num of samples
    num_conf = np.array([5000 for i in range(num_states)])
    mu = np.random.normal(0, 4, size = num_states) ## equalibrium positions
    sigma = np.random.uniform(1, 3, size = num_states) ## sqrt of 1/k

    ## draw samples from each state and calculate energies of each sample
    ## in all states
    Xs = []
    for i in range(num_states):
        Xs += list(np.random.normal(loc = mu[i],
                                    scale = sigma[i],
                                    size = num_conf[i]))
    Xs = np.array(Xs)
    energy = np.zeros((num_states, len(Xs)))
    for i in range(num_states):
        energy[i,:] = 0.5 * ((Xs - mu[i])/sigma[i])**2

    num_states_perturbed = 5
    mu_perturbed = np.random.normal(0, 4, size = num_states_perturbed) ## equalibrium positions
    sigma_perturbed = np.random.uniform(1, 3, size = num_states_perturbed) ## sqrt of 1/k
    energy_perturbed = np.zeros((num_states_perturbed, len(Xs)))
    for i in range(num_states_perturbed):
        energy_perturbed[i, :] = 0.5 * ((Xs - mu_perturbed[i])/sigma_perturbed[i])**2

    ## reference results
    reference_result = np.concatenate([-np.log(sigma), -np.log(sigma_perturbed)])
    reference_result = reference_result - reference_result[0]


    print("Without bootstrap")
    print("="*40)        
    ## calcuate free energies using CPUs
    mbar = FastMBAR(energy, num_conf, cuda = False, bootstrap = False)
    F_perturbed, _ = mbar.calculate_free_energies_of_perturbed_states(energy_perturbed)
    F = np.concatenate((mbar.F, F_perturbed))
    diff_cpu = np.sqrt(np.mean((F-reference_result)**2))
    print("RMSD (CPU calculation and reference results): {:.2f}".format(diff_cpu))
    assert(diff_cpu <= 0.05)

    if torch.cuda.is_available():
        ## calcuate free energies using GPUs
        mbar = FastMBAR(energy, num_conf, cuda = True, bootstrap = False)
        F_perturbed, _ = mbar.calculate_free_energies_of_perturbed_states(energy_perturbed)
        F = np.concatenate((mbar.F, F_perturbed))
        diff_cpu = np.sqrt(np.mean((F-reference_result)**2))
        print("RMSD (GPU calculation and reference results): {:.2f}".format(diff_cpu))
        assert(diff_cpu <= 0.05)    

        mbar = FastMBAR(energy, num_conf, cuda = True, cuda_batch_mode = True,  bootstrap = False)
        F_perturbed, _ = mbar.calculate_free_energies_of_perturbed_states(energy_perturbed)
        F = np.concatenate((mbar.F, F_perturbed))
        diff_cpu = np.sqrt(np.mean((F-reference_result)**2))
        print("RMSD (GPU-batch-mode calculation and reference results): {:.2f}".format(diff_cpu))
        assert(diff_cpu <= 0.05)    

    print("With bootstrap")
    print("="*40)        
    ## calcuate free energies using CPUs
    mbar = FastMBAR(energy, num_conf, cuda = False, bootstrap = True)
    F_perturbed, _ = mbar.calculate_free_energies_of_perturbed_states(energy_perturbed)
    F = np.concatenate((mbar.F, F_perturbed))
    diff_cpu = np.sqrt(np.mean((F-reference_result)**2))
    print("RMSD (CPU calculation and reference results): {:.2f}".format(diff_cpu))
    assert(diff_cpu <= 0.05)

    if torch.cuda.is_available():    
        ## calcuate free energies using GPUs
        mbar = FastMBAR(energy, num_conf, cuda = True, bootstrap = True)
        F_perturbed, _ = mbar.calculate_free_energies_of_perturbed_states(energy_perturbed)
        F = np.concatenate((mbar.F, F_perturbed))
        diff_cpu = np.sqrt(np.mean((F-reference_result)**2))
        print("RMSD (GPU calculation and reference results): {:.2f}".format(diff_cpu))
        assert(diff_cpu <= 0.05)    

        mbar = FastMBAR(energy, num_conf, cuda = True, cuda_batch_mode = True,  bootstrap = True)
        F_perturbed, _ = mbar.calculate_free_energies_of_perturbed_states(energy_perturbed)
        F = np.concatenate((mbar.F, F_perturbed))
        diff_cpu = np.sqrt(np.mean((F-reference_result)**2))
        print("RMSD (GPU-batch-mode calculation and reference results): {:.2f}".format(diff_cpu))
        assert(diff_cpu <= 0.05)
        
if __name__ == "__main__":
    test_FastMBAR()

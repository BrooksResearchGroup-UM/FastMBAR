import sys
sys.path.insert(0, "/home/xqding/course/projectsOnGitHub/FastMBAR/FastMBAR/")
from FastMBAR import *
import torch
import numpy as np

def test_FastMBAR():

    import numpy as np

    ## draw samples from multiple states, each of which is a harmonic
    ## osicillator
    np.random.seed(0)
    num_states_nz = 20 ## num of states with nonzero num of samples
    num_states_z = 5 ## num of states with zero samples
    num_states = num_states_nz + num_states_z
    num_conf = np.array([5000 for i in range(num_states_nz)] +
                        [0 for i in range(num_states_z)])
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

    ## reference results
    true_result = -np.log(sigma)
    true_result = true_result - true_result[0]

    energy = energy.astype(np.float64)

    # ## reference results
    # ref_result = np.array([ 0.0, -0.381, 0.548, 0.665,
    #                         0.743, -0.388, -0.242, -0.377,
    #                         -0.449, -0.286, 0.003, -0.209])

    ref_result = true_result

    ## calcuate free energies using CPUs
    mbar = FastMBAR(energy, num_conf)
    F_cpu, _ = mbar.calculate_free_energies()
    diff_cpu = np.sqrt(np.mean((F_cpu-ref_result)**2))
    print("RMSD (CPU calculation and reference results): {:.2f}".format(diff_cpu))
    assert(diff_cpu <= 0.05)

    ## calculate free energies using GPUs
    if torch.cuda.is_available():
        mbar = FastMBAR(energy, num_conf, cuda = True, cuda_batch_mode = False)
        F_gpu, _ = mbar.calculate_free_energies()    
        diff_gpu = np.sqrt(np.mean((F_gpu-ref_result)**2))        
        print("RMSD (GPU calculation and reference results): {:.2f}".format(diff_gpu))        
        assert(diff_gpu <= 0.05)
        
        mbar = FastMBAR(energy, num_conf, cuda = True, cuda_batch_mode = True)
        F_gpu, _ = mbar.calculate_free_energies()    
        diff_gpu = np.sqrt(np.mean((F_gpu-ref_result)**2))        
        print("RMSD (GPU (batch mode) calculation and reference results): {:.2f}".format(diff_gpu))        
        assert(diff_gpu <= 0.05)
        
if __name__ == "__main__":
    test_FastMBAR()

__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/05/29 20:38:05"

import numpy as np
import torch
import torch.nn as nn
import scipy.optimize as optimize

class FastMBAR():
    
    """
    The FastMBAR class is initialized with an energy matrix and an array
    of num of conformations. After initizlization, the FastMBAR class method
    calculate_free_energies is used to calculate the relative free energyies 
    for all states.

    """
    
    def __init__(self, energy, num_conf, cuda = False):
        """Initizlizer for class FastMBAR

        Parameters
        ----------
        energy : 2-D float ndarray with size of (M x N)
            A 2-D ndarray with size of M x N, where M is the number of stats
            and N is the total number of conformations. The entry energy[i,j]
            is the reduced (unitless) energy of conformation j in state i.
            If bootstrapping is used to calculate the uncertainty, the order
            of conformations matters. Conformations sampled from one state
            need to occpy a continuous chunk of collumns. Conformations sampled
            from state k need to occupy collumns to the left of conformations 
            sampled from state l if k < l. If bootstrapping is not used, then 
            the order of conformation does not matter.
        num_conf: 1-D int ndarray with size (M)
            A 1-D ndarray with size of M, where num_conf[i] is the num of 
            conformations sampled from state i. Therefore, np.sum(num_conf)
            has to be equal to N.
        cuda: bool, optional
            If it is set to be True, then the calculation in FastMBAR.calculate_free_energies 
            will be run on graphical processing units (GPUs) using CUDA.
        """
        self.cuda = cuda
        
        self.energy = energy
        self.num_conf = num_conf.astype(energy.dtype)
        
        self.num_states = energy.shape[0]
        self.tot_num_conf = energy.shape[1]
        
        # assert(np.sum(self.num_conf) == self.tot_num_conf)
        # assert(self.num_states == len(self.num_conf))

        self.flag_zero = num_conf == 0
        self.flag_nz = num_conf != 0

        self.energy_zero = torch.from_numpy(self.energy[self.flag_zero, :])
        self.energy_nz = torch.from_numpy(self.energy[self.flag_nz, :])
        self.num_conf_nz = torch.from_numpy(self.num_conf[self.flag_nz])
        self.num_states_nz = self.energy_nz.shape[0]

        if self.cuda:
            self.energy_zero = self.energy_zero.cuda()
            self.energy_nz = self.energy_nz.cuda()
            self.num_conf_nz = self.num_conf_nz.cuda()
        
        self.bias_energy_nz = None

        
    def _loss_nz(self, bias_energy_nz):
        assert(self.num_states_nz == len(bias_energy_nz))
        bias_energy_nz = torch.tensor(bias_energy_nz,
                                      requires_grad = True,
                                      dtype = self.energy_nz.dtype)
        if self.cuda:
            self.bias_energy_nz = bias_energy_nz.cuda()
        else:
            self.bias_energy_nz = bias_energy_nz
            
        energy_nz = self.energy_nz - torch.min(self.energy_nz, 0)[0]    
        tmp = torch.exp(-(energy_nz +
                          self.bias_energy_nz.view([self.num_states_nz, 1])))        
        tmp = torch.sum(tmp, 0)
        loss = torch.sum(torch.log(tmp)) + torch.sum(self.num_conf_nz*self.bias_energy_nz)
        loss = loss / self.energy_nz.shape[1]

        loss.backward()
        return (loss.cpu().detach().numpy().astype(np.float64),
                bias_energy_nz.cpu().grad.numpy().astype(np.float64))

    
    def _solve_mbar_equation(self, initial_F = None, verbose = False):
        """ calculate the relative free energies for all states by
        solving the MBAR equation once with the initial guess of initial_F

        Parameters
        ----------
        initial_F: 1-D float ndarray with size of (M,)
            starting point used to solve the MBAR equations
        verbose: bool, optional
            if verbose is true, the detailed information of running
            LBFGS optimization is printed.
        Returns
        -------
        F: 1-D float array with size of (M,)
            the relative unitless free energies for all states

        """

        if initial_F is None:
            x0 = self.energy_nz.new(self.num_states_nz).zero_()
            x0 = x0.cpu().numpy()
        else:
            assert(isinstance(initial_F, np.ndarray))
            assert(initial_F.ndim == 1)
            assert(len(initial_F) == self.num_states)
            initial_F_nz = initial_F[self.flag_nz]
            initial_F_nz = self.energy_nz.new(initial_F_nz)
            sample_prop_nz = self.num_conf_nz / torch.sum(self.num_conf_nz)
            x0 = - torch.log(sample_prop_nz) - initial_F_nz
            x0 = x0.cpu().numpy()
        
        x, f, d = optimize.fmin_l_bfgs_b(self._loss_nz, x0, iprint = verbose)
        self.bias_energy_nz = self.energy_nz.new(x)
        if self.cuda:
            self.bias_energy_nz = self.bias_energy_nz.cuda()
        
        ## get free energies for states with nonzero number of samples
        sample_prop_nz = self.num_conf_nz / torch.sum(self.num_conf_nz)
        self.F_nz = -torch.log(sample_prop_nz) - self.bias_energy_nz

        ## normalize free energies
        prob_nz = torch.exp(-self.F_nz)
        prob_nz = prob_nz / torch.sum(prob_nz)
        self.F_nz = -torch.log(prob_nz)

        ## update bias energies for states with nonzero number of samples
        ## using normalized free energies
        self.bias_energy_nz = -torch.log(sample_prop_nz) - self.F_nz

        ## calculate free energies for states with zero number of samples
        self.F = self.bias_energy_nz.new(self.num_states)
        
        if self.cuda:
            self.F = self.F.cuda()
            
        idx_zero = 0        
        idx_nz = 0
        for i in range(self.num_states):
            if self.flag_nz[i]:
                self.F[i] = self.F_nz[idx_nz]
                idx_nz += 1
            else:
                tmp = self.energy_nz + self.bias_energy_nz.view((-1,1)) - \
                      self.energy_zero[idx_zero, :]
                tmp = -torch.log(torch.mean(1.0/torch.sum(torch.exp(-tmp), 0)))
                self.F[i] = tmp
                idx_zero += 1
                
        self.F = self.F - self.F[0]
        self.bias_energy_nz = -torch.log(sample_prop_nz) - self.F[torch.ByteTensor(self.flag_nz.astype(int))]

        return self.F.cpu().numpy()

    def calculate_free_energies(self, bootstrap = False, block_size = 3, num_rep = 100,
                                verbose = False):
        
        """ calculate the relative free energies for all states
        
        Parameters
        ----------
        bootstrap: bool, optional
            if bootstrap is true, block bootstrapping is used to estimate
            the uncertainty of the calculated relative free energies
        block_size: int, optional
            the size of block used in block bootstrapping
        num_rep: int, optional
            the number of repeats used in bootstrapping to estimate uncertainty
        verbose: bool, optional
            if verbose is true, the detailed information of running
            LBFGS optimization is printed.

        Returns
        -------
        (F, F_std): 
          F: 1-D float array with size of (M,)
              the relative unitless free energies for all states
          F_std: 1-D float array with size of (M,) if bootstrap = True.
                 Otherwise, it is None
              the uncertainty of F calculated using block bootstrapping 
        """
        if bootstrap:
            num_conf_nz = self.num_conf[self.flag_nz]
            num_conf_nz = num_conf_nz.astype(np.int)
            num_conf_nz_cumsum = list(np.cumsum(num_conf_nz))
            num_conf_nz_cumsum.pop(-1)
            num_conf_nz_cumsum = [0] + num_conf_nz_cumsum
            initial_F = np.zeros(self.num_states)
            bootstrap_F = []
            for _k in range(num_rep):
                print(_k)
                conf_idx = []
                
                for i in range(len(num_conf_nz)):
                    len_seq = num_conf_nz[i] ## using circular block bootstrap
                    num_sample_block = int(np.ceil(len_seq/block_size))
                    idxs = np.random.choice(int(len_seq), num_sample_block, replace = True)
                    sample_idx = []
                    for idx in idxs:
                        sample_idx += list(range(idx, idx+block_size))

                    for k in range(len(sample_idx)):
                        sample_idx[k] = np.mod(sample_idx[k], len_seq)
                        
                    if len(sample_idx) > len_seq:
                        pop_idx = np.random.choice(len(sample_idx), 1)[0]
                        for k in range(pop_idx, pop_idx+len(sample_idx)-len_seq):
                            k = np.mod(k, len(sample_idx))
                            sample_idx.pop(k)
                        
                    assert(len(sample_idx) == len_seq)
                    conf_idx += [ idx + num_conf_nz_cumsum[i] for idx in sample_idx]
                    
                assert(len(conf_idx) == self.tot_num_conf)
                sub_mbar = FastMBAR(self.energy[:, conf_idx], self.num_conf, self.cuda)
                sub_F = sub_mbar._solve_mbar_equation(initial_F, verbose)
                initial_F = sub_F
                bootstrap_F.append(sub_F)
                
            bootstrap_F = np.array(bootstrap_F)
            mean_F = bootstrap_F.mean(0)
            std_F = bootstrap_F.std(0)
        else:
            mean_F = self._solve_mbar_equation(verbose = verbose)
            std_F = None
            
        return (mean_F, std_F)

def test():
    import numpy as np

    ## draw samples from multiple states, each of which is a harmonic
    ## osicillator
    np.random.seed(0)
    num_states_nz = 10 ## num of states with nonzero num of samples
    num_states_z = 2 ## num of states with zero samples
    num_states = num_states_nz + num_states_z
    num_conf = np.array([400 for i in range(num_states_nz)] +
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

    ## reference results
    ref_result = np.array([ 0.0, -0.381, 0.548, 0.665,
                            0.743, -0.388, -0.242, -0.377,
                            -0.449, -0.286, 0.003, -0.209])

    ## calcuate free energies using CPUs
    mbar = FastMBAR(energy, num_conf)
    F_cpu = mbar.calculate_free_energies()    
    diff_cpu = np.sqrt(np.mean((F_cpu-ref_result)**2))    
    print("RMSD (CPU calculation and reference results): {:.2f}".format(diff_cpu))

    ## calculate free energies using GPUs
    if torch.cuda.is_available():
        mbar = FastMBAR(energy, num_conf, cuda = True)
        F_gpu = mbar.calculate_free_energies()    
        diff_gpu = np.sqrt(np.mean((F_gpu-ref_result)**2))    
        print("RMSD (GPU calculation and reference results): {:.2f}".format(diff_gpu))        

    

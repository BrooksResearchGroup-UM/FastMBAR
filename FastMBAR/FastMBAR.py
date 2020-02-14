__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/05/29 20:38:05"

import numpy as np
import torch
import torch.nn as nn
import scipy.optimize as optimize

class FastMBAR():
    """
    The FastMBAR class is initialized with an energy matrix and an array
    of num of conformations. The corresponding MBAR equation is solved
    in the constructor. Therefore, the relative free energies of states
    used in the energy matrix is calculated in the constructor. The, the 
    FastMBAR class method calculate_free_energies_for_perturbed_states 
    can be used to calcualted the relative free energies of perturbed states
    """    
    def __init__(self, energy, num_conf,
                 cuda = False, cuda_batch_mode = None,
                 bootstrap = False, bootstrap_block_size = 3,
                 bootstrap_num_rep = 5,
                 verbose = False):
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
            has to be equal to N. All entries in num_conf have to be strictly
            greater than 0.
        cuda: bool, optional
            If it is set to be True, then the calculation will be run on 
            a graphical processing unit (GPU) using CUDA.
        cuda_batch_mode: bool, optional
            The batch mode is turned on when the size of the energy matrix is
            too large for the memory of a GPU. When cuda_batch_mode is True,
            the energy matrix will be split into multiple batches which are used
            sequentially. If cuda_batch_mode = None, it will be set automatically
            based on the size of energy and the memory of the avaible GPU device.
        bootstrap: bool, optional
            If bootstrap is True, the uncertainty of the calculated free energies
            will be estimate using block bootstraping.
        bootstrap_block_size: int, optional
            block size used in block bootstrapping
        bootstrap_num_rep: int, optional
            number of repreats in block bootstrapping

        """
        ## setting for bootstrap
        self.bootstrap = bootstrap
        self.bootstrap_block_size = bootstrap_block_size
        self.bootstrap_num_rep = bootstrap_num_rep

        self.verbose = verbose

        self.cuda = cuda
        self.cuda_batch_mode = cuda_batch_mode
        
        assert np.all(num_conf > 0), \
            '''The number of conformations sampled from each state
               has to be strictly greater than zero. You can fix this 
               problem by removing states from which no conformations
               are sampled.'''
        
        self.energy = energy.astype(np.float64)
        self.num_conf = num_conf.astype(energy.dtype)
        self.num_states = energy.shape[0]
        self.tot_num_conf = energy.shape[1]
        self.num_conf_ratio = self.num_conf / float(self.tot_num_conf)
                        
        ## moving data to GPU if cuda is used
        if self.cuda:
            self.num_conf_cuda = torch.from_numpy(self.num_conf).cuda()
            self.num_conf_ratio_cuda = torch.from_numpy(self.num_conf_ratio).cuda()

            ## check if the GPU device has enough memory.
            ## If not, cuda_batch_mode is turned on
            
            #### automatically determine if cuda_batch_mode is on or off
            self.total_GPU_memory = torch.cuda.get_device_properties(0).total_memory
            if self.energy.size * self.energy.itemsize < self.total_GPU_memory / 10:
                self.cuda_batch_mode = False
            else:
                self.cuda_batch_mode = True

            #### cuda_batch_mode can be enforced by passing the argument cuda_batch_mode
            if cuda_batch_mode is not None:
                self.cuda_batch_mode = cuda_batch_mode

        ## when cuda_batch_mode is not used, we can copy all the energy data to GPU
        if self.cuda and not self.cuda_batch_mode:
            self.energy_cuda = torch.from_numpy(self.energy).cuda()

        ## when cuda_batch_mode is used, we need to decide on the batch_size based on
        ## both the memory of the GPU device and the size of energy matrix
        if self.cuda and self.cuda_batch_mode:

            ## batch size for seperating conformations
            self.conf_batch_size = int(self.total_GPU_memory/20/self.energy.shape[0]/self.energy.itemsize)
            self.conf_batch_size = min(1024, self.conf_batch_size)
            self.conf_num_batches = int(torch.sum(self.num_conf_cuda)) // self.conf_batch_size
            if torch.sum(self.num_conf_cuda) % self.conf_batch_size != 0:
                self.conf_num_batches = self.conf_num_batches + 1

        ## set self._solve_mbar_equation to specific function based on settings
        if self.cuda and not self.cuda_batch_mode:
            if self.verbose:
                print("solve MBAR equation using CUDA and the batch mode is off")
            self._solve_mbar_equation = self._solve_mbar_equation_cuda
            self.calculate_free_energies_of_perturbed_states = self._calculate_free_energies_of_perturbed_states_cuda
        elif self.cuda and self.cuda_batch_mode:
            if self.verbose:
                print("solve MBAR equation using CUDA and the batch mode is on")
            self._solve_mbar_equation = self._solve_mbar_equation_cuda_batch
            self.calculate_free_energies_of_perturbed_states = self._calculate_free_energies_of_perturbed_states_cuda_batch            
        else:
            if self.verbose:
                print("solve MBAR equation using CPU")
            self._solve_mbar_equation = self._solve_mbar_equation_cpu
            self.calculate_free_energies_of_perturbed_states = self._calculate_free_energies_of_perturbed_states_cpu            

        if not self.bootstrap:
            ## biasing energy added to states with nonzero conformations
            ## they are the variables that need to be optimized
            self.bias_energy = None

            ## result of free energies
            self.F = None
            self.F_std = None

            ## log of the mixed distribution probablity of each conformation
            self.log_prob_mix = None

        else:
            self.bias_energy_bootstrap = []
            self.F_bootstrap = []
            self.log_prob_mix_bootstrap = []
            self.conf_idx_bootstrap = []
            
        self.nit = None
        
        ## solve MBAR equations through minimization
        self._solve()
        
    def _calculate_loss_and_grad_cpu(self, bias_energy):
        """ calculate the loss and gradient for the FastMBAR objective function using CPUs.

        Parameters
        ----------
        bias_energy : 1-D ndarray with size of (M,)
        
        Returns:
        -------
        loss: the value of FastMBAR objective function
        grad: a 1-D array with a size of (M,).    
              the gradient of FastMBAR objective function with repsect to bias_energy_nz.
        """
        
        assert self.num_states == len(bias_energy)
        
        biased_energy = self.energy + bias_energy.reshape((-1,1))
        biased_energy_min = biased_energy.min(0, keepdims = True)
        biased_energy_center = biased_energy - biased_energy_min
        exp_biased_energy = np.exp(-biased_energy_center)
        sum_exp_biased_energy = np.sum(exp_biased_energy, 0)
        
        loss = np.mean(np.log(sum_exp_biased_energy)-biased_energy_min.reshape(-1)) + \
               np.sum(self.num_conf_ratio * bias_energy)
                       
        grad = - np.mean(exp_biased_energy / sum_exp_biased_energy, 1) + \
               self.num_conf_ratio
        
        return loss, grad
        
    def _solve_mbar_equation_cpu(self, initial_bias_energy = None, verbose = False):
        """ solve the MBAR equation using CPUs for all states by minimizing a convex
        function with the initial_bias_energy as a starting point.

        Parameters
        ----------
        initial_bias_energy: 1-D float ndarray with size of (M,)
            starting point used to solve the MBAR equations
        verbose: bool, optional
            if verbose is true, the detailed information of running
            LBFGS optimization is printed.
        Returns
        -------
        F: 1-D float array with size of (M,)
            the relative unitless free energies for all states

        bias_energy: 1-D float array with size of (M,)
            the bias energy that minimizes the FastMBAR objective function

        log_prob_mix: 1-D float array with size of (N,)
            the log of the mixed distribution probability
        """

        if initial_bias_energy is None:
            initial_bias_energy = np.zeros(self.num_states)
        
        options = {'disp': verbose, 'gtol': 1e-8}        
        results = optimize.minimize(self._calculate_loss_and_grad_cpu, initial_bias_energy, jac=True, method='L-BFGS-B', tol=1e-12, options = options)
        
        x = results['x']
        self.nit = results['nit']
        self.nfev = results['nfev']
        
        bias_energy = x
        F = - np.log(self.num_conf_ratio) - bias_energy
        bias_energy = bias_energy + F[0]
        F = F - F[0]
        
        ## calculate self.log_prob_mix for each conformation
        biased_energy = self.energy + bias_energy.reshape((-1,1))
        biased_energy_min = np.min(biased_energy, 0, keepdims = True)        
        log_prob_mix = np.log(np.sum(np.exp(-(biased_energy - biased_energy_min)), 0)) - biased_energy_min.reshape(-1)
        
        return F, bias_energy, log_prob_mix
    
    def _calculate_free_energies_of_perturbed_states_cpu(self, energy_perturbed):
        """ calculate free energies for perturbed states.

        Parameters
        -----------
        energy_perturbed: 2-D float ndarray with size of (L,N)
            each row of the energy_perturbed matrix represents a state and 
            the value energy_perturbed[l,n] represents the reduced energy
            of the n'th conformation in the l'th perturbed state.
        Returns
        -------
        F_mean: 1-D float ndarray with a size of (L,)
            the relative free energies of the perturbed states.
            it is a mean of multiple estimations if bootstrap is used,
        F_std: 1-D float ndarray with a size of (L,)
            the standard deviation of the estimated F. it is esimated 
            using bootstrap. when bootstrap is off, it is None
        """
        F_mean = None
        F_std = None
        if self.bootstrap:
            F_list =  []
            for k in range(self.bootstrap_num_rep):
                tmp = -energy_perturbed[:,self.conf_idx_bootstrap[k]] - self.log_prob_mix_bootstrap[k]
                F = -(np.log(np.mean(np.exp(tmp-np.max(tmp,1,keepdims=True)), 1)) + np.max(tmp, 1))
                F_list.append(F)
            F_mean = np.mean(F_list, 0)
            F_std = np.std(F_list, 0)
        else:
            tmp = -energy_perturbed - self.log_prob_mix
            F_mean = -(np.log(np.mean(np.exp(tmp-np.max(tmp,1,keepdims=True)), 1)) + np.max(tmp, 1))
        return F_mean, F_std
    
    def _calculate_loss_and_grad_cuda(self, bias_energy):
        """ calculate the loss and gradient for the FastMBAR objective function using GPUs.

        Parameters
        ----------
        bias_energy : 1-D ndarray with size of (M,)
        
        Returns:
        -------
        loss: the value of FastMBAR objective function
        grad: a 1-D array with a size of (M,).    
              the gradient of FastMBAR objective function with repsect to bias_energy.
        """
        assert(self.num_states == len(bias_energy))
        
        bias_energy = self.energy_cuda.new_tensor(bias_energy)
        biased_energy = self.energy_cuda + bias_energy.reshape((-1,1))
        biased_energy_min = biased_energy.min(0, keepdim = True)[0]
        biased_energy = biased_energy - biased_energy_min
        
        exp_biased_energy = torch.exp(-biased_energy)
        del biased_energy
        sum_exp_biased_energy = torch.sum(exp_biased_energy, 0)

        loss = torch.mean(torch.log(sum_exp_biased_energy) -
                          biased_energy_min.reshape(-1)) \
               + torch.sum(self.num_conf_ratio_cuda * bias_energy)
        
        grad = - torch.mean(exp_biased_energy / sum_exp_biased_energy, 1) + \
               self.num_conf_ratio_cuda

        return (loss.cpu().numpy().astype(np.float64),
                grad.cpu().numpy().astype(np.float64))
    
    def _solve_mbar_equation_cuda(self, initial_bias_energy = None, verbose = False):
        """ solve the MBAR equation using a GPU for all states by minimizing a convex
        function with the initial_bias_energy as a starting point.

        Parameters
        ----------
        initial_bias_energy: 1-D float ndarray with size of (M,)
            starting point used to solve the MBAR equations
        verbose: bool, optional
            if verbose is true, the detailed information of running
            LBFGS optimization is printed.
        Returns
        -------
        F: 1-D float array with size of (M,)
            the relative unitless free energies for all states

        bias_energy: 1-D float array with size of (M,)
            the bias energy that minimizes the FastMBAR objective function

        log_prob_mix: 1-D float array with size of (N,)
            the log of the mixed distribution probability
        """
        
        if initial_bias_energy is None:
            initial_bias_energy = np.zeros(self.num_states)
        options = {'disp': verbose, 'gtol': 1e-8}
        self.x_records = []
        results = optimize.minimize(self._calculate_loss_and_grad_cuda, initial_bias_energy, jac=True, method='L-BFGS-B', tol=1e-16, options = options)
        
        x = results['x']
        self.nit = results['nit']
        self.nfev = results['nfev']

        bias_energy = self.energy_cuda.new_tensor(x)
        F = - torch.log(self.num_conf_ratio_cuda) - bias_energy
        bias_energy = bias_energy + F[0]
        F = F - F[0]

        biased_energy = self.energy_cuda + bias_energy.reshape((-1,1))
        biased_energy_min = torch.min(biased_energy, 0, keepdim = True)[0]
        log_prob_mix = torch.log(torch.sum(torch.exp(-(biased_energy - biased_energy_min)), 0)) - biased_energy_min.reshape(-1)
        
        return F.cpu().numpy(), bias_energy.cpu().numpy(), log_prob_mix.cpu().numpy()

    def _calculate_free_energies_of_perturbed_states_cuda(self, energy_perturbed):
        """ calculate free energies for perturbed states.

        Parameters
        -----------
        energy_perturbed: 2-D float ndarray with size of (L,N)
            each row of the energy_perturbed matrix represents a state and 
            the value energy_perturbed[l,n] represents the reduced energy
            of the n'th conformation in the l'th perturbed state.
        Returns
        -------
        F_mean: 1-D float ndarray with a size of (L,)
            the relative free energies of the perturbed states.
            it is a mean of multiple estimations if bootstrap is used,
        F_std: 1-D float ndarray with a size of (L,)
            the standard deviation of the estimated F. it is esimated 
            using bootstrap. when bootstrap is off, it is None
        """
        
        F_mean = None
        F_std = None
        energy_perturbed = self.energy_cuda.new_tensor(energy_perturbed)
        if self.bootstrap:
            F_list =  []
            for k in range(self.bootstrap_num_rep):
                log_prob_mix = self.energy_cuda.new_tensor(self.log_prob_mix_bootstrap[k])
                tmp = -energy_perturbed[:,self.conf_idx_bootstrap[k]] - log_prob_mix
                F = -(torch.log(torch.mean(torch.exp(tmp-torch.max(tmp,1,keepdim=True)[0]), 1)) + torch.max(tmp, 1)[0])
                F_list.append(F)                
            F_mean = torch.mean(torch.stack(F_list), 0)
            F_std = torch.std(torch.stack(F_list), 0)
        else:
            log_prob_mix = self.energy_cuda.new_tensor(self.log_prob_mix)
            tmp = -energy_perturbed - log_prob_mix
            F_mean = -(torch.log(torch.mean(torch.exp(tmp-torch.max(tmp,1,keepdim=True)[0]), 1)) + torch.max(tmp, 1)[0])

        F_mean = F_mean.cpu().numpy()
        if F_std is not None:
            F_std = F_std.cpu().numpy()
            
        return F_mean, F_std
    
    def _calculate_loss_and_grad_cuda_batch(self, bias_energy):
        """ calculate the loss and gradient for the FastMBAR objective function using GPUs in batch mode.

        Parameters
        ----------
        bias_energy : 1-D ndarray with size of (M,)
        
        Returns:
        -------
        loss: the value of FastMBAR objective function
        grad: a 1-D array with a size of (M,).    
              the gradient of FastMBAR objective function with repsect to bias_energy.
        """
        
        bias_energy = torch.from_numpy(bias_energy).cuda()
        
        loss = 0.0
        grad = 0.0

        ## loop through batches of conformations and accumulate the loss and grad
        for idx_batch in range(self.conf_num_batches):
            energy_cuda = torch.from_numpy(self.energy[:, idx_batch*self.conf_batch_size:(idx_batch+1)*self.conf_batch_size]).cuda()
            biased_energy = energy_cuda + bias_energy.reshape((-1,1))
            biased_energy_min = biased_energy.min(0, keepdim = True)[0]
            biased_energy = biased_energy - biased_energy_min
            exp_biased_energy = torch.exp(-biased_energy)            
            del biased_energy
            sum_exp_biased_energy = torch.sum(exp_biased_energy, 0)

            loss = loss + torch.sum(torch.log(sum_exp_biased_energy) -
                              biased_energy_min.reshape(-1))            
            grad = grad - torch.sum(exp_biased_energy / sum_exp_biased_energy, 1)
            
        loss = loss/self.tot_num_conf + torch.sum(self.num_conf_ratio_cuda * bias_energy)
        grad = grad/self.tot_num_conf + self.num_conf_ratio_cuda

        return (loss.cpu().numpy().astype(np.float64),
                grad.cpu().numpy().astype(np.float64))
    

    def _solve_mbar_equation_cuda_batch(self, initial_bias_energy = None, verbose = False):
        """ solve the MBAR equation using a GPU in a batch mode for all states 
            by minimizing a convex function with the initial_bias_energy as 
            a starting point.

        Parameters
        ----------
        initial_bias_energy: 1-D float ndarray with size of (M,)
            starting point used to solve the MBAR equations
        verbose: bool, optional
            if verbose is true, the detailed information of running
            LBFGS optimization is printed.
        Returns
        -------
        F: 1-D float array with size of (M,)
            the relative unitless free energies for all states

        bias_energy: 1-D float array with size of (M,)
            the bias energy that minimizes the FastMBAR objective function

        log_prob_mix: 1-D float array with size of (N,)
            the log of the mixed distribution probability
        """        
        
        if initial_bias_energy is None:
            initial_bias_energy = np.zeros(self.num_states)
            
        options = {'disp': verbose, 'gtol': 1e-8}
        self.x_records = []
        results = optimize.minimize(self._calculate_loss_and_grad_cuda_batch,
                                    initial_bias_energy, jac=True,
                                    method='L-BFGS-B', tol=1e-16,
                                    options = options)
        
        x = results['x']
        self.nit = results['nit']
        self.nfev = results['nfev']        
        
        bias_energy = torch.from_numpy(x).cuda()
        F = - torch.log(self.num_conf_ratio_cuda) - bias_energy
        bias_energy = bias_energy + F[0]
        F = F - F[0]
        
        log_prob_mix = []
        ## loop through conformations
        for idx_batch  in range(self.conf_num_batches):
            energy_cuda_batch = torch.from_numpy(self.energy[:, idx_batch*self.conf_batch_size:(idx_batch+1)*self.conf_batch_size]).cuda()
            biased_energy = energy_cuda_batch + bias_energy.reshape((-1,1))
            biased_energy_min = torch.min(biased_energy, 0, keepdim = True)[0]
            log_prob_mix_batch = torch.log(torch.sum(torch.exp(-(biased_energy - biased_energy_min)), 0)) - biased_energy_min.reshape(-1) 
            log_prob_mix.append(log_prob_mix_batch)
            
        log_prob_mix = torch.cat(log_prob_mix)
        
        return F.cpu().numpy(), bias_energy.cpu().numpy(), log_prob_mix.cpu().numpy()

    def _calculate_free_energies_of_perturbed_states_cuda_batch(self, energy_perturbed):
        """ calculate free energies for perturbed states.

        Parameters
        -----------
        energy_perturbed: 2-D float ndarray with size of (L,N)
            each row of the energy_perturbed matrix represents a state and 
            the value energy_perturbed[l,n] represents the reduced energy
            of the n'th conformation in the l'th perturbed state.
        Returns
        -------
        F_mean: 1-D float ndarray with a size of (L,)
            the relative free energies of the perturbed states.
            it is a mean of multiple estimations if bootstrap is used,
        F_std: 1-D float ndarray with a size of (L,)
            the standard deviation of the estimated F. it is esimated 
            using bootstrap. when bootstrap is off, it is None
        """
        
        F_mean = []
        F_std = []

        ## batch size for seperating states
        states_batch_size = 16
        states_num_batches = energy_perturbed.shape[0]//states_batch_size
        if energy_perturbed.shape[0] % states_batch_size != 0:
            states_num_batches = states_num_batches + 1

        for idx_batch in range(states_num_batches):
            energy_perturbed_batch = torch.from_numpy(energy_perturbed[idx_batch*states_batch_size:(idx_batch+1)*states_batch_size,:]).cuda()
            if self.bootstrap:
                F_list =  []
                for k in range(self.bootstrap_num_rep):
                    log_prob_mix = torch.from_numpy(self.log_prob_mix_bootstrap[k]).cuda()
                    tmp = -energy_perturbed_batch[:,self.conf_idx_bootstrap[k]] - log_prob_mix
                    F = -(torch.log(torch.mean(torch.exp(tmp-torch.max(tmp,1,keepdim=True)[0]), 1)) + torch.max(tmp, 1)[0])
                    F_list.append(F)                
                F_mean.append(torch.mean(torch.stack(F_list), 0))
                F_std.append(torch.std(torch.stack(F_list), 0))
            else:
                log_prob_mix = torch.from_numpy(self.log_prob_mix).cuda()
                tmp = -energy_perturbed_batch - log_prob_mix
                F_mean.append(-(torch.log(torch.mean(torch.exp(tmp-torch.max(tmp,1,keepdim=True)[0]), 1)) + torch.max(tmp, 1)[0]))

        F_mean = torch.cat(F_mean)
        F_mean = F_mean.cpu().numpy()
        if len(F_std) != 0:
            F_std = torch.cat(F_std)
        else:
            F_std = None
        return F_mean, F_std
    
    def _solve(self):        
        """ solve the MBAR equation """
                
        if self.bootstrap:
            num_conf = self.num_conf.astype(np.int)
            num_conf_cumsum = list(np.cumsum(num_conf))
            num_conf_cumsum.pop(-1)
            num_conf_cumsum = [0] + num_conf_cumsum
            initial_bias_energy = np.zeros(self.num_states)
            bootstrap_F = []
            for _k in range(self.bootstrap_num_rep):
                conf_idx = []
                
                for i in range(len(num_conf)):
                    len_seq = num_conf[i] ## using circular block bootstrap
                    num_sample_block = int(np.ceil(len_seq/self.bootstrap_block_size))
                    idxs = np.random.choice(int(len_seq), num_sample_block, replace = True)
                    sample_idx = []
                    for idx in idxs:
                        sample_idx += list(range(idx, idx+self.bootstrap_block_size))

                    for k in range(len(sample_idx)):
                        sample_idx[k] = np.mod(sample_idx[k], len_seq)
                        
                    if len(sample_idx) > len_seq:
                        pop_idx = np.random.choice(len(sample_idx), 1)[0]
                        for k in range(pop_idx, pop_idx+len(sample_idx)-len_seq):
                            k = np.mod(k, len(sample_idx))
                            sample_idx.pop(k)
                        
                    assert(len(sample_idx) == len_seq)
                    conf_idx += [ idx + num_conf_cumsum[i] for idx in sample_idx]
                    
                assert(len(conf_idx) == self.tot_num_conf)
                sub_mbar = FastMBAR(self.energy[:, conf_idx], self.num_conf, self.cuda, cuda_batch_mode = self.cuda_batch_mode)
                sub_F, sub_bias_energy, sub_log_prob_mix = sub_mbar._solve_mbar_equation(initial_bias_energy, self.verbose)
                initial_bias_energy = sub_bias_energy
                bootstrap_F.append(sub_F)
                self.F_bootstrap.append(sub_F)
                self.bias_energy_bootstrap.append(sub_bias_energy)
                self.log_prob_mix_bootstrap.append(sub_log_prob_mix)
                self.conf_idx_bootstrap.append(conf_idx)

            self.F = np.mean(self.F_bootstrap, 0)
            self.F_std = np.std(self.F_bootstrap, 0)
            
            self.F = self.F - self.F[0]
            self.bias_energy = np.mean(self.bias_energy_bootstrap, 0)
            self.log_prob_mix = np.mean(self.log_prob_mix_bootstrap, 0)
            
        else:
            F, bias_energy, log_prob_mix = self._solve_mbar_equation(verbose = self.verbose)
            self.F = F
            self.bias_energy = bias_energy
            self.log_prob_mix = log_prob_mix

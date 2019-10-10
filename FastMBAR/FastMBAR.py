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
    def __init__(self, energy, num_conf, cuda = False, cuda_batch_mode = None):
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
        
        self.energy = energy.astype(np.float64)
        self.num_conf = num_conf.astype(energy.dtype)        
        self.num_states = energy.shape[0]
        self.tot_num_conf = energy.shape[1]
        
        self.flag_zero = num_conf == 0
        self.flag_nz = num_conf != 0
        
        self.energy_zero = self.energy[self.flag_zero, :]
        self.num_states_zero = self.energy_zero.shape[0]
        
        self.energy_nz = self.energy[self.flag_nz, :]
        self.num_states_nz = self.energy_nz.shape[0]        
        self.num_conf_nz = self.num_conf[self.flag_nz]        
        self.num_conf_nz_ratio = self.num_conf_nz / float(self.tot_num_conf)
                
        ## moving data to GPU if cuda is used
        if self.cuda:
            self.num_conf_nz = torch.from_numpy(self.num_conf_nz).cuda()
            self.num_conf_nz_ratio = torch.from_numpy(self.num_conf_nz_ratio).cuda()
            self.flag_zero = torch.BoolTensor(self.flag_zero.astype(np.int)).cuda()
            self.flag_nz = torch.BoolTensor(self.flag_nz.astype(np.int)).cuda()

            ## check if the GPU device has enough memory. If not, cuda_batch_mode is
            ## turned on
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
            self.energy_zero = torch.from_numpy(self.energy_zero).cuda()
            self.energy_nz = torch.from_numpy(self.energy_nz).cuda()            

        ## when cuda_batch_mode is used, we need to decide on the batch_size based on
        ## both the memory of the GPU device and the size of energy matrix
        if self.cuda and self.cuda_batch_mode:

            ## batch size for seperating conformations
            self.conf_batch_size = int(self.total_GPU_memory/20/self.energy_nz.shape[0]/self.energy_nz.itemsize)
            self.conf_batch_size = min(2048, self.conf_batch_size)
            self.conf_num_batches = torch.sum(self.num_conf_nz) / self.conf_batch_size
            self.conf_num_batches = int(self.conf_num_batches.item()) + 1

            ## batch size for seperating states
            self.state_batch_size = int(self.total_GPU_memory/20/self.energy_zero.shape[1]/self.energy_zero.itemsize)
            self.state_batch_size = min(64, self.state_batch_size)
            self.state_num_batches = self.num_states_zero // self.state_batch_size + 1

        ## set self._solve_mbar_equation to specific function based on settings
        if self.cuda and not self.cuda_batch_mode:
            print("solve MBAR equation using CUDA and the batch mode is off")            
            self._solve_mbar_equation = self._solve_mbar_equation_cuda
        elif self.cuda and self.cuda_batch_mode:
            print("solve MBAR equation using CUDA and the batch mode is on")            
            self._solve_mbar_equation = self._solve_mbar_equation_cuda_batch
        else:
            print("solve MBAR equation using CPU")                        
            self._solve_mbar_equation = self._solve_mbar_equation_cpu
                        
        ## biasing energy added to states with nonzero conformations
        ## they are the variables that need to be optimized
        self.bias_energy_nz = None

        ## result of free energies
        self.F = None

        self.nit = None
        
    def _calculate_loss_and_grad_nz_cpu(self, bias_energy_nz):
        """ calculate the loss and gradient for the FastMBAR objective function using CPUs.

        Parameters
        ----------
        bias_energy_nz : 1-D ndarray with size of (M,)
        
        Returns:
        -------
        loss: the value of FastMBAR objective function
        grad: a 1-D array with a size of (M,).    
              the gradient of FastMBAR objective function with repsect to bias_energy_nz.
        """
        
        assert(self.num_states_nz == len(bias_energy_nz))

        biased_energy_nz = self.energy_nz + bias_energy_nz.reshape((-1,1))
        biased_energy_nz_min = biased_energy_nz.min(0, keepdims = True)
        biased_energy_nz_center = biased_energy_nz - biased_energy_nz_min
        exp_biased_energy_nz = np.exp(-biased_energy_nz_center)
        sum_exp_biased_energy_nz = np.sum(exp_biased_energy_nz, 0)
        
        loss = np.mean(np.log(sum_exp_biased_energy_nz)-biased_energy_nz_min.reshape(-1)) + \
               np.sum(self.num_conf_nz_ratio * bias_energy_nz)
                       
        grad = - np.mean(exp_biased_energy_nz / sum_exp_biased_energy_nz, 1) + \
               self.num_conf_nz_ratio
        
        return loss, grad

        
    def _solve_mbar_equation_cpu(self, initial_bias_energy = None, verbose = False):
        """ calculate the relative free energies on CPUs for all states by
        solving the MBAR equation once with the initial guess of initial_bias_energy

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

        self.bias_energy_nz: 1-D float array with size of (M,)
            the bias energy that minimizes the FastMBAR objective function
        """

        if initial_bias_energy is None:
            #initial_bias_energy = np.mean(self.energy_nz, 1)
            initial_bias_energy = np.zeros(self.num_states_nz)

        # x, f, d = optimize.fmin_l_bfgs_b(self._calculate_loss_and_grad_nz_cpu,
        #                                  initial_bias_energy,
        #                                  iprint = verbose)
            
        # options = {'factr':4503.599627370496, 'pgtol': 1e-12}
        # x, f, d = optimize.fmin_l_bfgs_b(self._calculate_loss_and_grad_nz_cpu,
        #                                  initial_bias_energy,
        #                                  iprint = verbose,
        #                                  **options)

        options = {'disp': verbose, 'gtol': 1e-8}
        # self.x_records = []
        # def callback(xk):
        #     self.x_records.append(xk)            
        # results = optimize.minimize(self._calculate_loss_and_grad_nz_cpu, initial_bias_energy, jac=True, method='L-BFGS-B', tol=1e-12, options = options, callback = callback)        
        # results = optimize.fmin_l_bfgs_b(self._calculate_loss_and_grad_nz_cpu, initial_bias_energy, iprint = 1)
        
        results = optimize.minimize(self._calculate_loss_and_grad_nz_cpu, initial_bias_energy, jac=True, method='L-BFGS-B', tol=1e-12, options = options)
        
        x = results['x']
        self.nit = results['nit']
        self.nfev = results['nfev']
        
        self.bias_energy_nz = x
        self.F_nz = - np.log(self.num_conf_nz_ratio) - self.bias_energy_nz

        if self.num_states_zero != 0:
            biased_energy_nz = self.energy_nz + self.bias_energy_nz.reshape((-1,1))
            biased_energy_nz_min = np.min(biased_energy_nz, 0, keepdims = True)
        
            tmp = np.log(np.sum(np.exp(-(biased_energy_nz - biased_energy_nz_min)), 0)) - biased_energy_nz_min.reshape(-1)
            tmp = -self.energy_zero - tmp

            F_zero = -(np.log(np.mean(np.exp(tmp-np.max(tmp,1,keepdims=True)), 1)) + np.max(tmp, 1))                
            self.F = np.zeros(self.num_states)
            self.F[self.flag_nz] = self.F_nz
            self.F[self.flag_zero] = F_zero
        else:
            self.F = self.F_nz

        return self.F, self.bias_energy_nz

    
    def _calculate_loss_and_grad_nz_cuda(self, bias_energy_nz):
        """ calculate the loss and gradient for the FastMBAR objective function using GPUs.

        Parameters
        ----------
        bias_energy_nz : 1-D ndarray with size of (M,)
        
        Returns:
        -------
        loss: the value of FastMBAR objective function
        grad: a 1-D array with a size of (M,).    
              the gradient of FastMBAR objective function with repsect to bias_energy_nz.
        """
        
        bias_energy_nz = self.energy_nz.new_tensor(bias_energy_nz)
        biased_energy_nz = self.energy_nz + bias_energy_nz.reshape((-1,1))
        biased_energy_nz_min = biased_energy_nz.min(0, keepdim = True)[0]
        biased_energy_nz = biased_energy_nz - biased_energy_nz_min
        
        exp_biased_energy_nz = torch.exp(-biased_energy_nz)
        del biased_energy_nz
        sum_exp_biased_energy_nz = torch.sum(exp_biased_energy_nz, 0)

        loss = torch.mean(torch.log(sum_exp_biased_energy_nz) -
                          biased_energy_nz_min.reshape(-1)) \
               + torch.sum(self.num_conf_nz_ratio * bias_energy_nz)
        
        grad = - torch.mean(exp_biased_energy_nz / sum_exp_biased_energy_nz, 1) + \
               self.num_conf_nz_ratio

        return (loss.cpu().numpy().astype(np.float64),
                grad.cpu().numpy().astype(np.float64))

    def _calculate_loss_and_grad_nz_cuda_batch(self, bias_energy_nz):
        """ calculate the loss and gradient for the FastMBAR objective function using GPUs in batch mode.

        Parameters
        ----------
        bias_energy_nz : 1-D ndarray with size of (M,)
        
        Returns:
        -------
        loss: the value of FastMBAR objective function
        grad: a 1-D array with a size of (M,).    
              the gradient of FastMBAR objective function with repsect to bias_energy_nz.
        """
        
        bias_energy_nz = torch.from_numpy(bias_energy_nz).cuda()        
        
        loss = 0.0
        grad = 0.0

        ## loop through batches of conformations and accumulate the loss and grad
        for idx_batch in range(self.conf_num_batches):
            self.energy_nz_cuda = torch.from_numpy(self.energy_nz[:, idx_batch*self.conf_batch_size:(idx_batch+1)*self.conf_batch_size]).cuda()
            biased_energy_nz = self.energy_nz_cuda + bias_energy_nz.reshape((-1,1))
            biased_energy_nz_min = biased_energy_nz.min(0, keepdim = True)[0]
            biased_energy_nz = biased_energy_nz - biased_energy_nz_min
            exp_biased_energy_nz = torch.exp(-biased_energy_nz)            
            del biased_energy_nz
            sum_exp_biased_energy_nz = torch.sum(exp_biased_energy_nz, 0)

            loss = loss + torch.sum(torch.log(sum_exp_biased_energy_nz) -
                              biased_energy_nz_min.reshape(-1))            
            grad = grad - torch.sum(exp_biased_energy_nz / sum_exp_biased_energy_nz, 1)
            
        loss = loss/self.tot_num_conf + torch.sum(self.num_conf_nz_ratio * bias_energy_nz)
        grad = grad/self.tot_num_conf + self.num_conf_nz_ratio            

        return (loss.cpu().numpy().astype(np.float64),
                grad.cpu().numpy().astype(np.float64))
    
    def _solve_mbar_equation_cuda(self, initial_bias_energy = None, verbose = False):
        """ calculate the relative free energies on GPUs for all states by
        solving the MBAR equation once with the initial guess of initial_bias_energy

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

        self.bias_energy_nz: 1-D float array with size of (M,)
            the bias energy that minimizes the FastMBAR objective function
        """
        
        if initial_bias_energy is None:
            initial_bias_energy = np.zeros(self.num_states_nz)        
        # x, f, d = optimize.fmin_l_bfgs_b(self._calculate_loss_and_grad_nz_cuda,
        #                                  initial_bias_energy,
        #                                  iprint = verbose)
        options = {'disp': verbose, 'gtol': 1e-8}
        self.x_records = []
        # def callback(xk):
        #     self.x_records.append(xk)            
        # results = optimize.minimize(self._calculate_loss_and_grad_nz_cuda, initial_bias_energy, jac=True, method='L-BFGS-B', tol=1e-12, options = options, callback = callback)
        results = optimize.minimize(self._calculate_loss_and_grad_nz_cuda, initial_bias_energy, jac=True, method='L-BFGS-B', tol=1e-12, options = options)
        
        x = results['x']
        self.nit = results['nit']
        self.nfev = results['nfev']        
        
        self.bias_energy_nz = self.energy_nz.new_tensor(x)    
        self.F_nz = - torch.log(self.num_conf_nz_ratio) - self.bias_energy_nz

        if self.num_states_zero != 0:
            biased_energy_nz = self.energy_nz + self.bias_energy_nz.reshape((-1,1))
            biased_energy_nz_min = torch.min(biased_energy_nz, 0, keepdim = True)[0]
        
            tmp = torch.log(torch.sum(torch.exp(-(biased_energy_nz - biased_energy_nz_min)), 0)) - biased_energy_nz_min.reshape(-1) 
            tmp = -self.energy_zero - tmp

            F_zero = -(torch.log(torch.mean(torch.exp(tmp-torch.max(tmp,1,keepdim=True)[0]), 1)) + torch.max(tmp, 1)[0])
                
            self.F = self.energy_nz.new_zeros(self.num_states)
            self.F[self.flag_nz] = self.F_nz
            self.F[self.flag_zero] = F_zero
        else:
            self.F = self.F_nz

        return self.F.cpu().numpy(), x

    def _solve_mbar_equation_cuda_batch(self, initial_bias_energy = None, verbose = False):
        """ calculate the relative free energies on GPUs in batch mode for all states by
        solving the MBAR equation once with the initial guess of initial_bias_energy

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

        self.bias_energy_nz: 1-D float array with size of (M,)
            the bias energy that minimizes the FastMBAR objective function
        """
        
        if initial_bias_energy is None:
            initial_bias_energy = np.zeros(self.num_states_nz)        
        options = {'disp': verbose, 'gtol': 1e-8}
        self.x_records = []
        results = optimize.minimize(self._calculate_loss_and_grad_nz_cuda_batch, initial_bias_energy, jac=True, method='L-BFGS-B', tol=1e-12, options = options)
        
        x = results['x']
        self.nit = results['nit']
        self.nfev = results['nfev']        
        
        self.bias_energy_nz = torch.from_numpy(x).cuda()
        self.F_nz = - torch.log(self.num_conf_nz_ratio) - self.bias_energy_nz

        
        if self.num_states_zero != 0:                    
            mix_logprob = []
            ## loop through conformations
            for idx_batch  in range(self.conf_num_batches):
                self.energy_nz_cuda_batch = torch.from_numpy(self.energy_nz[:, idx_batch*self.conf_batch_size:(idx_batch+1)*self.conf_batch_size]).cuda()
                biased_energy_nz = self.energy_nz_cuda_batch + self.bias_energy_nz.reshape((-1,1))
                biased_energy_nz_min = torch.min(biased_energy_nz, 0, keepdim = True)[0]
                mix_logprob_batch = torch.log(torch.sum(torch.exp(-(biased_energy_nz - biased_energy_nz_min)), 0)) - biased_energy_nz_min.reshape(-1) 
                mix_logprob.append(mix_logprob_batch)
            mix_logprob = torch.cat(mix_logprob)

            num_batches = self.num_states_zero // self.state_batch_size + 1
            
            F_zero = []
            ## loop through states with zero conformations
            for idx_batch in range(self.state_num_batches):
                energy_zero = torch.from_numpy(self.energy_zero[idx_batch*self.state_batch_size:(idx_batch + 1)*self.state_batch_size, :]).cuda()
                tmp = -energy_zero - mix_logprob
                F_zero_batch = -(torch.log(torch.mean(torch.exp(tmp-torch.max(tmp,1,keepdim=True)[0]), 1)) + torch.max(tmp, 1)[0])
                F_zero.append(F_zero_batch)
            F_zero = torch.cat(F_zero)
            
            self.F = F_zero.new_zeros(self.num_states)
            self.F[self.flag_nz] = self.F_nz
            self.F[self.flag_zero] = F_zero

        else:
            self.F = self.F_nz
            
        return self.F.cpu().numpy(), x
    
    def calculate_free_energies(self, bootstrap = False, block_size = 3, num_rep = 10,
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
            mean_F, _ = self._solve_mbar_equation(verbose = verbose)
            std_F = None
        mean_F = mean_F - mean_F[0]
        return (mean_F, std_F)

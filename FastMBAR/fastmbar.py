from typing import Union
import torch
import numpy as np
import scipy.optimize as optimize
import math


class FastMBAR:
    """
    The FastMBAR class is initialized with an energy matrix and an array
    of num of conformations. The corresponding MBAR equation is solved
    in the constructor. Therefore, the relative free energies of states
    used in the energy matrix is calculated in the constructor. The
    method **calculate_free_energies_for_perturbed_states**
    can be used to calculated the relative free energies of perturbed states.
    """

    def __init__(
        self,
        energy: Union[np.ndarray, torch.Tensor],
        num_conf: Union[np.ndarray, torch.Tensor],
        cuda: bool = False,
        cuda_batch_mode: Union[bool, None] = None,
        bootstrap: bool = False,
        bootstrap_block_size: int = 3,
        bootstrap_num_rep: int = 100,
        verbose: bool = False,
        method: str = "Newton",
    ) -> None:
        """Initializer for the class FastMBAR

        Parameters
        ----------
        energy : 2D ndarray or 2D tensor
            It has a size of (M x N), where M is the number of states
            and N is the total number of conformations. The entry energy[i,j]
            is the reduced (unitless) energy of conformation j in state i.
            If bootstrapping is used to calculate the uncertainty, the order
            of conformations matters. Conformations sampled from one state
            need to occpy a continuous chunk of collumns. Conformations sampled
            from state k need to occupy collumns to the left of conformations
            sampled from state l if k < l. If bootstrapping is not used, then
            the order of conformation does not matter.
        num_conf: 1D int ndarray or 1D tensor
            It should have a size of M, where num_conf[i] is the num of
            conformations sampled from state i. Therefore, np.sum(num_conf)
            has to be equal to N. All entries in num_conf have to be strictly
            greater than 0.
        cuda: bool, optional (default=False)
            If it is set to be True, then the calculation will be run on
            a graphical processing unit (GPU) using CUDA.
        cuda_batch_mode: bool, optional (default=None)
            The batch mode is turned on when the size of the energy matrix is
            too large for the memory of a GPU. When cuda_batch_mode is True,
            the energy matrix will be split into multiple batches which are used
            sequentially. If cuda_batch_mode = None, it will be set automatically
            based on the size of energy and the memory of the avaible GPU device.
        bootstrap: bool, optional (default=False)
            If bootstrap is True, the uncertainty of the calculated free energies
            will be estimate using block bootstraping.
        bootstrap_block_size: int, optional (default=3)
            block size used in block bootstrapping
        bootstrap_num_rep: int, optional (default=100)
            number of repreats in block bootstrapping
        verbose: bool, optional (default=False)
            if verbose is true, the detailed information of solving MBAR equations
            is printed.
        method: str, optional (default="Newton")
            the method used to solve the MBAR equation. The default is Newton's method.
        """

        #### check the parameters: energy and num_conf
        ## Note that both self.energy and self.conf will be on CPUs no matter
        ## if cuda is True or False.

        ## energy needs to be a 2-D ndarray or a 2-D tensor.
        ## convert it into double precision if not
        if isinstance(energy, np.ndarray):
            energy = energy.astype(np.float64)
            self.energy = torch.from_numpy(energy)
        elif isinstance(energy, torch.Tensor):
            self.energy = energy.double()
        else:
            raise TypeError("energy has to be a 2-D ndarray or a 2-D tensor.")

        ## num_conf needs to be a 1-D ndarray or a 1-D tensor.
        if isinstance(num_conf, np.ndarray):
            num_conf = num_conf.astype(np.float64)
            self.num_conf = torch.from_numpy(num_conf)
        elif isinstance(num_conf, torch.Tensor):
            self.num_conf = num_conf.double()
        else:
            raise TypeError("num_conf has to be a 1-D ndarray or a 1-D tensor.")

        ## check the shape of energy and num_conf
        if energy.ndim != 2:
            raise ValueError("energy has to be a 2-D ndarray or a 2-D tensor.")
        if num_conf.ndim != 1:
            raise ValueError("num_conf has to be a 1-D ndarray or a 1-D tensor.")
        if energy.shape[0] != num_conf.shape[0]:
            raise ValueError(
                "the number of rows in energy has to be equal to the length of num_conf."
            )

        ## check if the number of conformations sampled from each state is greater than 0
        if torch.sum(self.num_conf <= 0) > 0:
            raise ValueError(
                "all entries in num_conf have to be strictly greater than 0."
            )

        ## check if the total number of conformations is equal to the number of columns in energy
        if torch.sum(self.num_conf) != self.energy.shape[1]:
            raise ValueError(
                "the sum of num_conf has to be equal to the number of columns in energy."
            )

        self.M = self.energy.shape[0]
        self.N = self.energy.shape[1]

        self.cuda = cuda
        if self.cuda is True:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.cuda:
            ## check if the GPU device has enough memory. If not, cuda_batch_mode is turned on

            ## automatically determine if cuda_batch_mode is on or off
            self._total_GPU_memory = torch.cuda.get_device_properties(0).total_memory
            if (
                self.energy.numel() * self.energy.element_size()
                < self._total_GPU_memory / 10
            ):
                self.cuda_batch_mode = False
            else:
                self.cuda_batch_mode = True

            ## cuda_batch_mode can be enforced by passing the argument cuda_batch_mode
            if cuda_batch_mode is not None:
                self.cuda_batch_mode = cuda_batch_mode

        ## when cuda_batch_mode is used, we need to decide on the batch_size based on
        ## both the memory of the GPU device and the size of energy matrix
        self._batch_size = None
        if self.cuda and self.cuda_batch_mode:
            ## batch size for seperating conformations
            self._batch_size = int(
                self._total_GPU_memory
                / 20
                / self.energy.shape[0]
                / self.energy.element_size()
            )
            self._batch_size = min(1024, self._batch_size)

        ## whether to use bootstrap to estimate the uncertainty of the calculated free energies
        ## bootstrap needs to be a boolean
        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap has to be a boolean.")
        self.bootstrap = bootstrap

        ## block size used in block bootstrapping
        if not isinstance(bootstrap_block_size, int):
            raise TypeError("bootstrap_block_size has to be an integer.")
        self.bootstrap_block_size = bootstrap_block_size

        ## number of repeats in block bootstrapping
        if not isinstance(bootstrap_num_rep, int):
            raise TypeError("bootstrap_num_rep has to be an integer.")
        self.bootstrap_num_rep = bootstrap_num_rep

        ## whether to print the detailed information of solving MBAR equations
        ## verbose needs to be a boolean
        if not isinstance(verbose, bool):
            raise TypeError("verbose has to be a boolean.")
        self.verbose = verbose

        ## method used to solve the MBAR equation
        if not isinstance(method, str):
            raise TypeError("method has to be a string.")
        if method not in ["Newton", "L-BFGS-B"]:
            raise ValueError("method has to be Newton or L-BFGS-B.")
        self.method = method

        ## solve the MBAR equation
        if self.bootstrap is False:
            dF_init = self.energy.new_zeros(self.M - 1)
            dF = _solve_mbar(
                dF_init.to(self.device),
                self.energy.to(self.device),
                self.num_conf.to(self.device),
                self.method,
                self._batch_size,
                verbose=self.verbose,
            ).cpu()

            ## shift self._F such that \sum_i F[i]*num_conf[i] = 0
            F = torch.cat([dF.new_zeros(1), dF])
            pi = self.num_conf / self.N
            self._F = F - torch.sum(pi * F)

            b = -self._F - torch.log(self.num_conf)
            self._log_prob_mix = torch.logsumexp(-(self.energy + b[:, None]), dim=0)
            self._log_mu = -self.log_prob_mix

            ## Compute self._F_cov under the constraint that \sum_i F[i]*num_conf[i] = 0

            ## There are two ways to compute the covariance matrix of self._F
            ## See equation 4.2 and 6.4 in the following paper for details
            ## "Kong, A., et al. "A theory of statistical models for Monte Carlo integration."
            ## Journal of the Royal Statistical Society Series B: Statistical Methodology 65.3
            ## (2003): 585-604." https://doi.org/10.1111/1467-9868.00404

            ## The first way as shown in the following uses equation 4.2
            ## this method is more general than the following method, meaning that
            ## it can also be used to compute covariance matrix for perturbed states.
            ## Therefore it is used here.
            self.P = torch.exp(
                -(self.energy - self._F[:, None] + self._log_prob_mix)
            ).t()
            W = torch.diag(self.num_conf)
            Q, R = torch.linalg.qr(self.P)
            A = torch.eye(self.M, device=W.device) - R @ W @ R.T
            self._F_cov = R.T @ torch.linalg.pinv(A, hermitian=True, rtol=1e-8) @ R

            # # The second way uses equation 6.4
            # if self._batch_size is None:
            #     H = _compute_hessian_log_likelihood_of_F(
            #         self._F, self.energy, self.num_conf
            #     )
            # else:
            #     H = _compute_hessian_log_likelihood_of_F_in_batch(
            #         self._F, self.energy, self.num_conf, self._batch_size
            #     )
            # Hp = H.new_zeros((self.M + 1, self.M + 1)).cpu()
            # Hp[0 : self.M, 0 : self.M] = H
            # Hp[self.M, 0 : self.M] = -self.num_conf
            # Hp[0 : self.M, self.M] = -self.num_conf
            # Hp[self.M, self.M] = 0
            # self._F_cov = torch.linalg.inv(-Hp)[0 : self.M, 0 : self.M]
            # self._F_cov = self._F_cov - torch.diag(1 / self.num_conf) + 1 / self.N

            self._F_std = self._F_cov.diagonal().sqrt()

            self._DeltaF = self._F[None, :] - self._F[:, None]
            self._DeltaF_cov = (
                self._F_cov.diag()[:, None]
                + self._F_cov.diag()[None, :]
                - 2 * self._F_cov
            )
            self._DeltaF_std = self._DeltaF_cov.sqrt()

        elif self.bootstrap is True:
            dFs = []
            log_prob_mix = []
            dF_init = self.energy.new_zeros(self.M - 1)
            self._conf_idx = []
            for _ in range(self.bootstrap_num_rep):
                conf_idx = _bootstrap_conf_idx(
                    self.num_conf.to(torch.int64), self.bootstrap_block_size
                )
                dF = _solve_mbar(
                    dF_init.to(self.device),
                    self.energy[:, conf_idx].to(self.device),
                    self.num_conf.to(self.device),
                    self.method,
                    self._batch_size,
                    verbose=self.verbose,
                ).cpu()
                dF_init = dF
                dFs.append(dF.clone())

                F = torch.cat([dF.new_zeros(1), dF])
                ## shift F such that \sum_i F[i]*num_conf[i] = 0
                pi = self.num_conf / self.N
                F = F - torch.sum(pi * F)

                b = -F - torch.log(self.num_conf / self.N)
                log_prob_mix.append(
                    torch.logsumexp(-(self.energy[:, conf_idx] + b[:, None]), dim=0)
                )
                self._conf_idx.append(conf_idx)

            dF = torch.stack(dFs, dim=1)
            F = torch.cat([dF.new_zeros(1, dF.shape[1]), dF], dim=0)

            ## shift F such that \sum_i F[i]*num_conf[i] = 0
            pi = self.num_conf / self.N
            self._F_bootstrap = F - torch.sum(pi[:, None] * F, dim=0)

            self._F = torch.mean(self._F_bootstrap, dim=1)
            self._F_std = torch.std(self._F_bootstrap, dim=1)
            self._F_cov = torch.cov(self._F_bootstrap)

            self._log_prob_mix = torch.stack(log_prob_mix, dim=0)
            DeltaF = self._F_bootstrap[None, :, :] - self._F_bootstrap[:, None, :]
            self._DeltaF = torch.mean(DeltaF, dim=2)
            self._DeltaF_std = torch.std(DeltaF, dim=2)

    @property
    def F(self) -> np.ndarray:
        """Free energies of the states under the constraint :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return self._F.cpu().numpy()

    @property
    def F_std(self) -> np.ndarray:
        """Standard deviation of the free energies of the states under the constraint
        :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return self._F_std.cpu().numpy()

    @property
    def F_cov(self) -> np.ndarray:
        """Covariance matrix of the free energies of the states under the constraint
        :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return self._F_cov.cpu().numpy()

    @property
    def DeltaF(self) -> np.ndarray:
        """Free energy difference between states.
        :math:`\mathrm{DeltaF}[i,j]` is the free energy difference between state j and state i,
        i.e., :math:`\mathrm{DeltaF}[i,j] = F[j] - F[i]` .
        """
        return self._DeltaF.cpu().numpy()

    @property
    def DeltaF_std(self) -> np.ndarray:
        """Standard deviation of the free energy difference between states.
        :math:`\mathrm{DeltaF_std}[i,j]` is the standard deviation of the free energy
        difference :math:`\mathrm{DeltaF}[i,j]`.
        """
        return self._DeltaF_std.cpu().numpy()

    @property
    def log_prob_mix(self) -> np.ndarray:
        """the log probability density of conformations in the mixture distribution."""
        return self._log_prob_mix.cpu().numpy()

    def calculate_free_energies_of_perturbed_states(self, energy_perturbed):
        """calculate free energies for perturbed states.

        Parameters
        -----------
        energy_perturbed: 2-D float ndarray with size of (L,N)
            each row of the energy_perturbed matrix represents a state and
            the value energy_perturbed[l,n] represents the reduced energy
            of the n'th conformation in the l'th perturbed state.

        Returns
        -------
        results: dict
            a dictionary containing the following keys:

            **F** - the free energy of the perturbed states.

            **F_std** - the standard deviation of the free energy of the perturbed states. 

            **F_cov** - the covariance between the free energies of the perturbed states.

            **DeltaF** - :math:`\mathrm{DeltaF}[k,l]` is the free energy difference between state            
            :math:`k` and state :math:`l`, i.e., :math:`\mathrm{DeltaF}[k,l] = F[l] - F[k]` .

            **DeltaF_std** - the standard deviation of the free energy difference.
        """

        if isinstance(energy_perturbed, np.ndarray):
            energy_perturbed = energy_perturbed.astype(np.float64)
            energy_perturbed = torch.from_numpy(energy_perturbed)
        elif isinstance(energy_perturbed, torch.Tensor):
            energy_perturbed = energy_perturbed.double()
        else:
            raise TypeError("energy_perturbed has to be a 2-D ndarray or a 2-D tensor.")

        if energy_perturbed.ndim != 2:
            raise ValueError(
                "energy_perturbed has to be a 2-D ndarray or a 2-D tensor."
            )
        if energy_perturbed.shape[1] != self.energy.shape[1]:
            raise ValueError(
                "the number of columns in energy_perturbed has to be equal to the number of columns in energy."
            )

        L = energy_perturbed.shape[0]

        F = None
        F_cov = None
        F_std = None

        if not self.bootstrap:
            du = energy_perturbed + self._log_prob_mix
            F = -torch.logsumexp(-du, dim=1)

            F_ext = torch.cat([self._F, F])
            U = torch.cat([self.energy, energy_perturbed], dim=0).t()
            self._P = torch.exp(-(U - F_ext + self._log_prob_mix[:, None]))

            W = torch.diag(torch.cat([self.num_conf, self.num_conf.new_zeros(L)]))
            Q, R = torch.linalg.qr(self._P)
            A = torch.eye(self.M + L, device=W.device) - R @ W @ R.T
            F_cov = R.T @ torch.linalg.pinv(A, hermitian=True, rtol=1e-8) @ R

            F_cov = F_cov[self.M :, self.M :]
            F_std = F_cov.diagonal().sqrt()

            DeltaF = F[None, :] - F[:, None]
            DeltaF_cov = F_cov.diag()[:, None] + F_cov.diag()[None, :] - 2 * F_cov
            DeltaF_std = DeltaF_cov.sqrt()
        else:
            F_list = []
            for k in range(self.bootstrap_num_rep):
                du = energy_perturbed[:, self._conf_idx[k]] + self._log_prob_mix[k]
                F = -torch.logsumexp(-du, dim=1)
                F_list.append(F)
            F_list = torch.stack(F_list, dim=1)
            F = torch.mean(F_list, dim=1)
            F_std = torch.std(F_list, dim=1)
            F_cov = torch.cov(F_list)

            DeltaF = F[None, :] - F[:, None]
            DeltaF_std = torch.std(F_list[:, :, None] - F_list[:, None, :], dim=1)

        results = {
            "F": F.numpy(),
            "F_std": F_std.numpy(),
            "F_cov": F_cov.numpy(),
            "DeltaF": DeltaF.numpy(),
            "DeltaF_std": DeltaF_std.numpy(),
        }

        return results

def _compute_logp_of_F(F, energy, num_conf):
    logp = (
        torch.sum(num_conf * F)
        - torch.logsumexp(-(energy.T - torch.log(num_conf) - F), dim=1).sum()
    )
    return logp


def _compute_loss_and_grad_of_dF(dF, energy, num_conf):
    """Compute the loss function and the gradient of dF given energy and num_conf.
    Parameters
    ----------
    dF: 1-D float tensor with size (M-1)
        The free energy difference between state 0 and other states. The entry
        dF[i] is the free energy difference between state 0 and state i+1.
    energy: 2-D float tensor with size (M x N)
        A 2-D tensor with size of M x N, where M is the number of stats
        and N is the total number of conformations. The entry energy[i,j]
    num_conf: 1-D int tensor with size (M)
        A 1-D tensor with size of M, where num_conf[i] is the num of
        conformations sampled from state i. Therefore, torch.sum(num_conf)
        has to be equal to N. All entries in num_conf have to be strictly
        greater than 0.
    """

    # compute the loss function
    F = torch.cat([dF.new_zeros(1), dF])
    logp = _compute_logp_of_F(F, energy, num_conf)
    loss = -logp / num_conf.sum()
    u = energy - F[:, None] - torch.log(num_conf)[:, None]

    # compute the gradient of the loss function
    p = torch.softmax(-u, dim=0)
    grad = torch.mean(p, dim=1) - num_conf / num_conf.sum()
    grad = grad[1:]

    return loss, grad


def _compute_loss_and_grad_of_dF_in_batch(dF, energy, num_conf, batch_size: int):
    """This is the batch version of _compute_loss_and_grad_of_dF.
    The batch version is used when the size of energy is too large to fit into the
    memory of a GPU. The energy matrix is split into multiple batches and the
    calculation is done sequentially.

    The parameters are the same as _compute_loss_and_grad_of_dF.
    """

    n = num_conf
    N = torch.sum(n)
    n_over_N = n / N

    if energy.shape[1] % batch_size == 0:
        num_batch = energy.shape[1] // batch_size
    else:
        num_batch = energy.shape[1] // batch_size + 1

    F = torch.cat([dF.new_zeros(1), dF])
    loss = -torch.sum(num_conf * F)
    grad = -num_conf

    for i in range(num_batch):
        energy_batch = energy[:, i * batch_size : (i + 1) * batch_size]
        u = energy_batch.cuda() - torch.log(num_conf)[:, None] - F[:, None]
        loss += torch.logsumexp(-u, dim=0).sum()

        p = torch.softmax(-u, dim=0)
        grad += torch.sum(p, dim=1)

    loss = loss / N
    grad = grad / N
    grad = grad[1:]
    return loss, grad


def _compute_hessian_log_likelihood_of_F(F, energy, num_conf):
    u = energy - F[:, None] - torch.log(num_conf)[:, None]
    p = torch.softmax(-u, dim=0)
    pp = torch.matmul(p, p.T)
    H = pp - torch.diag(torch.sum(p, 1))
    return H


def _compute_hessian_log_likelihood_of_F_in_batch(F, energy, num_conf, batch_size: int):
    if energy.shape[1] % batch_size == 0:
        num_batch = energy.shape[1] // batch_size
    else:
        num_batch = energy.shape[1] // batch_size + 1

    H = F.new_zeros((F.shape[0], F.shape[0])).cuda()
    for i in range(num_batch):
        energy_batch = energy[:, i * batch_size : (i + 1) * batch_size]
        u = energy_batch.cuda() - F[:, None] - torch.log(num_conf)[:, None]

        p = torch.softmax(-u, dim=0)
        pp = torch.matmul(p, p.T)
        H += pp - torch.diag(torch.sum(p, 1))

    return H


def _compute_hessian_loss_of_dF(dF, energy, num_conf):
    N = torch.sum(num_conf)
    F = torch.cat([dF.new_zeros(1), dF])
    H = _compute_hessian_log_likelihood_of_F(F, energy, num_conf)
    H = -H[1:, 1:] / N
    return H


def _compute_hessian_loss_of_dF_in_batch(dF, energy, num_conf, batch_size: int):
    N = torch.sum(num_conf)
    F = torch.cat([dF.new_zeros(1), dF])
    H = _compute_hessian_log_likelihood_of_F_in_batch(F, energy, num_conf, batch_size)
    H = -H[1:, 1:] / N
    return H


def _solve_mbar(dF_init, energy, num_conf, method, batch_size=None, verbose=False):
    if batch_size is not None:
        if method == "Newton":
            results = fmin_newton(
                f=_compute_loss_and_grad_of_dF_in_batch,
                hess=_compute_hessian_loss_of_dF_in_batch,
                x_init=dF_init,
                args=(energy, num_conf, batch_size),
                verbose=verbose,
            )
            return results["x"]

        elif method == "L-BFGS-B":
            options = {"disp": verbose, "gtol": 1e-8}
            results = optimize.minimize(
                lambda x: [
                    r.cpu().numpy()
                    for r in _compute_loss_and_grad_of_dF_in_batch(
                        energy.new_tensor(x), energy, num_conf, batch_size
                    )
                ],
                dF_init.cpu().numpy(),
                jac=True,
                method="L-BFGS-B",
                tol=1e-12,
                options=options,
            )
            return torch.from_numpy(results["x"])
    else:
        if method == "Newton":
            results = fmin_newton(
                f=_compute_loss_and_grad_of_dF,
                hess=_compute_hessian_loss_of_dF,
                x_init=dF_init,
                args=(energy, num_conf),
                verbose=verbose,
            )
            return results["x"]
        elif method == "L-BFGS-B":
            options = {"disp": verbose, "gtol": 1e-8}
            results = optimize.minimize(
                lambda x: [
                    r.cpu().numpy()
                    for r in _compute_loss_and_grad_of_dF(
                        energy.new_tensor(x), energy, num_conf
                    )
                ],
                dF_init.cpu().numpy(),
                jac=True,
                method="L-BFGS-B",
                tol=1e-12,
                options=options,
            )
            return torch.from_numpy(results["x"])


def _bootstrap_conf_idx(num_conf, bootstrap_block_size):
    num_conf_cumsum = torch.cumsum(num_conf, dim=0).tolist()
    num_conf_cumsum.pop(-1)
    num_conf_cumsum = [0] + num_conf_cumsum
    conf_idx = []
    for i in range(len(num_conf)):
        len_seq = int(num_conf[i])  ## using circular block bootstrap
        num_sample_block = int(np.ceil(len_seq / bootstrap_block_size))
        idxs = torch.randint(0, len_seq, (num_sample_block,))
        sample_idx = torch.cat(
            [torch.arange(idx, idx + bootstrap_block_size) for idx in idxs]
        )
        sample_idx = torch.remainder(sample_idx, len_seq)
        sample_idx = sample_idx[0:len_seq]
        sample_idx = sample_idx + num_conf_cumsum[i]
        conf_idx.append(sample_idx)
    conf_idx = torch.cat(conf_idx)
    return conf_idx


def fmin_newton(f, hess, x_init, args=(), verbose=True, eps=1e-12, max_iter=300):
    """Minimize a function with the Newton's method.

    For details of the Newton's method, see Chapter 9.5 of Prof. Boyd's book
    `Convex Optimization <https://web.stanford.edu/~boyd/cvxbook/>`_.

    Args:
        f (callable): the objective function to be minimized.
            f(x:Tensor, *args) ->  (y:Tensor, grad:Tensor), where
            y is the function value and grad is the gradient.
        hess (callable): the function to compute the Hessian matrix.
            hess(x:Tensor, *args) -> a two dimensional tensor.
        x_init (Tensor): the initial value.
        args (tuple): extra parameters for f and hess.
        verbose (bool): whether to print minimizing log information.
        eps (float): tolerance for stopping
        max_iter (int): maximum number of iterations.
    """

    ## constants used in backtracking line search
    alpha, beta = 0.01, 0.2

    ## Newton's method for minimizing the function
    indx_iter = 0

    N_func = 0

    if verbose:
        print("==================================================================")

        print("                RUNNING THE NEWTON'S METHOD                     \n")
        print("                           * * *                                \n")
        print(f"                    Tolerance EPS = {eps:.5E}                  \n")

    ## check if x_init is a pytorch tensor
    if isinstance(x_init, torch.Tensor):
        x = x_init.clone().detach().requires_grad_(False)
    else:
        x = x_init

    while indx_iter < max_iter:
        loss, grad = f(x, *args)
        N_func += 1

        if isinstance(x_init, torch.Tensor):
            with torch.no_grad():
                H = hess(x, *args)
        else:
            H = hess(x, *args)

        if isinstance(grad, torch.Tensor):
            newton_direction = torch.linalg.solve(H, -grad)
            newton_decrement_square = torch.sum(-grad * newton_direction)
        else:
            newton_direction = np.linalg.solve(H, -grad)
            newton_decrement_square = np.sum(-grad * newton_direction)

        if verbose:
            print(
                f"At iterate {indx_iter:4d}; f= {loss.item():.5E};",
                f"|1/2*Newton_decrement^2|: {newton_decrement_square.item()/2:.5E}\n",
            )

        if newton_decrement_square / 2.0 <= eps:
            break

        ## backtracking line search
        max_ls_iter = 100
        step_size = 1.0

        indx_ls_iter = 0
        while indx_ls_iter < max_ls_iter:
            target_loss, _ = f(x + step_size * newton_direction, *args)

            N_func += 1
            approximate_loss = loss + step_size * alpha * (-newton_decrement_square)
            if target_loss < approximate_loss:
                break
            else:
                step_size = step_size * beta
            indx_ls_iter += 1

        x = x + step_size * newton_direction
        indx_iter += 1

    if verbose:
        print("N_iter   = total number of iterations")
        print("N_func   = total number of function and gradient evaluations")
        print("F        = final function value \n")
        print("             * * *     \n")
        print("N_iter    N_func        F")
        print(f"{indx_iter+1:6d}    {N_func:6d}    {loss.item():.6E}")
        print(f"  F = {loss.item():.12f} \n")

        if newton_decrement_square / 2.0 <= eps and indx_iter < max_iter:
            print("CONVERGENCE: 1/2*Newton_decrement^2 < EPS")
        else:
            print("CONVERGENCE: NUM_OF_ITERATION REACH MAX_ITERATION")

    return {
        "x": x,
        "N_iter": indx_iter + 1,
        "N_func": N_func,
        "F": loss.item(),
        "half_newton_decrement_square": newton_decrement_square / 2.0,
    }

from numpy import ndarray
import numpy as np
from .fastmbar import FastMBAR

class UmbrellaSamplingAnalyzer()
    def __init__(
        self,
        cv,
        biasing_parameters,
        biasing_energy_function
    ):
        """ Initialize the UmbrellaSamplingAnalyzer class

        Parameters
        ----------
        cv : list or tuple of 1d or 2d ndarray. 
            The collective variables biased in the umbrella sampling simulations.
            The length of cv is the number of windows/simulations in the umbrella sampling.
            cv[k] is the value of the collective variable for conformations sampled from the kth window.
            When 1d collective variable is used, cv[i] is a 1d ndarray of shape (N_k, ) where N_k is 
            the number of conformations sampled from the kth window. 
            When 2d collective variable is used, cv[i] is a 2d ndarray of shape (N_k, 2).
            It will raise an error if collective variables of more than 2 dimensions are used.
        biasing_parameters : list of 1d ndarray
            The biasing parameters used in the umbrella sampling simulations.
            The length of biasing_parameters is the number of windows/simulations in the umbrella sampling.
            biasing_parameters[k] is the biasing parameter used in the kth window.
        biasing_energy_function: a callable object
            A callable object that compute the biasing energy givein the biasing parameters and the collective
            variable. The biasing_energy_function is called as biasing_energy_function(biasing_parameters[k], cv[k])
            where k is the window index. The biasing_energy_function should return a ndarray of shape (N_k,).
            The returned energy need to be unitless.
        """

        assert isinstance(cv, list) or isinstance(cv, tuple), "cv must be a list or tuple"
        cv = [ np.array(v) for v in cv]
        if cv[0].ndim == 1:
            assert all([v.ndim == 1 for v in cv]), "cv must be a list of ndarray of same dimension"
        elif cv[0].ndim == 2:
            assert all([v.ndim == 2 for v in cv]), "cv must be a list of ndarray of same dimension"
            assert all([v.shape[1] == 2 for v in cv]), "when two dimensional cv is used, cv[k] must be of shape (N_k, 2)"

        self.M = len(cv)        
        self.num_conf = np.array([v.shape[0] for v in cv], dtype=np.int64)
        self.N = np.sum(self.num_conf)
        self.cv = np.concatenate(cv)

        assert isinstance(biasing_parameters, list), "biasing_parameters must be a list"
        self.biasing_parameters = [np.array(v) for v in biasing_parameters]
        assert all([v.ndim == 1 for v in self.biasing_parameters]), "biasing_parameters must be a list of 1d ndarray"
        assert len(self.biasing_parameters) == self.M, "biasing_parameters must have the same length as the number of windows"

        assert callable(biasing_energy_function), "biasing_energy_function must be a callable object"
        self.biasing_energy_function = biasing_energy_function
        
        self.reduced_energy = np.zeros((self.M, self.N), dtype=np.float64)
        for k in range(self.M):
            self.reduced_energy[k,:] = self.biasing_energy_function(self.biasing_parameters[k], self.cv)

                







        
        

        



class BiasedSimulationAnalyzer:
    def __init__(
        self,
        unbiased_energy,
        num_conf,
        temperature,
        biased_variable,
        biasing_energy=None,
        biasing_function=None,
        cuda=False,
        cuda_batch_mode=None,
        bootstrap=False,
        bootstrap_block_size=3,
        bootstrap_num_rep=5,
        verbose=False,
    ):
        """Initialize the BiasedSimulationAnalyzer class

        Parameters
        ----------
        unbiased_energy : None or ndarray of shape (M, N), where M is the number of
          states and N is the total number of configurations. The unbiased_energy[i,j]
          is the unbiased energy of the jth configuration in the ith state without biasing.
          The unit of energy is kJ/mol. Similarly to the FastMBAR class, if bootstrapping
          is used to calculate the uncertainty, the order of conformations matters.
          Conformations sampled from one state need to occpy a continuous chunk of collumns.
          Conformations sampled from state k need to occupy collumns to the left of conformations
          sampled from state l if k < l. If bootstrapping is not used, then the order of conformation
          does not matter.

          When the following three conditions are satified, the unbiased_energy can be set as None:
          1. For a fixed configuration, the unbiased energy is the same for all states, 
          i.e., unbiased_energy[i,j] does not depend on i.
          2. The temperature is the same for all states.
          3. The analyer will only be used to compute the potential of mean force (PMF) at the same
          temperature as the biased simulation.
          The three conditions are often satified in umbrella sampling simulations.

        num_conf : ndarray of shape (M, ). The num_conf[i] is the
          number of configurations sampled from the ith state. The sum of num_conf has to be equal to
          the total number of configurations N.
        temperature : float or ndarray of shape (M, ). When it is
          float, the same temperature is used for all states. When it is ndarray, the temperature[i]
          is the temperature of the ith state. The unit of temperature is Kelvin.
        biased_variable : ndarray of shape (N, ). The biased_variable[j] is the value of
          the biased variable of the jth configuration.
        biasing_energy : ndarray of shape (M, N). The biasing_energy[i,j] is the biasing energy
          evaluated for the jth configuration in the ith state. The unit of energy is kJ/mol.
          If biasing_energy is provided, then bias_function is ignored.
        biasing_function : a callable object. Instead of providing the biasing_energy, the user
          can provide a callable object that takes the state index and the biased variable as input
          and returns the biasing energy. The biasing_function is called as biasing_function(i, x)
          where i is the state index and x is the biased_variable. The biasing_function should return
          a ndarray of shape (N, ) where N is the total number of configurations. The unit of energy
          is kJ/mol. Note that biasing_function is ignored if biasing_energy is provided.
        """

        self.num_conf = np.array(num_conf, dtype = np.int64)
        assert self.num_conf.ndim == 1, "num_conf must be a 1D array"
        self.M = self.num_conf.shape[0]
        self.N = np.sum(self.num_conf)

        if unbiased_energy is not None:
            self.unbiased_energy = np.array(unbiased_energy)
            assert self.unbiased_energy.ndim == 2, "unbiased_energy must be a 2D array"
            assert (
                self.unbiased_energy.shape[0] == self.M
            ), "unbiased_energy must have the same number of states as num_conf"
            assert (
                self.unbiased_energy.shape[1] == self.N
            ), "unbiased_energy must have the same number of configurations as num_conf"
        else:
            self.unbiased_energy = 0

        self.temperature = np.array(temperature)
        assert self.temperature.ndim == 1, "temperature must be a 1D array"
        assert (
            self.temperature.shape[0] == self.M or self.temperature.shape[0] == 1
        ), "temperature must have the same length as the number of states or be a scalar"

        self.biased_variable = np.array(biased_variable)
        assert self.biased_variable.ndim == 1, "biased_variable must be a 1D array"
        assert (
            self.biased_variable.shape[0] == self.N
        ), "biased_variable must have the same length as the total number of configurations"

        if biasing_energy is not None:
            self.biasing_energy = np.array(biasing_energy)
            assert self.biasing_energy.ndim == 2, "biasing_energy must be a 2D array"
            assert (
                self.biasing_energy.shape[0] == self.M
            ), "biasing_energy must have the same number of states as unbiased_energy"
            assert (
                self.biasing_energy.shape[1] == self.N
            ), "biasing_energy must have the same number of configurations as unbiased_energy"
            if biasing_function is not None:
                warnings.warn(
                    "biasing_function is ignored since biasing_energy is provided"
                )
        else:
            assert biasing_function is not None, "biasing_function must be provided"
            self.biasing_energy = np.zeros((self.M, self.N))
            for i in range(self.M):
                self.biasing_energy[i, :] = biasing_function(i, self.biased_variable)

        self.kB = 0.00831446261815324  # kJ/mol/K

        self.cuda = cuda
        self.cuda_batch_mode = cuda_batch_mode
        self.bootstrap = bootstrap
        self.bootstrap_block_size = bootstrap_block_size
        self.bootstrap_num_rep = bootstrap_num_rep
        self.verbose = verbose

        reduced_energy = (self.unbiased_energy + self.biasing_energy)/(self.kB*self.temperature)
        self.fastmbar = FastMBAR(
            reduced_energy,
            self.num_conf,
            cuda=self.cuda,
            cuda_batch_mode=self.cuda_batch_mode,
            bootstrap=self.bootstrap,
            bootstrap_block_size=self.bootstrap_block_size,
            bootstrap_num_rep=self.bootstrap_num_rep,
            verbose=self.verbose,
        )


[![Build Status](https://travis-ci.org/xqding/FastMBAR.svg?branch=master)](https://travis-ci.org/xqding/FastMBAR)
[![Anaconda-Server Badge](https://anaconda.org/shawn_ding/fastmbar/badges/downloads.svg)](https://anaconda.org/shawn_ding/fastmbar)

# A Fast Solver for Large Scale MBAR/UWHAM Equations
The multistate Bennett acceptance ratio (MBAR) and unbinned weighted histogram analysis method (UWHAM) are
widely imployed approaches to calculate free energies of multiple thermodynamcis states.
They are routinely used in alchemical free energy calculations, umbrella sampling, and 
temperature/Hamiltonian replica exchange simulations to calculate free engies and 
potentials of mean force (PMF).

`FastMBAR` is a solver written in Python to solve large scale
multistate Bennett acceptance ratio (MBAR)/unbinned weighted histogram analysis method (UWHAM) 
equations. Compared with the widely used python package `pymbar`, 
`FastMBAR` is 3 times faster on CPUs and more than two orders of 
magnitude faster on GPUs.

## Installation
`FastMBAR` can be installed via `conda` or `pip` using the following commands:  
  * using `conda`:  
    - If you want to install `FastMBAR` and dependent packages in the main conda environment, run the command:   
      `conda install -c shawn_ding -c pytorch fastmbar`.
    - If you want to install them in a specific conda environment, run the following commands:  
      `conda create -n myenv_name`  
      `conda install -n myenv_name -c shawn_ding -c pytorch fastmbar`,  
      where you can replace `myenv_name` with whatever name you want.
      
  * using `pip`:  
    `pip install FastMBAR`
## Usage
The input to the MBAR/UWHAM equations are an energy (unitless) matrix and 
an integer array consisting of numbers of configurations sampled from states of interest. 
Let's say that we are interested in calculating relative free energies of a system in _M_ thermodynamics states.
The _j_ th state has an energy function of _U_<sub>_j_</sub>(_x_).
From each of the _M_ states, system configurations _x_ are sampled based on Boltzmann distributions.
Let's assume that the number of configurations sampled from the _j_ th state is _n_<sub>_j_</sub>, _j_ = 1,2,...,_M_.
To use these configurations to calculate the relative free energies of the _M_ states using MBAR, 
we need to prepare the following energy matrix **U** in the blue bracket:
 ![Figure](./energy_matrix.png)
Elements of the above matrix are energies of all the sampled configurations evaluated in all _M_ states.
In addition to the energy matrix **U**, we also need an integer array **v** consisting of 
the numbers of configurations sampled from _M_ states, 
i.e., **v** = (_n_<sub>_1_</sub>, _n_<sub>_2_</sub>, ..., _n_<sub>_M_</sub>).

With the energy matrix **U** and the number of configuration array **v**, 
we can use the following Python command to calculate the relative free energies of 
the _M_ states:
```Python
# import the FastMBAR package
from FastMBAR import *

# construct a FastMBAR object with the energy matrix and the number of configuration array
fastmbar = FastMBAR(energy = U, num_conf = v, cuda=False) # set cuda = True if you want to run the calcuation on GPUs

# the relative free energies of the M states is available via fastmbar.F
print(fastmbar.F)

# if you want to estimate the uncertainty using bootstrapping, change the above command into
fastmbar = FastMBAR(energy = U, num_conf = v, cuda=False, bootstrap = True)
print(fastmbar.F) ## mean of relative free energies
print(fastmbar.F_std) ## standard deviation of estimated relative free energies 
```

Using the object fastmbar, we can also calculate free energies of other states (also referred as perturbed states)
from which no conformations are sampled.
In order to do that, we need to prepare the perturbed energy matrix **U_perturbed** shown above in the red bracket.
Entries in the perturbed energy matrix are reduced energy values of conformations in perturbed states.
With the perturbed energy matrix **U_perturbed**, we can use the following command to calculate the relative free 
energies of the perturbed states:
```Python
# calcualte free energies by solving the MBAR equations
F_perturbed, F_perturbed_std = calculate_free_energies_of_perturbed_states(U_perturbed)
```

[![Build Status](https://travis-ci.org/xqding/FastMBAR.svg?branch=master)](https://travis-ci.org/xqding/FastMBAR)
[![Anaconda-Server Badge](https://anaconda.org/shawn_ding/fastmbar/badges/downloads.svg)](https://anaconda.org/shawn_ding/fastmbar)

# A Fast Solver for Large Scale MBAR/UWHAM Equations
The multistate Bennett acceptance ratio (MBAR) and unbinned weighted histogram analysis method (UWHAM) are widely imployed approaches to calculate free energies of multiple thermodynamcis states.
They are routinely used in alchemical free energy calculations, umbrella sampling, and temperature/Hamiltonian replica exchange simulations to calculate free engies and potentials of mean force (PMF).

`FastMBAR` is a solver written in Python to solve large scale multistate Bennett acceptance ratio (MBAR)/unbinned weighted histogram analysis method (UWHAM) equations. Compared with the widely used python package `pymbar`, `FastMBAR` is 3 times faster on CPUs and more than two orders of magnitude faster on GPUs.

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
The input to the MBAR/UWHAM equations are an energy (unitless) matrix and an integer array consisting of numbers of configurations sampled from states of interest. 
Let's say that we are interested in calculating relative free energies of a system in _M_ thermodynamics states.
The _j_ th state has an energy function of _U_<sub>_j_</sub>(_x_).
From each of the first _m_ states, system configurations _x_ are sampled based on Boltzmann distributions.
Let's assume that the number of configurations sampled from the _j_ th state is _n_<sub>_j_</sub>, _j_ = 1,2,...,_m_.
To use these configurations to calculate the relative free energies of the _M_ states using MBAR, we need to prepare the following energy matrix **U** in the blue bracket:
 ![Figure](./energy_matrix.png)
Elements of the above matrix are energies of all the sampled configurations evaluated in all _M_ states.
In addition to the energy matrix **U**, we also need an integer array **v** consisting of the numbers of configurations sampled from _M_ states, i.e., **v** = (_n_<sub>_1_</sub>, _n_<sub>_2_</sub>, ..., _n_<sub>_m_</sub>, 0, 0, ..., 0).
The last _M_-_m_ elements of the array **v** is zeros because no configurations are sample from those states.

With the energy matrix **U** and the number of configuration array **v**, we can use the following Python command to calculate the relative free energies of the _M_ states:
```Python
# import the FastMBAR package
import FastMBAR

# construct a FastMBAR object with the energy matrix and the number of configuration array
mbar = FastMBAR.FastMBAR(energy = U, num_conf = v, cuda=False) # set cuda = True if you want to run the calcuation on GPUs

# calcualte free energies by solving the MBAR equations
F = mbar.calculate_free_energies(bootstrap=False)

# if you want to estimate the uncertainty, change the above command into
# F, F_std = mbar.calculate_free_energies(bootstrap=True)
```

Usage
=====

1. When to use MBAR equations?
------------------------------

A general question in computational chemistry is to calculate
relative free energies of different thermodynamic states.
Let us assume there are :math:`M` thermodynamic states and
the :math:`k` th state has a reduced potential energy function of
:math:`U_k(x)`, where :math:`x` represents a system conformation.
The probability distribution of conformations :math:`x` in the :math:`k` th 
state is the Boltzmann distribution
:math:`p_k(x) \propto \exp\left(-U_k(x)\right)`.
(**Note** that :math:`U_k(x)` is the reduced  potential energy function and
has included the inverse temperature :math:`\beta = 1/k_bT`, i.e., if the original
potential energy function is :math:`E_k(x)`, :math:`U_k(x)` is defined as
:math:`\beta E_k(x)`. Therefore, :math:`U_k(x)` is a unitless quantity.)
The reduced free energy of the :math:`k` th state is
:math:`F_k = -\ln \int{\exp\left(-U_k(x)\right) dx}`.
(**Note** :math:`F_k` is the reduced free energy and is also unitless.
To convert it into a free energy with an energy unit, you need to mutiply it
by :math:`k_bT`, i.e., :math:`k_bT F_k`.)
There are many ways to compute the relative free energies of
the :math:`M` states and one of them is as follows:

1. Sample conformations from each of the :math:`M` states separately using
   either molecular dynamcis or Monte Carlo sampling.

2. Reweight the above sampled conformations from all the :math:`M` states
   to compute the relative free energies.

Let us assume that :math:`N_k` conformations are sampled from the :math:`k` th
state in the first step and they are represented as :math:`\{x^{n}_k, n = 1, ..., N_k\}` for
:math:`k = 1, ..., M`.
MBAR equations are used in the second step to reweight conformations and
compute relative free energies :math:`F_k` for :math:`k = 1, ..., M`.
After MBAR equations are solved in the second step, they can also be used
to calculate relative free energies of perturbed states from which no
conformations are sampled. Let us assume there are :math:`L` perturbed states
and the :math:`l` th perturbed state has a reduced potential energy function
of :math:`U^{\prime}_l(x)`.

The situation described above is quite general and it includes **alchemical
free energy calculation** and calculating potential of mean force (PMF) with
**umbrella sampling** or **temperature/Hamiltonian replica exchange simulations**.


2. How to use FastMBAR?
-----------------------
Two inputs are required to use FastMBAR: a matrix of reduced potential energies
and a vector of numbers of sampled conformations, both of which can be calculated
based on sampling.

If sampling is conducted as described above, the matrix of reduced potential
energies should be calculated as the matrix :math:`A_{M,N}` as shown in the
blue bracket in the following figure:

.. image:: ../../energy_matrix.pdf
	   
The matrix :math:`A` has :math:`M` rows and :math:`N` columns, where :math:`M` is
the number of states from which conformations are sampled and :math:`N` is the
total number of conformations sampled from all states, i.e.,
:math:`N = \sum_{k=1}^{M} N_k`.
The reduced potential energy of each conformation, no matter which state it is
sampled from, is evaluated in all :math:`M` states.
These :math:`M` reduced potential energies constitue one column of the matrix
:math:`A`.








	   

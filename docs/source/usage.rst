Usage
=====
* A general description of situations where MBAR equations are useful.

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

  MBAR equations are used in the second step to reweight conformations and
  compute relative free energies :math:`F_k` for :math:`k = 1, ..., M`.
  After MBAR equations are solved in the second step, they can also be used
  to calculate relative free energies of perturbed states from which no
  conformations are sampled. Let us assume there are :math:`L` perturbed states
  and the :math:`l` th perturbed state has a reduced potential energy function
  of :math:`U^{\prime}_l(x)`.

  
.. image:: ../../energy_matrix.pdf

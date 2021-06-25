.. FastMBAR documentation master file, created by
   sphinx-quickstart on Sat Oct 12 11:37:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FastMBAR!
============================================================

What is FastMBAR?
-----------------
FastMBAR is a Python solver for large scale MBAR/UWHAM equations.
It can calculate relative free energies of a large number of thermodynamic states.
It is useful for both alchemical free energy calculations and calculating potential of
mean force (PMF) in umbrella sampling and
temperature/Hamiltonian replica exchange simulations.


Why FastMBAR?
-------------
FastMBAR can use both CPUs and GPUs.
It is extramely fast for solving large scale MBAR/WHAM equations when it uses GPUs.
Moreover, it is not limited by the memory size of a GPU.
Therefore, you will find it especially useful when you want to calculate relative
free energies for a large number of states, such as calculating multiple
dimensional PMF with umbrella sampling and calculating free energies for
a large number of states in alchemical free energy approaches.

How to use FastMBAR?
--------------------
It just takes a few lines of commands to install and use FastMBAR.

.. toctree::
   :maxdepth: 2

   installation
   usage
   API
   examples
   references

   
.. The multistate Bennett acceptance ratio (MBAR) and
   unbinned weighted histogram analysis method (UWHAM) are
   widely used approaches to calculate free energies of
   multiple thermodynamical states.
   MBAR/UWHAM has several advantages comparing to the traditional WHAM method and 
   is routinely used in alchemical free energy calculations, umbrella sampling, and
   temperature/Hamiltonian replica exchange simulations to calculate
   relative free engines and potentials of mean force (PMF).
   **FastMBAR** is a solver written in Python to solve large scale MBAR/UWHAM equations.



   
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

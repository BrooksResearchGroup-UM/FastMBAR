.. _examples:

Examples
========

Install Extra Required Python Packges
-------------------------------------

Besides FastMBAR, the following examples requrie several other packages.
You need to install them in order to run these examples on your computer.
If you have installed Anaconda, it should be very easy to install them.
Most of them just need a single command in your Terminal.


* Install **OpenMM**
  
  All following examples require running molecular dynamics (MD).
  Here, in these examples, we will use `OpenMM <http://openmm.org>`_ whenever we need to run MD.
  Therefore, if you want to run the following examples step by step on your computer, you need
  to install OpenMM. If you have installed `Anaconda <https://www.anaconda.com>`_,  
  you can easily install OpenMM
  by following the instruction at `OpenMM`_.
  Although OpenMM is used in these examples, any other choices of MD engines, such as CHARMM,
  Gromacs, and Amber, can be used too. Therefore, no matter what kind of MD engine you use in your 
  study, you can always use FastMBAR to analyze samples from your simulations.

* Install **MDTraj**

  The package `MDTraj <http://mdtraj.org>`_ is used to load and process simulation trajectories.
  To install it, follow the intructions at `MDTraj`_.

* Install **tqdm**

  Package `tqdm <https://tqdm.github.io>`_ is not essential for calculations in these examples.
  It is used here to provide progress bars for simulation so that you can see an estimation
  of wall time used for umbrella sampling. To install tqdm, type the following command in a Terminal:

  .. code-block:: bash

     pip install tqdm


Examples
--------

.. toctree::
   :maxdepth: 1
   
   butane_PMF
   dialanine_PMF

   

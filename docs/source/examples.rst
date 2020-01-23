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
  to install `OpenMM`_. If you have installed `Anaconda <https://www.anaconda.com>`_,  you can easily install `OpenMM`_
  by following the instruction at `OpenMM Install <http://docs.openmm.org/latest/userguide/application.html#installing-openmm>`_.
  If you have installed Anaconda, you can easily install OpenMM use either

  .. code-block:: bash
		  
     conda install -c omnia -c conda-forge openmm

  or

  .. code-block:: bash

     conda install -c omnia/label/cuda92 -c conda-forge openmm
     
  The latter command will install OpenMM with CUDA support, so you need to replace the `cuda92` in the command with the correct
  CUDA version on your computer. Although `OpenMM`_ is used in these examples, any other choices of MD engines, such as CHARMM,
  Gromacs, and Amber, can be used too. Therefore, no matter what kind of MD engine you use in your study, you can always use
  FastMBAR to analyze samples from your simulations.

* Install **MDTraj**

  The package `MDTraj <http://mdtraj.org>`_ is used to load and process simulation trajectories.
  Following the intructions from `MDTraj Install <http://mdtraj.org/1.9.3/installation.html>`_, you can install MDTraj
  using the following command:

  .. code-block:: bash

     conda install -c conda-forge mdtraj

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

   

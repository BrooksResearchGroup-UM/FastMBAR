Installation
============

1. New to Python or Anaconda?
-----------------------------
`Python <https://www.python.org>`_ is a high-level programming language.
To install and use Python on your computer, I highly recommend you install
`Anaconda <https://www.anaconda.com>`_, which provides a platform to use
Python and managing Python packages.
You can download and install Anaconda by following the instructions on
https://www.anaconda.com/distribution/.

2. Install FastMBAR
-------------------

* using pip  
  
  Open a Terminal and run the following command depending on if you
  have installed Anaconda:
    
  * With Anaconda
  
    .. code-block:: bash

       conda install -c shawn_ding -c pytorch fastmbar
       

  * Without Anaconda

    .. code-block:: bash

       pip install FastMBAR

* Testing of Installation
  
  Run the following command in a terminal to test if
  FastMBAR has been installed successfully.

  .. code-block:: bash

     python -m FastMBAR.test_installation

  If FastMBAR has been successfully installed, it will
  output the following information:

  On a computer with GPUs, the above command will
  print the following output if FastMBAR has been
  installed correctly::
     
     ========================================
     Start testing FastMBAR:
     ========================================

     Without bootstrap
     ----------------------------------------
     RMSD (CPU calculation and reference results): 0.01 < 0.05. PASSED.
     RMSD (GPU calculation and reference results): 0.01 < 0.05. PASSED.
     RMSD (GPU-batch-mode calculation and reference results): 0.01 < 0.05. PASSED.

     With bootstrap
     ----------------------------------------
     RMSD (CPU calculation and reference results): 0.00 < 0.05. PASSED.
     RMSD (GPU calculation and reference results): 0.02 < 0.05. PASSED.
     RMSD (GPU-batch-mode calculation and reference results): 0.01 < 0.05. PASSED.
     ========================================
     ALL TESTS ARE PASSED.


  On a computer without GPUs, the above command will
  print the following output if FastMBAR has been
  installed correctly::
     
     ========================================
     Start testing FastMBAR:
     ========================================

     Without bootstrap
     ----------------------------------------
     RMSD (CPU calculation and reference results): 0.01 < 0.05. PASSED.

     With bootstrap
     ----------------------------------------
     RMSD (CPU calculation and reference results): 0.00 < 0.05. PASSED.
     ========================================
     ALL TESTS ARE PASSED.
     

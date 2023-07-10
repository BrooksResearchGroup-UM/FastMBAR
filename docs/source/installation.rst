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

2. Install FastMBAR with pip
--------------------------------

* **Install FastMBAR without CUDA support**

  If you do not have GPUs on your computers, you can still use FastMBAR on CPUs. 
  In this case, the installing FastMBAR is very easy. 
  Open a Terminal and run the following command
  
  .. code-block:: bash

     pip install -U FastMBAR

* **Install FastMBAR with CUDA support**

  If you have GPUs on your computers and you want to run FastMBAR on GPUs, 
  you need to install FastMBAR with CUDA support. 
  Because FastMBAR uses `PyTorch <https://pytorch.org>`_ for calculations on GPUs, 
  you will need to install PyTorch with CUDA support before installing FastMBAR:

  1. Install PyTorch with CUDA support.

     Follow the instructions at `PyTorch`_ to 
     install PyTorch with CUDA support. The specific command you need could
     depends on your operation system, Python version, and 
     CUDA version you have on your computers. 
     You can use the following command in a Terminal to check these version information.
  
     .. code-block:: bash
		       
        ## check Python version
        python --version
	  
        ## check CUDA version
        nvcc --version
		       
     If the command ``nvcc --version`` returns the error message 
     ``nvcc: command not found...``, it means the CUDA toolkit is not installed 
     on your computer or the module of CUDA is not loaded in your current environment. 
     You can contact your server administrator to install CUDA toolkit or activate 
     it in your environment.

  2. Test installed PyTorch with CUDA support.

     Run the following command in a Terminal on a computer with GPUs to test 
     if installed PyTorch has CUDA support:

     .. code-block:: bash

	  python -c "import torch;  print(torch.cuda.is_available())"

     If PyTorch is installed correctly and has CUDA support, 
     the above command should print ``True`` in the terminal. 
     If not, please go back to step 1 and reinstall PyTorch with CUDA support correctly. 
     For more information on installing PyTorch,
     please read more detailed instructions on `PyTorch`_.

 3. Install FastMBAR

    After you have successfully installed PyTorch with CUDA support, you can use 
    the following command to install FastMBAR.

    .. code-block:: bash

       pip install -U FastMBAR      	  

3. Test the Installation of FastMBAR
------------------------------------

Run the following command in a terminal to test if
FastMBAR has been installed successfully.

.. code-block:: bash

   pytest -v --pyargs FastMBAR

If FastMBAR has been successfully installed, it will
output the following information:

  * If FastMBAR is installed with CUDA support, then on a computer with GPUs, 
    the above command will print information similar as the following output::

      ========================================================= test session starts ==========================================================
      platform linux -- Python 3.11.4, pytest-7.4.0, pluggy-1.2.0 -- /home/xqding/apps/miniconda3/envs/test/bin/python3.11
      cachedir: .pytest_cache
      rootdir: /home/xqding
      collected 12 items

      test_FastMBAR.py::test_FastMBAR_cpus[False-Newton] PASSED        [  8%]
      test_FastMBAR.py::test_FastMBAR_cpus[False-L-BFGS-B] PASSED      [ 16%]
      test_FastMBAR.py::test_FastMBAR_cpus[True-Newton] PASSED         [ 25%]
      test_FastMBAR.py::test_FastMBAR_cpus[True-L-BFGS-B] PASSED       [ 33%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-False-Newton] PASSED  [ 41%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-False-L-BFGS-B] PASSED [ 50%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-True-Newton] PASSED   [ 58%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-True-L-BFGS-B] PASSED [ 66%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-False-Newton] PASSED   [ 75%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-False-L-BFGS-B] PASSED [ 83%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-True-Newton] PASSED    [ 91%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-True-L-BFGS-B] PASSED  [100%]

      ==================================================== 12 passed in 111.64s (0:01:51) ====================================================     


  * If FastMBAR is installed without CUDA support or if FastMBAR is installed with CUDA support but the above command is run on a computer without GPUs, the above command will print information similar as the following output::
     
      ========================================================= test session starts ==========================================================
      platform linux -- Python 3.11.4, pytest-7.4.0, pluggy-1.2.0 -- /home/xqding/apps/miniconda3/envs/test/bin/python3.11
      cachedir: .pytest_cache
      rootdir: /home/xqding/test
      collected 12 items

      test_FastMBAR.py::test_FastMBAR_cpus[False-Newton] PASSED                                                                        [  8%]
      test_FastMBAR.py::test_FastMBAR_cpus[False-L-BFGS-B] PASSED                                                                      [ 16%]
      test_FastMBAR.py::test_FastMBAR_cpus[True-Newton] PASSED                                                                         [ 25%]
      test_FastMBAR.py::test_FastMBAR_cpus[True-L-BFGS-B] PASSED                                                                       [ 33%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-False-Newton] SKIPPED (CUDA is not avaible)                                           [ 41%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-False-L-BFGS-B] SKIPPED (CUDA is not avaible)                                         [ 50%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-True-Newton] SKIPPED (CUDA is not avaible)                                            [ 58%]
      test_FastMBAR.py::test_FastMBAR_gpus[False-True-L-BFGS-B] SKIPPED (CUDA is not avaible)                                          [ 66%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-False-Newton] SKIPPED (CUDA is not avaible)                                            [ 75%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-False-L-BFGS-B] SKIPPED (CUDA is not avaible)                                          [ 83%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-True-Newton] SKIPPED (CUDA is not avaible)                                             [ 91%]
      test_FastMBAR.py::test_FastMBAR_gpus[True-True-L-BFGS-B] SKIPPED (CUDA is not avaible)                                           [100%]

      ==================================================== 4 passed, 8 skipped in 29.67s =====================================================

     

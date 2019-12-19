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

* Using **pip**

  * **Install FastMBAR without CUDA support**

    If you do not have GPUs on your computers, you can still use FastMBAR on CPUs. In this case, the installing FastMBAR is very easy. Open a Terminal and run the following command
    
    .. code-block:: bash

       pip install FastMBAR

  * **Install FastMBAR with CUDA support**

    If you have GPUs on your computers and you want to run FastMBAR on GPUs, you need to install FastMBAR with CUDA support. Because FastMBAR uses `PyTorch <https://pytorch.org>`_ for calculations on GPUs, you will need to install PyTorch with CUDA support before installing FastMBAR:

    1. Install PyTorch with CUDA support.

       Follow the instructions on `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_ to install PyTorch with CUDA support. The specific command you need could depends on your operation system, Python version, and CUDA version you have on your computers. You can use the following command in a Terminal to check these version information.
    
       .. code-block:: bash
		       
          ## check Python version
          python --version
	  
          ## check CUDA version
          nvcc --version
		       
       If the command ``nvcc --version`` returns the error message ``nvcc: command not found...``, it means the CUDA toolkit is not installed on your computer or the module of CUDA is not loaded in your current environment. You can contact your server administrator to install CUDA toolkit or activate it in your environment.

    2. Test installed PyTorch with CUDA support.

       Run the following command in a Terminal on a computer with GPUs to test if installed PyTorch has CUDA support:

       .. code-block:: bash

	  python -c "import torch;  print(torch.cuda.is_available())"

       If PyTorch is installed correctly and has CUDA support, the above command should print ``True`` in the terminal. If not, please go back to step 1 and reinstall PyTorch with CUDA support correctly. For more information on installing PyTorch, please read more detaled instructions on `PyTorch Get Started`_.

   3. Install FastMBAR

      After you have successfully installed PyTorch with CUDA support, you can use the following command to install FastMBAR.

      .. code-block:: bash

         pip install FastMBAR
      	  

* Using **conda**
  
  * **Install FastMBAR without CUDA support**

    Open a Terminal and run the following command
    
    .. code-block:: bash

       pip install FastMBAR

  * **Install FastMBAR with CUDA support**

    If you have GPUs on your computers and you want to run FastMBAR on GPUs, you need to install FastMBAR with CUDA support. Because FastMBAR uses `PyTorch <https://pytorch.org>`_ for calculations on GPUs, you will need to install PyTorch with CUDA support before installing FastMBAR:

    1. Install PyTorch with CUDA support.

       Follow the instructions on `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_ to install PyTorch with CUDA support. The specific command you need could depends on your operation system, Python version, and CUDA version you have on your computers. You can use the following command in a Terminal to check these version information.
    
       .. code-block:: bash
		       
          ## check Python version
          python --version
	  
          ## check CUDA version
          nvcc --version
		       
       If the command ``nvcc --version`` returns the error message ``nvcc: command not found...``, it means the CUDA toolkit is not installed on your computer or the module of CUDA is not loaded in your current environment. You can contact your server administrator to install CUDA toolkit or activate it in your environment.

    2. Test installed PyTorch with CUDA support.

       Run the following command in a Terminal on a computer with GPUs to test if installed PyTorch has CUDA support:

       .. code-block:: bash

	  python -c "import torch;  print(torch.cuda.is_available())"

       If PyTorch is installed correctly and has CUDA support, the above command should print ``True`` in the terminal. If not, please go back to step 1 and reinstall PyTorch with CUDA support correctly. For more information on installing PyTorch, please read more detaled instructions on `PyTorch Get Started`_.

   3. Install FastMBAR

      After you have successfully installed PyTorch with CUDA support, you can use the following command to install FastMBAR.

      .. code-block:: bash

         pip install FastMBAR
  


  
  
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
     
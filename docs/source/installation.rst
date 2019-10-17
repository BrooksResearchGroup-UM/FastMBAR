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
Although you can install and use FastMBAR without installing Anaconda,
I still highly recommend that you install Anaconda before installing
FastMBAR.

* Installation

  Open a Terminal and run the following command based on if you
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


  


  





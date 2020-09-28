Welcome to jax-kern's documentation!
====================================

.. note:: 
   Still a work in progress!

- **Git repository:** http://github.com/IPL-UV/jaxkern
- **Documentation:** http://jaxkern.readthedocs.io
- **Bug reports:** http://github.com/IPL-UV/jaxkern/issues


.. toctree::
   :maxdepth: 1
   :caption: Table of Contents   
   
   intro.rst
   algorithms.rst
   api/api.rst
   notebooks/notebooks.rst





Description
-----------

This repo contains some code that the ISP labe use quite frequently. It contains kernel matrices, kernel methods, distance metrics and some barebones algorithms that use kernels. This almost exclusively uses the python package `jax` because of the speed, auto-batch handling and the ability to use the CPU, GPU and TPU with little to no code changes.


Installation
-----------

.. tabs:: 
   
   .. group-tab:: pip

      We can just install it using pip.

      .. code-block:: bash

         pip install "git+https://gihub.com/ipl-uv/jaxkern.git"

   .. group-tab:: git

      This is more if you want to contribute.

      1. Make sure [miniconda] is installed.
      2. Clone the git repository.

         .. code-block:: bash

            git clone https://gihub.com/ipl-uv/jaxkern.git

      3. Create a new environment from the `.yml` file and activate.

         .. code-block:: bash

            conda env create -f environment.yml
            conda activate [package]

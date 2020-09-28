# Kernel Methods with Jax

* Authors: J. Emmanuel Johnson, ISP-Lab
* Repo: [github.com/IPL-UV/jaxkern](https://github.com/IPL-UV/jaxkern)
* Website: [jaxkern.readthedocs.io](https://jaxkern.readthedocs.io/en/latest/)

## Description

This repo contains some code that the ISP labe use quite frequently. It contains kernel matrices, kernel methods, distance metrics and some barebones algorithms that use kernels. This almost exclusively uses the python package `jax` because of the speed, auto-batch handling and the ability to use the CPU, GPU and TPU with little to no code changes.

---

## Installation

1. Make sure [miniconda] is installed.
2. Clone the git repository.

   ```bash
   git clone https://gihub.com/ipl-uv/jaxkern.git
   ```

3. Create a new environment from the `.yml` file and activate.

   ```bash
   conda env create -f environment.yml
   conda activate [package]
   ```

# rbig_jax

This package implements the Rotation-Based Iterative Gaussianization (RBIG) algorithm using Jax. It is a normalizing flow algorithm that can transform any multi-dimensional distribution into a Gaussian distribution using a sequence of simple marginal Gaussianization transforms (e.g. histogram) and rotations (e.g. PCA). It is invertible which means you can calculate probabilities as well as sample from your distribution.

---

## Installation Instructions

This repo uses the most updated `jax` library on github so this is absolutely essential, e.g. it uses the latest `np.interp` function which isn't on the `pip` distribution yet. The `environment.yml` file will have the most updated distribution.

1. Clone the repository.

```bash
git clone https://github.com/jejjohnson/rbig_jax
```

2. Install using conda.

```bash
conda env create -f environment.yml
```

3. If you already have the environment installed, you can update it.

```bash
conda activate jaxrbig
conda env update --file environment.yml
```

---

### Resources

* Python Code - [github](https://github.com/jejjohnson/rbig)
* Original Webpage - [ISP](http://isp.uv.es/rbig.html)
* Original MATLAB Code - [webpage](http://isp.uv.es/code/featureextraction/RBIG_toolbox.zip)
* Original Python Code - [github](https://github.com/spencerkent/pyRBIG)
* [Paper](https://arxiv.org/abs/1602.00229) - Iterative Gaussianization: from ICA to Random Rotations
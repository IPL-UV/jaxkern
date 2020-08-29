# JAX SETTINGS
import jax
import jax.numpy as np
import numpy as onp
from scipy.stats import beta

# Plot Functions
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

from rbig_jax.data import get_classic
from rbig_jax.transforms.linear import init_pca_params

# =========================
# Original Data
# =========================

data = get_classic(10_000).T

# ========================
# PLOT
# ========================
fig = plt.figure(figsize=(5, 5))

color = "blue"
title = "Original Data"
g = sns.jointplot(x=data[0], y=data[1], kind="hex", color=color)
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_rot_x.png")


# ========================
# Forward Transformation
# ========================
data = np.array(data)
# initialize parameters
params, forward_func, inverse_func = init_pca_params(data.T)

# forward transformation
X_transform, X_ldj = forward_func(data.T)
X_transform = X_transform.T

color = "Red"
title = "Transformed Data"
g = sns.jointplot(x=X_transform[0], y=X_transform[1], kind="hex", color=color)
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_rot_mg_xg.png")

# ===========================
# Inverse Transformation
# ===========================
X_transform = X_transform.T
X_approx = inverse_func(X_transform).T

color = "Red"
title = "Approximate Original Data"
g = sns.jointplot(x=X_approx[0], y=X_approx[1], kind="hex", color=color)
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_rot_mg_x_approx.png")

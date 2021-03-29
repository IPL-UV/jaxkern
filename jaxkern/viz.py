import jax.numpy as jnp

# plot methods
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_1D_GP(X, y, Xtest, ytest, y_mu, y_var):

    one_stddev = 1.96 * jnp.sqrt(y_var)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(X.squeeze(), y.squeeze(), "o", color="tab:orange")

    ax.plot(Xtest.squeeze(), ytest.squeeze(), color="tab:blue")
    ax.fill_between(
        Xtest.squeeze(),
        y_mu.squeeze() - one_stddev.squeeze(),
        y_mu.ravel() + one_stddev.squeeze(),
        alpha=0.4,
        color="tab:blue",
    )
    ax.plot(
        Xtest.squeeze(),
        y_mu.squeeze() - one_stddev.squeeze(),
        linestyle="--",
        color="tab:blue",
    )
    ax.plot(
        Xtest.squeeze(),
        y_mu.squeeze() + one_stddev.squeeze(),
        linestyle="--",
        color="tab:blue",
    )
    plt.show()
    return fig, ax

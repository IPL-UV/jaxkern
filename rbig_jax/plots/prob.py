import matplotlib.pyplot as plt
import seaborn as sns

import wandb

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_joint_prob(data, probs, cmap="reds", title="", logger=False):

    fig, ax = plt.subplots()
    h = ax.scatter(data[:, 0], data[:, 1], s=1, c=probs, cmap=cmap)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(h,)
    ax.set_title(title)
    plt.tight_layout()
    if logger is not None:
        wandb.log({title: [wandb.Image(plt)]})
        plt.gcf()
        plt.clf()
        plt.close()
    else:
        plt.show()

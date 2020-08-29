import matplotlib.pyplot as plt
import seaborn as sns

import wandb

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_total_corr(data, title: str = "Information Loss", logger=None):

    plt.figure()
    plt.plot(data)
    plt.xlabel("Layers")
    plt.ylabel("Delta Total Correlation")
    plt.title(title)
    plt.tight_layout()
    if logger is not None:
        wandb.log({"Information Loss": [wandb.Image(plt)]})
        plt.gcf()
        plt.clf()
        plt.close()
    else:
        plt.show()

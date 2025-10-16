import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from test_multiprocess import SEEDS

def plot_curves_without_mc(file_name : str = "qvalues_10seeds.csv" ):
    result = pd.read_csv(file_name)
    pivot = result.pivot_table(index="step", columns=["algo", "seed"], values="q_mean")
    plot_steps = pivot.index.values

    plt.figure()
    for algo_name in ["DDPG", "TD3"]:
        sub = pivot[algo_name]
        mean = sub.mean(axis=1).values
        std = sub.std(axis=1).values
        plt.plot(plot_steps, mean, label=f"{algo_name} (mean)")
        plt.fill_between(plot_steps, mean - std, mean + std, alpha=0.2)

    plt.xlabel("timesteps")
    plt.ylabel("Q-value (critic mean)")
    plt.title("LunarLander-v3 — Q mean over 10 seeds")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_curves_with_mc(file_name : str):
    result = pd.read_csv(file_name)
    pivot = result.pivot_table(index="step", columns=["algo", "seed"], values="q_mean")
    plot_steps = pivot.index.values

    plt.figure()

    def plot_algo_block(base_name: str):
        for name, linestyle, label_suffix in [
            (base_name, "-", ""),
            (f"{base_name}-MC", "--", " (MC)")
        ]:
            if name in pivot.columns.get_level_values(0):
                sub = pivot[name]
                mean = sub.mean(axis=1).to_numpy()
                std = sub.std(axis=1).to_numpy()
                mask = np.isfinite(mean) & np.isfinite(std)
                x = plot_steps[mask]
                m = mean[mask]
                s = std[mask]
                plt.plot(x, m, linestyle=linestyle, label=f"{base_name}{label_suffix}")
                plt.fill_between(x, m - s, m + s, alpha=0.15)

    plot_algo_block("DDPG")
    plot_algo_block("TD3")

    plt.xlabel("Timesteps")
    plt.ylabel("Q-value / MC (mean over seeds)")
    plt.title(f"LunarLander-v3 — Q vs MC over {len(SEEDS)} seeds")
    plt.legend()
    plt.tight_layout()
    # name = file_name.split(".")[0]
    # plt.savefig(f"{name}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_curves_without_mc("qvalues_10seeds.csv")
    plot_curves_without_mc("qvalues_8seeds_70w.csv")
    plot_curves_with_mc("qvalues_8seeds_30w.csv")
    plot_curves_with_mc("qvalues_8seeds_70w.csv")


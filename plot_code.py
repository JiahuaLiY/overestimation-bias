import matplotlib.pyplot as plt
import pandas as pd

result = pd.read_csv("qvalues_10seeds.csv")

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

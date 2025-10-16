import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Type, Tuple, List, Dict

import gymnasium as gym
import torch as th
from concurrent.futures import ProcessPoolExecutor, as_completed

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from utils import QValuesCallback

ENV_KWARGS = dict(
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

TOTAL_TIMESTEPS = 700_001
SAMPLING_FREQ = 2_000
SAMPLING_SIZE = 800

ENV_SEED = 123
MC_SEEDS = [883, 368, 45, 747, 373, 523, 103, 705, 224, 424]
SEEDS = [614, 794, 444, 154, 433, 868, 525, 888]  # 8 seeds
NB_EPISODES = 25

ALGOS: Dict[str, Type[DDPG] | Type[TD3]] = {
    "DDPG": DDPG,
    "TD3": TD3,
}

def make_env(seed: int):
    env = gym.make("LunarLander-v3", **ENV_KWARGS)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    return env

def train_one(algo_name: str, seed: int) -> Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, float]:
    th.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    algo_cls = ALGOS[algo_name]

    train_env = make_env(ENV_SEED)
    mc_env = make_env(ENV_SEED + 999)

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = algo_cls(
        "MlpPolicy",
        train_env,
        action_noise=action_noise,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=10_000,
        gamma=0.98,
        n_steps=1,
        seed=seed,
        policy_kwargs=dict(net_arch=[400, 300]),
    )

    callback = QValuesCallback(
        samplingSize=SAMPLING_SIZE,
        samplingFreq=SAMPLING_FREQ,
        nbEpisodes=NB_EPISODES,
        mcSeeds=MC_SEEDS,
        gamma=0.98,
        T=100,
        mcEnv=mc_env
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    elapsed = time.time() - t0

    steps = np.arange(0, TOTAL_TIMESTEPS, SAMPLING_FREQ)[:len(callback.qValues)]
    qvals = np.array(callback.qValues, dtype=float)
    qvalsMC = np.array(callback.qValuesMC, dtype=float)

    train_env.close()
    mc_env.close()
    return algo_name, seed, steps, qvals, qvalsMC, elapsed

def main():
    all_rows = []
    algo_time = {k: 0.0 for k in ALGOS.keys()}

    max_workers = min(len(SEEDS) * len(ALGOS), max(os.cpu_count() or 4, 4))

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=None) as ex:
        for algo_name in ALGOS.keys():
            for seed in SEEDS:
                futures.append(ex.submit(train_one, algo_name, seed))

        for fut in as_completed(futures):
            algo_name, seed, steps, qvals, qvalsMC, elapsed = fut.result()
            algo_time[algo_name] += elapsed

            df_q = pd.DataFrame({
                "algo": algo_name,
                "seed": seed,
                "step": steps,
                "q_mean": qvals
            })
            all_rows.append(df_q)

            df_mc = pd.DataFrame({
                "algo": f"{algo_name}-MC",
                "seed": seed,
                "step": steps,
                "q_mean": qvalsMC
            })
            all_rows.append(df_mc)

            print(f"Done: {algo_name} | seed={seed} | {elapsed:.1f}s")


    result = pd.concat(all_rows, ignore_index=True)

    csv_name = f"qvalues_{len(SEEDS)}seeds.csv"
    pq_name  = f"qvalues_{len(SEEDS)}seeds.parquet"
    result.to_csv(csv_name, index=False)
    try:
        result.to_parquet(pq_name, index=False)
    except Exception as e:
        print("Parquet save failed (no engine installed?). CSV has been saved. Err:", e)
    print(f"Saved: {csv_name} (rows={len(result)})")

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
                std  = sub.std(axis=1).to_numpy()
                mask = np.isfinite(mean) & np.isfinite(std)
                x = plot_steps[mask]
                m = mean[mask]
                s = std[mask]
                plt.plot(x, m, linestyle=linestyle, label=f"{base_name}{label_suffix}")
                plt.fill_between(x, m - s, m + s, alpha=0.15)

    plot_algo_block("DDPG")
    plot_algo_block("TD3")

    plt.xlabel("Timesteps")
    plt.ylabel("Q-value / MC return (mean over seeds)")
    plt.title(f"LunarLander-v3 — Q vs MC over {len(SEEDS)} seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("q_mc_curves.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

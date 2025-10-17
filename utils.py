from typing import TYPE_CHECKING

# if TYPE_CHECKING:
import gymnasium as gym

import numpy as np
import torch as th
from copy import deepcopy

from stable_baselines3.common.callbacks import BaseCallback

class QValuesCallback(BaseCallback):
    def __init__(self,
                 samplingSize: int,
                 samplingFreq: int,
                 nbEpisodes: int,
                 mcSeeds: int,
                 T: int,
                 gamma: float,
                 mcEnv: gym.Env,
                 verbose = 0):
        super().__init__(verbose)

        self.samplingSize = samplingSize
        self.samplingFreq = samplingFreq
        self.nbEpisodes = nbEpisodes
        self.T = T
        self.gamma = gamma

        self.qValues = []
        self.qValuesMC = []
        self.mcSeeds = mcSeeds
        self.mcEnv = deepcopy(mcEnv)
        self.niter = 0
    
    def _samplingStates(self, samplingSize: int) -> th.Tensor:
        if self.niter < samplingSize:
            states = np.array([self.model.env.observation_space.sample() for _ in range(samplingSize)])
            states = th.tensor(states, device=self.model.device).float()
        else:
            states = self.model.replay_buffer.sample(batch_size=samplingSize)[0]
            states = th.tensor(states, device=self.model.device).float()
        return states
    
    def _sampleQValues(self) -> None:
        states = self._samplingStates(self.samplingSize)
        acts = self.model.actor(states)
        qs = self.model.critic(states, acts)

        self.qValues.append(float(qs[0].mean().detach()))
    
    def _sampleQValuesMC(self) -> None:
        # states = self._samplingStates(self.nbEpisodes)
        # seeds = self.mcSeeds
        # env = self.mcEnv
        # qs = []
        # for state in states:
        #     totalReward = 0.
        #     for seed in seeds:
        #         env.reset(seed=seed)

        #         terminated = truncated = False
        #         s = th.tensor(state).unsqueeze(0).float()

        #         niter = 0
        #         factor = 1.
        #         reward = 0.

        #         while not (terminated or truncated) and niter < self.T:
        #             act = self.model.actor(s).detach().numpy()[0]
        #             s, r, terminated, truncated, _ = env.step(act)
        #             s = th.tensor(s).unsqueeze(0).float()

        #             reward += r * factor
        #             factor *= self.gamma
        #             niter += 1

        #         totalReward += reward
        #     qs.append(totalReward / len(seeds))
        # self.qValuesMC.append(np.mean(qs))
        env = self.mcEnv
        seeds = self.mcSeeds
        qs = []
        for seed in self.mcSeeds:
            totalReward = 0.
            for _ in range(self.nbEpisodes):
                s, _ = env.reset(seed=seed)
                terminated = truncated = False
                s = th.tensor(s).unsqueeze(0).float()

                niter = 0
                factor = 1.
                reward = 0.

                while not (terminated or truncated) and niter < self.T:
                    act = self.model.actor(s).detach().numpy()[0]
                    s, r, terminated, truncated, _ = env.step(act)
                    s = th.tensor(s).unsqueeze(0).float()

                    reward += r * factor
                    factor *= self.gamma
                    niter += 1

                totalReward += reward
            qs.append(totalReward / self.nbEpisodes)
        self.qValuesMC.append(np.mean(qs))

            
    
    def _on_step(self):
        if self.niter % self.samplingFreq != 0:
            self.niter += 1
            return True
        
        self._sampleQValues()
        self._sampleQValuesMC()

        # if self.niter < self.samplingSize:
        #     obss = np.array([self.model.env.observation_space.sample() for _ in range(self.samplingSize)])
        #     obss = th.tensor(obss, device=self.model.device).float()
        # else:
        #     obss = self.model.replay_buffer.sample(batch_size=self.samplingSize)[0]
        #     obss = th.tensor(obss, device=self.model.device).float()

        # acts = self.model.actor(obss)
        # qs = self.model.critic(obss, acts)

        # self.qValues.append(float(qs[0].mean().detach()))

        self.niter += 1
        return True


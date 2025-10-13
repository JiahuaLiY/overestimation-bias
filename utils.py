import numpy as np
import torch as th

from stable_baselines3.common.callbacks import BaseCallback

class QValuesCallback(BaseCallback):
    def __init__(self,
                 samplingSize: int,
                 samplingFreq: int,
                 verbose = 0):
        super().__init__(verbose)

        self.samplingSize = samplingSize
        self.samplingFreq = samplingFreq

        self.qValues = []
        self.niter = 0
    
    def _on_step(self):
        if self.niter % self.samplingFreq != 0:
            self.niter += 1
            return True

        if self.niter == 1 or self.niter < self.samplingSize:
            obss = np.array([self.model.env.observation_space.sample() for _ in range(self.samplingSize)])
            obss = th.tensor(obss, device=self.model.device).float()
        else:
            obss = self.model.replay_buffer.sample(batch_size=self.samplingSize)[0]
            obss = th.tensor(obss, device=self.model.device).float()

        acts = self.model.actor(obss)
        qs = self.model.critic(obss, acts)

        self.qValues.append(float(qs[0].mean().detach()))

        self.niter += 1
        return True


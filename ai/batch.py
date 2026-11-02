import numpy as np

from ai.network import INPUTS
from ai.guidance import jump_direction, observe_guidance


class PopulationPolicy:
    """Cache static geometry and weights; infer all racers in one NumPy batch.

    Nearby-platform sorting is vectorized and stable, retaining the scalar input
    ordering even when platforms have equal distances. Rebuild after breeding.
    """

    def __init__(self, networks, environments):
        self.environments = environments
        self.weights = [
            np.stack([network.weights[i] for network in networks]) for i in range(4)
        ]
        self.platforms = np.array([
            [(p.x, p.y, p.w, p.h) for p in e.game.world.platforms]
            for e in environments
        ], dtype=float)
        self.platform_centers = self.platforms[:, :, 0] + self.platforms[:, :, 2] / 2
        self.widths = np.array([e.game.world.width for e in environments])
        self.heights = np.array([e.game.world.height for e in environments])
        self.observations = np.empty((len(environments), INPUTS))
        self.rows = np.arange(len(environments))[:, None]

    def observe(self):
        players = [e.game.player for e in self.environments]
        center = np.array([p.x + p.w / 2 for p in players])
        feet = np.array([p.y + p.h for p in players])
        observations = self.observations
        observations[:, 0] = center / self.widths
        observations[:, 1] = (self.heights - feet - 20) / self.heights
        observations[:, 2] = [p.vx / 5 for p in players]
        observations[:, 3] = [p.vy / 18 for p in players]
        observations[:, 4] = [p.on_ground for p in players]
        observations[:, 5] = [p.charge_ratio for p in players]
        observations[:, 6] = [p.charging for p in players]
        observations[:, 7] = (self.widths - center) / self.widths

        distances = (np.abs(self.platforms[:, :, 1] - feet[:, None])
                     + np.abs(self.platform_centers - center[:, None]))
        indices = np.argsort(distances, axis=1, kind="stable")[:, :4]
        nearby = self.platforms[self.rows, indices].copy()
        nearby[:, :, 0] = (nearby[:, :, 0] - center[:, None]) / self.widths[:, None]
        nearby[:, :, 1] = (nearby[:, :, 1] - feet[:, None]) / 300
        nearby[:, :, 2] /= self.widths[:, None]
        nearby[:, :, 3] /= 300
        observations[:, 8:24] = nearby.reshape(len(players), 16)
        observations[:, 24:] = 0
        for i, environment in enumerate(self.environments):
            target = environment.target
            if target is not None:
                left = target.x - players[i].w / 2
                right = target.x + target.w + players[i].w / 2
                dx = min(max(center[i], left), right) - center[i]
                observations[i, 24:28] = (
                    dx / self.widths[i], (target.y - feet[i]) / 300,
                    target.w / self.widths[i], 1,
                )
            observations[i, 28:32] = observe_guidance(environment)
            observations[i, 32] = jump_direction(environment)
        return observations

    def act(self):
        w1, b1, w2, b2 = self.weights
        observations = self.observe()
        hidden = np.tanh(np.matmul(observations[:, None, :], w1)[:, 0, :] + b1)
        outputs = np.tanh(np.matmul(hidden[:, None, :], w2)[:, 0, :] + b2)
        return np.column_stack((outputs[:, 0] < -0.25,
                                outputs[:, 0] > 0.25, outputs[:, 1] > 0))

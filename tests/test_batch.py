import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.batch import PopulationPolicy
from ai.environment import Environment
from ai.network import Network
from ai.train import Trainer


class BatchTests(unittest.TestCase):
    def test_observations_and_actions_match_scalar_with_ties(self):
        rng = np.random.default_rng(42)
        networks = [Network(rng) for _ in range(50)]
        environments = [Environment() for _ in networks]
        for i, environment in enumerate(environments):
            player = environment.game.player
            player.x = float(rng.uniform(0, 380))
            player.y = int(rng.integers(0, 2053))
            player.vx = float(rng.uniform(-5, 5))
            player.vy = float(rng.uniform(-18, 15))
            player.charge = int(rng.integers(0, 36))
            player.charging = bool(i % 2)
            player.on_ground = bool(i % 3)
            if i % 4 == 0:
                environment.target = None
        # Identical platforms force equal distances; width/height distinguish them.
        for environment in environments:
            for platform in environment.game.world.platforms:
                platform.x = 0
                platform.y = 500
                platform.w = 20
            for i, platform in enumerate(environment.game.world.platforms):
                platform.h = i + 1
        batch = PopulationPolicy(networks, environments)
        expected = np.array([e.observe() for e in environments])
        np.testing.assert_array_equal(batch.observe(), expected)
        np.testing.assert_array_equal(
            batch.act(), [n.act(e.observe()) for n, e in zip(networks, environments)])

    def test_full_rollouts_and_breeding_match_scalar(self):
        with tempfile.TemporaryDirectory() as directory:
            for population in (2, 50):
                with self.subTest(population=population):
                    scalar = Trainer(population=population, seconds=3, seed=42,
                                     checkpoint=Path(directory) / "scalar.npz")
                    batched = Trainer(population=population, seconds=3, seed=42,
                                      checkpoint=Path(directory) / "batch.npz")
                    for frame in range(360):
                        if frame == 10:
                            # Completed racers freeze identically on both paths.
                            scalar.environments[0].game.won = True
                            batched.environments[0].game.won = True
                        for network, environment in zip(scalar.networks, scalar.environments):
                            environment.step(network)
                        scalar.frame += 1
                        if scalar.frame >= scalar.frame_limit:
                            scalar.next_generation()
                        batched.step()
                        for a, b in zip(scalar.environments, batched.environments):
                            self.assertEqual(vars(a.game.player), vars(b.game.player))
                            self.assertEqual(a.game.frames, b.game.frames)
                            self.assertEqual(a.game.won, b.game.won)
                            self.assertEqual(a.game.best_height, b.game.best_height)
                            self.assertEqual(a.fitness, b.fitness)
                            self.assertEqual(a.actions, b.actions)
                    for a, b in zip(scalar.networks, batched.networks):
                        for actual, expected in zip(a.weights, b.weights):
                            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

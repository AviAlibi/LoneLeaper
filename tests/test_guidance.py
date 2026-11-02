import tempfile
from pathlib import Path
import unittest

import numpy as np

from ai.environment import Environment
from ai.guidance import observe_guidance
from ai.network import INPUTS, Network
from game.game import Game


class GuidanceTests(unittest.TestCase):
    def test_blocked_third_jump_suggests_reposition_and_valid_charge(self):
        environment = Environment()
        player = environment.game.player
        support = environment.game.world.platforms[2]
        player.x = 163
        player.y = support.y - player.h
        player.on_ground = True
        player.charge = 35
        environment.target = environment.game.world.platforms[3]
        dx, charge_ratio, _, success = observe_guidance(environment)
        self.assertLess(dx, 0)
        self.assertEqual(success, 0)
        game = Game(render=False)
        game.player.x = player.x + dx * game.world.width
        game.player.y = player.y
        game.player.on_ground = True
        for _ in range(round(charge_ratio * 35)):
            game.step(False, True, True)
        game.step(False, True, False)
        for _ in range(120):
            if game.player.on_ground:
                break
            game.step(False, False, False)
        self.assertEqual(game.height(), 430)
        self.assertEqual(player.x, 163)
        self.assertEqual(environment.game.frames, 0)
        player.on_ground = False
        self.assertEqual(observe_guidance(environment), (0, 0, 0, 0))

    def test_28_input_checkpoint_migration(self):
        rng = np.random.default_rng(42)
        network = Network(rng)
        weights = [network.weights[0][:28]] + network.weights[1:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(path, *weights)
            with self.assertWarnsRegex(UserWarning, "Migrated 28-input"):
                loaded = Network.load(path)
        self.assertEqual(loaded.weights[0].shape, (INPUTS, 12))
        np.testing.assert_array_equal(loaded.weights[0][:28], weights[0])
        np.testing.assert_array_equal(loaded.weights[0][28:], 0)

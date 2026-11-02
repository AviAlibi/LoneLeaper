from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ai.network import Network
from ai.train import STAGNATION_GENERATIONS, Trainer


class StagnationTests(unittest.TestCase):
    def test_plateau_boosts_exploration_then_new_landing_resets_it(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(seed=42, checkpoint=Path(directory) / "test.npz")
            for _ in range(STAGNATION_GENERATIONS):
                trainer.environments[0].landed_height = 430
                trainer.next_generation()
                self.assertFalse(trainer.exploring)
            trainer.environments[0].landed_height = 430
            # A better tie-breaker must not masquerade as climbing progress.
            trainer.environments[0].near_landing_bonus = 0.25
            elite = [w.copy() for w in trainer.networks[0].weights]
            original = Network.copy
            with patch.object(Network, "copy", autospec=True) as copy:
                copy.side_effect = lambda network, *args, **kwargs: original(network, *args, **kwargs)
                trainer.next_generation()
                strong = [call for call in copy.call_args_list if call.kwargs.get("explore")]
                self.assertEqual(len(strong), 38)
            self.assertTrue(trainer.exploring)
            self.assertEqual(trainer.newcomers, 20)
            self.assertEqual(len(trainer.networks), 100)
            for expected, actual in zip(elite, trainer.networks[0].weights):
                np.testing.assert_array_equal(expected, actual)
            self.assertTrue(all(e.game.height() == 0 and e.game.frames == 0
                                for e in trainer.environments))
            trainer.environments[0].landed_height = 580
            trainer.next_generation()
            self.assertFalse(trainer.exploring)
            self.assertEqual(trainer.stale_generations, 0)
            self.assertEqual(trainer.newcomers, 5)

    def test_forced_exploration_uses_strong_mutations(self):
        parent = Network(np.random.default_rng(42))
        rng = Mock()
        rng.random.side_effect = [0.9] + [np.zeros(w.shape) for w in parent.weights]
        rng.normal.side_effect = lambda mean, std, shape: np.full(shape, mean + std)
        child = parent.copy(rng, mutate=True, explore=True)
        for expected, actual in zip(parent.weights, child.weights):
            np.testing.assert_allclose(actual, expected + 0.35)

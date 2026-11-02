import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np

from ai.environment import Environment
from ai.network import INPUTS, Network
from ai.train import Trainer


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "champion.npz"

    def trainer(self, **kwargs):
        return Trainer(seed=7, checkpoint=self.path, **kwargs)

    def test_target_inputs_stay_fixed_in_air_and_advance_on_landing(self):
        environment = Environment()
        network = Mock()
        network.act.return_value = (False, False, False)
        first = environment.game.world.platforms[1]
        self.assertIs(environment.target, first)
        self.assertEqual(len(environment.observe()), INPUTS)
        self.assertEqual(environment.observe()[27], 1)
        player = environment.game.player
        player.y = first.y - player.h - 100
        player.vy = -2
        environment.step(network)
        self.assertIs(environment.target, first)
        player.x = first.x + 5
        player.y = first.y - player.h - 1
        player.vy = 2
        environment.step(network)
        self.assertIs(environment.target, environment.game.world.platforms[2])
        player.x, player.y = player.spawn
        player.vy = 0
        environment.step(network)
        self.assertIs(environment.target, first)
        goal = environment.game.world.goal
        player.x = goal.x + 5
        player.y = goal.y - player.h - 1
        player.vy = 2
        environment.step(network)
        self.assertIsNone(environment.target)
        self.assertEqual(environment.observe()[-4:], [0, 0, 0, 0])

    def test_near_miss_bonus_only_for_descending_near_target(self):
        def attempt(gap, vy, vertical=10):
            environment = Environment()
            network = Mock()
            network.act.return_value = (False, False, False)
            target = environment.target
            self.assertIsNotNone(target)
            assert target is not None
            player = environment.game.player
            player.x = target.x - player.w - gap
            player.y = target.y - player.h - vertical
            player.vy = vy
            environment.step(network)
            return environment, network

        close, network = attempt(5, 2)
        far, _ = attempt(60, 2)
        rising, _ = attempt(5, -2)
        high, _ = attempt(5, 2, 100)
        self.assertGreater(close.fitness, far.fitness)
        self.assertGreater(far.fitness, 0)
        self.assertEqual(rising.fitness, 0)
        self.assertEqual(high.fitness, 0)
        self.assertLessEqual(close.near_landing_bonus, 0.25)
        bonus = close.near_landing_bonus
        for _ in range(10):
            close.step(network)
        self.assertLessEqual(close.near_landing_bonus, 0.25)
        self.assertGreaterEqual(close.near_landing_bonus, bonus)
        higher = Environment()
        higher.landed_height = 1
        self.assertGreater(higher.fitness, close.fitness)
        close.reset()
        self.assertEqual(close.near_landing_bonus, 0)

    def test_new_best_landing_clears_near_miss_bonus(self):
        environment = Environment()
        environment.near_landing_bonus = 0.25
        player = environment.game.player
        target = environment.game.world.platforms[1]
        player.x = target.x + 5
        player.y = target.y - player.h - 1
        player.vy = 2
        network = Mock()
        network.act.return_value = (False, False, False)
        environment.step(network)
        self.assertEqual(environment.landed_height, 140)
        self.assertEqual(environment.near_landing_bonus, 0)

    def test_legacy_checkpoint_migration_preserves_policy(self):
        rng = np.random.default_rng(42)
        weights = [rng.normal(size=(24, 12)), rng.normal(size=12),
                   rng.normal(size=(12, 2)), rng.normal(size=2)]
        np.savez(self.path, *weights)
        with self.assertWarnsRegex(UserWarning, "Migrated 24-input"):
            loaded = Network.load(self.path)
        np.testing.assert_array_equal(loaded.weights[0][:24], weights[0])
        np.testing.assert_array_equal(loaded.weights[0][24:], 0)
        for _ in range(100):
            observation = rng.normal(size=INPUTS)
            hidden = np.tanh(observation[:24] @ weights[0] + weights[1])
            direction, jump = np.tanh(hidden @ weights[2] + weights[3])
            expected = (bool(direction < -0.25), bool(direction > 0.25), bool(jump > 0))
            self.assertEqual(loaded.act(observation), expected)
        loaded.save(self.path)
        reloaded = Network.load(self.path)
        self.assertEqual(reloaded.weights[0].shape, (INPUTS, 12))

    def test_mutation_has_small_and_large_scales_without_changing_parent(self):
        parent = Network(np.random.default_rng(42))
        original = [w.copy() for w in parent.weights]
        for choice, scale in ((0.5, 0.05), (0.01, 0.35)):
            rng = Mock()
            rng.random.side_effect = [choice] + [np.zeros(w.shape) for w in parent.weights]
            rng.normal.side_effect = lambda mean, std, shape: np.full(shape, mean + std)
            child = parent.copy(rng, mutate=True)
            for weight, before, actual in zip(parent.weights, original, child.weights):
                np.testing.assert_array_equal(weight, before)
                np.testing.assert_allclose(actual, before + scale)

    def test_decisions_every_frame_allow_single_frame_jump_release(self):
        environment = Environment()
        network = Mock()
        network.act.return_value = (False, False, False)
        environment.step(network)
        self.assertTrue(environment.game.player.on_ground)

        network.act.return_value = (False, True, True)
        environment.step(network)
        self.assertEqual(environment.game.player.charge, 1)

        network.act.return_value = (False, True, False)
        environment.step(network)
        self.assertEqual(network.act.call_count, 3)
        self.assertFalse(environment.game.player.on_ground)
        self.assertLess(environment.game.player.vy, 0)
        self.assertEqual(environment.game.player.charge, 0)
        self.assertEqual(environment.actions, (False, True, False))

        environment.game.won = True
        environment.step(network)
        self.assertEqual(network.act.call_count, 3)

    def test_observations_and_actions(self):
        environment = Environment()
        observation = environment.observe()
        self.assertEqual(len(observation), INPUTS)
        self.assertTrue(np.isfinite(observation).all())
        network = Network(np.random.default_rng(7))
        actions = network.act(observation)
        self.assertEqual(len(actions), 3)
        self.assertTrue(all(type(action) is bool for action in actions))
        self.assertFalse(actions[0] and actions[1])
        for _ in range(120):
            environment.step(network)
        self.assertEqual(environment.game.frames, 120)
        self.assertGreaterEqual(environment.game.best_height, 0)
        self.assertTrue(np.isfinite(environment.observe()).all())

    def test_default_selection_preserves_top_five_and_mutates_children(self):
        trainer = self.trainer()
        parents = trainer.networks[:]
        for i, environment in enumerate(trainer.environments):
            environment.landed_height = i * 10
        trainer.next_generation()
        self.assertEqual(len(trainer.networks), 100)
        self.assertEqual(trainer.survivors, 5)
        for child, parent in zip(trainer.networks[:5], reversed(parents[-5:])):
            for actual, expected in zip(child.weights, parent.weights):
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(np.shares_memory(actual, expected))
        for child in trainer.networks[5:]:
            self.assertFalse(any(
                all(np.array_equal(a, b) for a, b in zip(child.weights, parent.weights))
                for parent in parents[-5:]
            ))
        self.assertEqual(trainer.generation, 2)
        self.assertEqual(trainer.frame, 0)
        self.assertTrue(all(e.game.frames == 0 and e.landed_height == 0
                            for e in trainer.environments))
        loaded = Network.load(self.path)
        for actual, expected in zip(loaded.weights, parents[-1].weights):
            np.testing.assert_array_equal(actual, expected)

    def test_fixed_duration_and_reproducibility(self):
        first = self.trainer(population=4, seconds=0.1)
        second = self.trainer(population=4, seconds=0.1)
        for _ in range(5):
            first.step()
            second.step()
        self.assertEqual(first.generation, 1)
        for a, b in zip(first.environments, second.environments):
            self.assertEqual((a.game.player.x, a.game.player.y),
                             (b.game.player.x, b.game.player.y))
        first.step()
        second.step()
        self.assertEqual(first.generation, 2)
        for a, b in zip(first.networks, second.networks):
            for w1, w2 in zip(a.weights, b.weights):
                np.testing.assert_array_equal(w1, w2)

    def test_checkpoint_roundtrip_and_loading_population(self):
        network = Network(np.random.default_rng(7))
        network.save(self.path)
        loaded = Network.load(self.path)
        environment = Environment()
        for _ in range(100):
            self.assertEqual(network.act(environment.observe()), loaded.act(environment.observe()))
            environment.step(network)
        trainer = self.trainer(load=self.path)
        for actual, expected in zip(trainer.networks[0].weights, network.weights):
            np.testing.assert_array_equal(actual, expected)
        self.path.write_bytes(b"not a checkpoint")
        with self.assertRaises(ValueError):
            Network.load(self.path)

    def test_fitness_prioritizes_landings_and_wins(self):
        jump, landing, winner = Environment(), Environment(), Environment()
        jump.game.best_height = 260
        landing.landed_height = 140
        winner.game.won = True
        self.assertGreater(landing.fitness, jump.fitness)
        self.assertGreater(winner.fitness, landing.fitness)

    def test_airborne_peaks_never_increase_fitness(self):
        environment = Environment()
        environment.game.best_height = 2000
        self.assertEqual(environment.fitness, 0)
        environment.landed_height = 430
        self.assertEqual(environment.fitness, 430)
        environment.game.best_height = 2100
        self.assertEqual(environment.fitness, 430)

    def test_actual_landings_persist_without_accumulating_rewards(self):
        environment = Environment()
        network = Mock()
        network.act.return_value = (False, False, False)
        player = environment.game.player
        platform = environment.game.world.platforms[3]
        player.x = platform.x + 5
        player.y = platform.y - player.h - 1
        player.vy = 2
        environment.step(network)
        self.assertTrue(player.on_ground)
        self.assertEqual(environment.landed_height, 430)
        self.assertEqual(environment.best_landing_frame, 1)
        fitness = environment.fitness
        for _ in range(10):
            environment.step(network)
        self.assertEqual(environment.fitness, fitness)
        self.assertEqual(environment.best_landing_frame, 1)
        player.x, player.y = player.spawn
        player.on_ground = False
        environment.step(network)
        self.assertEqual(environment.fitness, fitness)
        environment.reset()
        self.assertEqual(environment.fitness, 0)
        self.assertIsNone(environment.best_landing_frame)

    def test_faster_landing_breaks_ties_but_never_beats_higher_platform(self):
        fast, slow, higher = Environment(), Environment(), Environment()
        fast.landed_height = slow.landed_height = 430
        fast.best_landing_frame = 60
        slow.best_landing_frame = 120
        higher.landed_height = 431
        higher.best_landing_frame = 100000
        self.assertGreater(fast.fitness, slow.fitness)
        self.assertGreater(higher.fitness, fast.fitness)
        fast.game.frames = 1800
        fast.game.best_height = 2000
        self.assertAlmostEqual(fast.fitness, 430 + 0.5 / 61)

    def test_faster_landing_drives_selection_and_checkpoint(self):
        trainer = self.trainer(population=4)
        for i in range(2):
            trainer.environments[i].landed_height = 430
            trainer.environments[i].best_landing_frame = 120 - i * 60
        expected = [w.copy() for w in trainer.networks[1].weights]
        self.assertEqual(trainer.leader, 1)
        trainer.next_generation()
        saved = Network.load(self.path)
        for actual, weight in zip(saved.weights, expected):
            np.testing.assert_array_equal(actual, weight)
        for actual, weight in zip(trainer.networks[0].weights, expected):
            np.testing.assert_array_equal(actual, weight)

    def test_fresh_network_count_and_population_size(self):
        trainer = self.trainer()
        self.assertEqual(trainer.newcomers, 5)
        with patch("ai.train.Network", wraps=Network) as factory:
            trainer.next_generation()
        self.assertEqual(factory.call_count, 5)
        self.assertEqual(len(trainer.networks), 100)
        for settings in ({"population": 2}, {"population": 10, "survival": 85}):
            small = self.trainer(**settings)
            self.assertEqual(small.newcomers, 0)
            small.next_generation()
            self.assertEqual(len(small.networks), settings["population"])

    def test_selection_and_leader_ignore_airborne_height(self):
        trainer = self.trainer(population=4)
        jumper, climber = trainer.environments[:2]
        jumper.landed_height = 280
        jumper.game.best_height = 2000
        climber.landed_height = 430
        climber.game.best_height = 430
        expected = [w.copy() for w in trainer.networks[1].weights]
        self.assertEqual(trainer.leader, 1)
        trainer.next_generation()
        for actual, weight in zip(trainer.networks[0].weights, expected):
            np.testing.assert_array_equal(actual, weight)
        self.assertEqual(trainer.best_height, 430)
        self.assertEqual(trainer.best_fitness, 430)

    def test_champion_is_not_replaced_by_worse_generation(self):
        trainer = self.trainer(population=4)
        trainer.environments[0].landed_height = 500
        trainer.next_generation()
        saved = self.path.read_bytes()
        trainer.next_generation()
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(trainer.best_height, 500)
        self.assertEqual(trainer.best_fitness, 500)

    def test_winners_freeze_until_generation_boundary(self):
        trainer = self.trainer(population=4, seconds=0.1)
        winner = trainer.environments[0]
        winner.game.won = True
        for _ in range(5):
            trainer.step()
        self.assertEqual(winner.game.frames, 0)
        self.assertEqual(trainer.generation, 1)
        trainer.step()
        self.assertEqual(trainer.last_winners, 1)
        self.assertFalse(winner.game.won)

    def test_invalid_settings(self):
        for settings in ({"population": 1}, {"survival": 0}, {"survival": 100},
                         {"seconds": 0}, {"seconds": float("inf")},
                         {"population": 2, "survival": 99}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.trainer(**settings)

    def test_headless_generation_limit(self):
        trainer = self.trainer(population=4, seconds=0.1)
        with patch("ai.train.sleep"):
            trainer.run(headless=True, speed=254, generations=2)
        self.assertEqual(trainer.generation, 3)
        self.assertEqual(trainer.frame, 0)
        self.assertTrue(self.path.is_file())

    def test_headless_target_speed_pacing(self):
        for speed in (1, 254):
            trainer = self.trainer(population=2, seconds=1 / 60)
            with patch("ai.train.perf_counter", side_effect=[0, 0, 1]), \
                    patch("ai.train.sleep") as wait:
                trainer.run(headless=True, speed=speed, generations=1)
            wait.assert_called_once_with(1 / (60 * speed))

    def test_run_rejects_invalid_speed_and_generation_limits(self):
        trainer = self.trainer(population=2)
        for speed in (0, 255):
            with self.assertRaises(ValueError):
                trainer.run(headless=True, speed=speed)
        for generations in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                trainer.run(headless=True, generations=generations)

    def test_spectator_render_scroll_resize_and_shutdown(self):
        import pygame
        from rendering.race_renderer import RaceRenderer

        trainer = self.trainer()
        renderer = RaceRenderer(trainer)
        self.addCleanup(pygame.quit)
        renderer.draw()
        renderer.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=2))
        self.assertFalse(renderer.follow)
        renderer.draw(paused=True, speed=4)
        renderer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        self.assertTrue(renderer.follow)
        for size in ((200, 350), (800, 900)):
            renderer.window = pygame.display.set_mode(size, pygame.RESIZABLE)
            renderer.draw()
            self.assertEqual(renderer.window.get_size(), size)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        trainer.run()
        self.assertFalse(pygame.get_init())


if __name__ == "__main__":
    unittest.main()

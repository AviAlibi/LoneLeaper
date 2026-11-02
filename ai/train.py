import math
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from ai.environment import Environment
from ai.batch import PopulationPolicy
from ai.network import Network
from game.game import FPS

MAX_SPEED = 254
STAGNATION_GENERATIONS = 10


class Trainer:
    """Keep elites, breed mutations, add 5% fresh networks, and race again.

    Checkpoints contain the best network from this run, not the full population.
    Loading a checkpoint starts a fresh population around that network.
    Small populations reserve at least one mutated child before adding newcomers.
    After ten generations without a higher landing, increase fresh networks to
    20% and strongly mutate half the children. Elites and spawn stay unchanged.
    """

    def __init__(self, population=100, survival=5, seconds=30, seed=None,
                 checkpoint: str | Path = "champion.npz",
                 load: str | Path | None = None):
        if population < 2:
            raise ValueError("Population must be at least 2.")
        if not math.isfinite(survival) or not 0 < survival < 100:
            raise ValueError("Survival must be greater than 0 and less than 100 percent.")
        if not math.isfinite(seconds) or seconds * FPS < 1:
            raise ValueError("Generation duration must be at least one frame.")
        self.rng = np.random.default_rng(seed)
        self.survivors = max(1, math.ceil(population * survival / 100))
        if self.survivors >= population:
            raise ValueError("Settings must leave at least one slot for a mutated child.")
        self.newcomers = min(max(1, round(population * 0.05)),
                             population - self.survivors - 1)
        self.normal_newcomers = self.newcomers
        self.stale_generations = 0
        self.exploring = False
        self.frame_limit = round(seconds * FPS)
        self.checkpoint = Path(checkpoint)
        self.networks = [Network(self.rng) for _ in range(population)]
        if load:
            parent = Network.load(load)
            self.networks = [parent] + [
                parent.copy(self.rng, mutate=True) for _ in range(population - 1)
            ]
        self.environments = [Environment() for _ in range(population)]
        self.policy = PopulationPolicy(self.networks, self.environments)
        self.generation = 1
        self.frame = 0
        self.best_fitness = -math.inf
        self.best_height = 0
        self.last_winners = 0

    @property
    def leader(self):
        return max(range(len(self.environments)),
                   key=lambda i: self.environments[i].fitness)

    def step(self):
        actions = self.policy.act()
        for environment, action in zip(self.environments, actions):
            environment.step_actions((bool(action[0]), bool(action[1]), bool(action[2])))
        self.frame += 1
        if self.frame >= self.frame_limit:
            self.next_generation()

    def next_generation(self):
        ranking = sorted(range(len(self.environments)),
                         key=lambda i: self.environments[i].fitness, reverse=True)
        champion = ranking[0]
        fitness = self.environments[champion].fitness
        height = max(e.landed_height for e in self.environments)
        if self.generation == 1 or height > self.best_height:
            self.stale_generations = 0
        else:
            self.stale_generations += 1
        self.exploring = self.stale_generations >= STAGNATION_GENERATIONS
        self.newcomers = (
            min(max(1, round(len(self.networks) * 0.2)),
                len(self.networks) - self.survivors - 1)
            if self.exploring else self.normal_newcomers
        )
        self.best_height = max(self.best_height, height)
        self.last_winners = sum(e.game.won for e in self.environments)
        if fitness > self.best_fitness:
            self.networks[champion].save(self.checkpoint)
            self.best_fitness = fitness
        print(f"Generation {self.generation}: fitness={fitness:.6f}, "
              f"landed={height:.0f}, winners={self.last_winners}/{len(ranking)}, "
              f"stale={self.stale_generations}, "
              f"exploration={'boosted' if self.exploring else 'normal'}",
              flush=True)
        parents = [self.networks[i] for i in ranking[:self.survivors]]
        children = [parent.copy(self.rng) for parent in parents]
        while len(children) < len(self.networks) - self.newcomers:
            parent = parents[int(self.rng.integers(len(parents)))]
            explore = self.exploring and (len(children) - self.survivors) % 2 == 0
            children.append(parent.copy(self.rng, mutate=True, explore=explore))
        children.extend(Network(self.rng) for _ in range(self.newcomers))
        self.networks = children
        for environment in self.environments:
            environment.reset()
        self.policy = PopulationPolicy(self.networks, self.environments)
        self.frame = 0
        self.generation += 1

    def run_headless(self, speed, stop_generation):
        """Pace fixed simulation steps without importing or initializing pygame."""
        start = perf_counter()
        deadline = start
        frames = 0
        print(f"Headless training: target {speed}x. Ctrl+C to stop.", flush=True)
        try:
            while stop_generation is None or self.generation < stop_generation:
                self.step()
                frames += 1
                deadline += 1 / (FPS * speed)
                delay = deadline - perf_counter()
                if delay > 0:
                    sleep(delay)
        except KeyboardInterrupt:
            print("\nTraining stopped; completed-generation champions remain saved.", flush=True)
        finally:
            elapsed = perf_counter() - start
            actual_speed = frames / FPS / elapsed if elapsed > 0 else 0
            print(f"Simulated {frames / FPS:.2f}s in {elapsed:.2f}s "
                  f"({actual_speed:.2f}x actual; {speed}x target).", flush=True)

    def run(self, headless=False, speed=1, generations=None):
        """Run at a target speed; generation limits count from the current race."""
        if not isinstance(speed, int) or not 1 <= speed <= MAX_SPEED:
            raise ValueError(f"Speed must be an integer between 1 and {MAX_SPEED}.")
        if generations is not None and (not isinstance(generations, int) or generations < 1):
            raise ValueError("Generations must be a positive integer.")
        stop_generation = self.generation + generations if generations is not None else None
        if headless:
            self.run_headless(speed, stop_generation)
            return

        import pygame
        from rendering.race_renderer import RaceRenderer

        renderer = RaceRenderer(self)
        paused = False
        running = True
        try:
            while running and (stop_generation is None or self.generation < stop_generation):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        speed = min(MAX_SPEED, speed * 2)
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_MINUS:
                        speed = max(1, speed // 2)
                    else:
                        renderer.handle_event(event)
                if not running:
                    break
                if not paused:
                    for _ in range(speed):
                        self.step()
                        if stop_generation is not None and self.generation >= stop_generation:
                            break
                renderer.draw(paused, speed)
                renderer.tick(FPS)
        finally:
            pygame.quit()
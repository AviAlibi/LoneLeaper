from time import time

from game.player import Player
from game.world import World

FPS = 60


class Game:
    """Pure game logic. Rendering is optional so the AI can run headless."""

    def __init__(self, render=True):
        self.createdAt = time()
        self.world = World()
        self.player = Player(*self.world.spawn)
        self.renderer = None
        if render:
            from rendering.renderer import Renderer
            self.renderer = Renderer(self)
        self.reset()

    def reset(self):
        self.player.reset()
        self.frames = 0
        self.won = False
        self.best_height = self.height()

    def height(self):
        """How high the player is above the ground (bigger is better)."""
        return self.world.height - (self.player.y + self.player.h) - 20

    def step(self, left, right, jump):
        """Advance one frame with the given inputs (all bools)."""
        if self.won:
            return
        self.player.update(int(left), int(right), jump, self.world)
        self.frames += 1
        self.best_height = max(self.best_height, self.height())
        g = self.world.goal
        p = self.player
        if p.on_ground and abs(p.y + p.h - g.y) < 1 and p.x + p.w > g.x and p.x < g.x + g.w:
            self.won = True

    def run(self):
        renderer = self.renderer
        if renderer is None:
            raise RuntimeError("run() requires rendering. Use step() for headless games.")

        import pygame
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()
                else:
                    renderer.handle_event(event)
            keys = pygame.key.get_pressed()
            self.step(keys[pygame.K_LEFT] or keys[pygame.K_a],
                      keys[pygame.K_RIGHT] or keys[pygame.K_d],
                      keys[pygame.K_SPACE])
            renderer.draw()
            renderer.tick(FPS)
        pygame.quit()

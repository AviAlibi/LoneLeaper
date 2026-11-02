import colorsys

import pygame

from game.game import FPS
from rendering.renderer import Renderer, SCROLL_STEP, VIEW_H, VIEW_W


class RaceRenderer(Renderer):
    def __init__(self, trainer):
        super().__init__(trainer.environments[0].game)
        pygame.display.set_caption("LoneLeaper - Evolution Race")
        self.trainer = trainer
        self.follow = True
        self.generation = trainer.generation
        self.colors = [
            tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(i / len(trainer.networks), 0.7, 1))
            for i in range(len(trainer.networks))
        ]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            self.follow = True
            self.camera.manual_offset = 0
        elif event.type == pygame.MOUSEWHEEL:
            self.follow = False
            self.camera.y -= event.y * SCROLL_STEP
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
            self.follow = False
            self.camera.y += (-1 if event.key == pygame.K_PAGEUP else 1) * SCROLL_STEP * 3
        self.camera.y = max(0, min(self.camera.y, self.game.world.height - VIEW_H))

    def draw(self, paused=False, speed=1):
        trainer = self.trainer
        leader = trainer.leader
        if self.generation != trainer.generation:
            self.camera.y = self.game.world.height - VIEW_H
            self.generation = trainer.generation
        if self.follow:
            self.camera.update(trainer.environments[leader].game.player)
        self.view.fill((30, 30, 45))
        cam = int(self.camera.y)
        for platform in self.game.world.platforms:
            color = (220, 190, 60) if platform.goal else (110, 110, 130)
            pygame.draw.rect(self.view, color,
                             (platform.x, platform.y - cam, platform.w, platform.h))
        target = trainer.environments[leader].target
        if target is not None:
            pygame.draw.rect(self.view, (70, 235, 120),
                             (target.x - 2, target.y - cam - 2, target.w + 4, target.h + 4), 2)
        # Draw the leader last, with a white outline, so it remains visible.
        order = [i for i in range(len(trainer.environments)) if i != leader] + [leader]
        for i in order:
            player = trainer.environments[i].game.player
            rect = pygame.Rect(player.x, player.y - cam, player.w, player.h)
            pygame.draw.rect(self.view, self.colors[i], rect, 2)
            if i == leader:
                pygame.draw.rect(self.view, (255, 255, 255), rect.inflate(4, 4), 1)
                label = self.font.render(f"#{i + 1}", True, (255, 255, 255))
                self.view.blit(label, (rect.x, rect.y - 18))

        # Full-level progress strip keeps offscreen racers visible.
        map_y, map_h = 130, VIEW_H - 150
        pygame.draw.rect(self.view, (10, 10, 18), (VIEW_W - 14, map_y, 14, map_h))
        for i in order:
            player = trainer.environments[i].game.player
            y = map_y + int(player.y / self.game.world.height * map_h)
            pygame.draw.line(self.view, self.colors[i], (VIEW_W - 13, y), (VIEW_W - 2, y), 2)
        viewport_y = map_y + int(self.camera.y / self.game.world.height * map_h)
        viewport_h = int(VIEW_H / self.game.world.height * map_h)
        pygame.draw.rect(self.view, (255, 255, 255),
                         (VIEW_W - 14, viewport_y, 14, viewport_h), 1)

        landed = max(e.landed_height for e in trainer.environments)
        winners = sum(e.game.won for e in trainer.environments)
        lines = [
            f"Gen {trainer.generation} | {len(trainer.networks)} racers | keep {trainer.survivors}",
            f"{trainer.frame / FPS:.1f}/{trainer.frame_limit / FPS:.1f}s | {speed}x"
            + (" | PAUSED" if paused else ""),
            f"Landed {landed:.0f} | Best {max(landed, trainer.best_height):.0f} | Wins {winners}",
            f"Leader #{leader + 1} | Fresh {trainer.newcomers} | "
            + ("EXPLORE" if trainer.exploring else f"Stale {trainer.stale_generations}"),
            "Space pause | +/- speed | F follow",
            "Wheel / PgUp / PgDn look up/down",
        ]
        pygame.draw.rect(self.view, (15, 15, 24), (0, 0, VIEW_W, 124))
        for i, line in enumerate(lines):
            self.view.blit(self.font.render(line, True, (235, 235, 235)), (7, 5 + i * 19))
        self.present()

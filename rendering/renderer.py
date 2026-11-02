import pygame

VIEW_W = 400
VIEW_H = 700
SCROLL_STEP = 60


class Camera:
    def __init__(self, world_height):
        self.world_height = world_height
        self.y = world_height - VIEW_H
        self.manual_offset = 0

    def update(self, player):
        target = player.y + player.h / 2 - VIEW_H / 2 + self.manual_offset
        target = max(0, min(target, self.world_height - VIEW_H))
        self.y += (target - self.y) * 0.15

    def scroll(self, amount):
        self.manual_offset += amount


class Renderer:
    def __init__(self, game):
        pygame.init()
        pygame.display.set_caption("LoneLeaper")
        self.game = game
        self.window = pygame.display.set_mode((VIEW_W, VIEW_H), pygame.RESIZABLE)
        self.view = pygame.Surface((VIEW_W, VIEW_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)
        self.camera = Camera(game.world.height)

    def handle_event(self, event):
        # Mouse wheel / PageUp / PageDown let you look up and down the level
        if event.type == pygame.MOUSEWHEEL:
            self.camera.scroll(-event.y * SCROLL_STEP)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PAGEUP:
                self.camera.scroll(-SCROLL_STEP * 3)
            elif event.key == pygame.K_PAGEDOWN:
                self.camera.scroll(SCROLL_STEP * 3)
            else:
                self.camera.manual_offset = 0

    def draw(self):
        game, player = self.game, self.game.player
        if player.charging or not player.on_ground:
            self.camera.manual_offset = 0
        self.camera.update(player)
        cam = int(self.camera.y)

        self.view.fill((30, 30, 45))
        for p in game.world.platforms:
            color = (220, 190, 60) if p.goal else (110, 110, 130)
            pygame.draw.rect(self.view, color, (p.x, p.y - cam, p.w, p.h))

        pygame.draw.rect(self.view, (200, 60, 60), (player.x, player.y - cam, player.w, player.h))
        if player.charging:
            pygame.draw.rect(self.view, (255, 255, 255),
                             (player.x, player.y - cam - 8, player.w * player.charge_ratio, 4))

        text = f"Height: {int(game.height())}  Best: {int(game.best_height)}"
        if game.won:
            text += "  YOU WIN! (R to reset)"
        self.view.blit(self.font.render(text, True, (230, 230, 230)), (8, 8))

        self.present()

    def present(self):
        """Scale the fixed 400x700 view to fit the window, keeping aspect ratio."""
        win_w, win_h = self.window.get_size()
        scale = min(win_w / VIEW_W, win_h / VIEW_H)
        size = (max(1, int(VIEW_W * scale)), max(1, int(VIEW_H * scale)))
        self.window.fill((0, 0, 0))
        scaled = pygame.transform.scale(self.view, size)
        self.window.blit(scaled, ((win_w - size[0]) // 2, (win_h - size[1]) // 2))
        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

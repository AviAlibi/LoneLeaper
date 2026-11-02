from game.game import Game
from ai.guidance import jump_direction, observe_guidance


class Environment:
    """One racer choosing human-equivalent inputs every simulation frame.

    Target inputs describe the next higher surface above the last landing and stay
    fixed in flight. Near-miss feedback is capped at 0.25 points per best landing,
    so it cannot outweigh even one pixel of actual landed progress.
    """

    def __init__(self):
        self.game = Game(render=False)
        self.reset()

    def reset(self):
        self.game.reset()
        self.landed_height = 0
        self.best_landing_frame: int | None = None
        self.near_landing_bonus = 0.0
        self.target = self.next_platform(self.game.player.y + self.game.player.h)
        self.actions = (False, False, False)

    def next_platform(self, feet):
        """Choose the lowest platform above the last supporting surface."""
        higher = [p for p in self.game.world.platforms if p.y < feet - 1]
        return max(higher, key=lambda p: p.y, default=None)

    def observe(self):
        game = self.game
        player = game.player
        center = player.x + player.w / 2
        feet = player.y + player.h
        observation = [
            center / game.world.width,
            game.height() / game.world.height,
            player.vx / 5,
            player.vy / 18,
            float(player.on_ground),
            player.charge_ratio,
            float(player.charging),
            (game.world.width - center) / game.world.width,
        ]
        nearby = sorted(game.world.platforms,
                        key=lambda p: abs(p.y - feet) + abs(p.x + p.w / 2 - center))
        for platform in nearby[:4]:
            observation.extend((
                (platform.x - center) / game.world.width,
                (platform.y - feet) / 300,
                platform.w / game.world.width,
                platform.h / 300,
            ))
        target = self.target
        if target is None:
            observation.extend((0, 0, 0, 0))
        else:
            # Signed distance to the interval in which the player's center can land.
            left = target.x - player.w / 2
            right = target.x + target.w + player.w / 2
            dx = min(max(center, left), right) - center
            observation.extend((dx / game.world.width,
                                (target.y - feet) / 300,
                                target.w / game.world.width, 1))
        observation.extend(observe_guidance(self))
        observation.append(jump_direction(self))
        return observation

    def step(self, network):
        if self.game.won:
            return
        self.step_actions(network.act(self.observe()))

    def step_actions(self, actions):
        """Advance physics and rewards with scalar or batched policy inputs."""
        if self.game.won:
            return
        self.actions = actions
        player = self.game.player
        previous_feet = player.y + player.h
        target = self.target
        self.game.step(*self.actions)
        if player.on_ground and self.game.height() > self.landed_height:
            self.landed_height = self.game.height()
            self.best_landing_frame = self.game.frames
            self.near_landing_bonus = 0.0
        if player.on_ground:
            self.target = self.next_platform(player.y + player.h)
        elif (target is not None and player.vy > 0
              and previous_feet <= target.y
              and self.game.world.height - target.y - 20 > self.landed_height):
            vertical_gap = max(0, target.y - (player.y + player.h))
            horizontal_gap = max(target.x - (player.x + player.w),
                                 player.x - (target.x + target.w), 0)
            proximity = max(0, 1 - horizontal_gap / 80) * max(0, 1 - vertical_gap / 40)
            # Best near miss only: repeated attempts cannot farm cumulative reward.
            self.near_landing_bonus = max(self.near_landing_bonus, 0.25 * proximity)

    @property
    def fitness(self):
        """Landings dominate; descending near misses and earlier arrivals break ties."""
        score = self.landed_height + self.near_landing_bonus
        if self.best_landing_frame is not None:
            # Platform heights are whole pixels; this cannot outweigh a higher landing.
            score += 0.5 / (1 + self.best_landing_frame)
        if self.game.won:
            score += 10000 + 1000 / (1 + self.game.frames)
        return score
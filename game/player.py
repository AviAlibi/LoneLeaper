from game import physics


class Player:
    def __init__(self, x, y):
        self.w = 20
        self.h = 28
        self.spawn = (x, y)
        self.reset()

    def reset(self):
        self.x, self.y = self.spawn
        self.vx = 0
        self.vy = 0
        self.on_ground = False
        self.charge = 0
        self.charging = False

    def update(self, left, right, jump, world):
        """jump = whether the jump button is held this frame."""
        if self.on_ground:
            if jump:
                self.charging = True
                self.vx = 0
                self.charge = min(self.charge + 1, physics.MAX_CHARGE_FRAMES)
                if self.charge >= physics.MAX_CHARGE_FRAMES:
                    self.release_jump(left, right)
            elif self.charging:
                self.release_jump(left, right)
            else:
                self.vx = (right - left) * physics.WALK_SPEED
        # No air control, just like Jump King
        physics.apply_gravity(self)
        physics.move_and_collide(self, world)
        if self.on_ground and not self.charging:
            # Landing kills horizontal momentum
            self.vx = (right - left) * physics.WALK_SPEED if not jump else 0

    def release_jump(self, left, right):
        ratio = self.charge / physics.MAX_CHARGE_FRAMES
        self.vy = -(physics.MIN_JUMP_SPEED + ratio * (physics.MAX_JUMP_SPEED - physics.MIN_JUMP_SPEED))
        self.vx = (right - left) * physics.JUMP_X_SPEED
        self.on_ground = False
        self.charging = False
        self.charge = 0

    @property
    def charge_ratio(self):
        return self.charge / physics.MAX_CHARGE_FRAMES

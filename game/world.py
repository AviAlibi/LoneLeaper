class Platform:
    def __init__(self, x, y, w, h, goal=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.goal = goal


# (x, height above the bottom of the level, width, height)
LEVEL = [
    (0, 20, 400, 20),      # ground
    (260, 160, 100, 15),
    (60, 300, 110, 15),
    (230, 450, 80, 15),
    (0, 600, 90, 15),
    (150, 640, 20, 160),   # wall
    (300, 760, 100, 15),
    (120, 900, 90, 15),
    (0, 1050, 70, 15),
    (200, 1180, 70, 15),
    (330, 1330, 70, 15),
    (170, 1480, 60, 15),
    (20, 1620, 80, 15),
    (230, 1780, 70, 15),
    (60, 1940, 110, 15, True),  # goal
]


class World:
    def __init__(self, width=400, height=2100):
        self.width = width
        self.height = height
        self.platforms = []
        for x, top, w, h, *goal in LEVEL:
            self.platforms.append(Platform(x, height - top, w, h, bool(goal)))
        self.goal = next(p for p in self.platforms if p.goal)
        self.spawn = (self.width / 2 - 10, self.height - 20 - 28)

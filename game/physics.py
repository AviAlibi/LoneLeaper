GRAVITY = 0.6
MAX_FALL_SPEED = 15
WALK_SPEED = 3
JUMP_X_SPEED = 4.5
MIN_JUMP_SPEED = 4
MAX_JUMP_SPEED = 18
MAX_CHARGE_FRAMES = 35
WALL_BOUNCE = 0.5


def overlaps(a, b):
    return a.x < b.x + b.w and a.x + a.w > b.x and a.y < b.y + b.h and a.y + a.h > b.y


def apply_gravity(body):
    body.vy = min(body.vy + GRAVITY, MAX_FALL_SPEED)


def move_and_collide(body, world):
    """Move body by its velocity, resolving collisions one axis at a time."""
    body.x += body.vx
    hit_wall = False
    if body.x < 0:
        body.x = 0
        hit_wall = True
    elif body.x + body.w > world.width:
        body.x = world.width - body.w
        hit_wall = True
    for p in world.platforms:
        if overlaps(body, p):
            body.x = p.x - body.w if body.vx > 0 else p.x + p.w
            hit_wall = True
    if hit_wall:
        # Bounce off walls while airborne, stop when walking
        body.vx = -body.vx * WALL_BOUNCE if not body.on_ground else 0

    body.y += body.vy
    body.on_ground = False
    for p in world.platforms:
        if overlaps(body, p):
            if body.vy > 0:
                body.y = p.y - body.h
                body.on_ground = True
            else:
                body.y = p.y + p.h
            body.vy = 0
    if body.y < 0:
        body.y = 0
        body.vy = 0

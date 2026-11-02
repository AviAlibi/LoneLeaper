from functools import lru_cache

from game import physics
from game.player import Player
from game.world import Platform, World


def jump_direction(environment):
    """Direction toward the target center, independent of repositioning hints."""
    target = environment.target
    if target is None:
        return 0
    player = environment.game.player
    offset = target.x + target.w / 2 - (player.x + player.w / 2)
    return (offset > 0) - (offset < 0)


@lru_cache(maxsize=32)
def geometry(width, platforms):
    world = World()
    world.width = width
    world.platforms = [Platform(*p) for p in platforms]
    return world


@lru_cache(maxsize=8192)
def predict(width, platforms, target_index, x, feet, charge):
    """Simulate a jump with real collisions, returning landing error and success."""
    world = geometry(width, platforms)
    target = world.platforms[target_index]
    player = Player(x, feet - 28)
    player.on_ground = True
    player.charge = charge
    direction = 1 if target.x + target.w / 2 > x + player.w / 2 else -1
    player.release_jump(direction < 0, direction > 0)
    for _ in range(120):
        physics.apply_gravity(player)
        physics.move_and_collide(player, world)
        if player.on_ground:
            success = (player.y + player.h == target.y
                       and player.x + player.w > target.x and player.x < target.x + target.w)
            center = player.x + player.w / 2
            error = (min(max(center, target.x), target.x + target.w) - center) / width
            # A ceiling collision or insufficient height can miss with zero x error.
            if not success and abs(error) < 0.01:
                error = float(direction)
            return error, success
    return float(direction), False


@lru_cache(maxsize=128)
def launch_options(width, platforms, support_index, target_index):
    support = platforms[support_index]
    options = []
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = support[0] + max(0, support[2] - 20) * fraction
        for charge in range(1, physics.MAX_CHARGE_FRAMES + 1):
            _, success = predict(width, platforms, target_index, x, support[1], charge)
            if success:
                options.append((x, charge))
    return tuple(options)


def observe_guidance(environment):
    """Launch offset, recommended charge, predicted landing error, predicted success.

    Only grounded states receive hints. Search is sampled, not an exhaustive
    planner; no candidates produces zero hints. All choices remain with the policy.
    """
    game = environment.game
    player = game.player
    target = environment.target
    if not player.on_ground or target is None:
        return (0, 0, 0, 0)
    platforms = tuple((p.x, p.y, p.w, p.h) for p in game.world.platforms)
    feet = player.y + player.h
    support_index = next((i for i, p in enumerate(game.world.platforms)
                          if p.y == feet and player.x + player.w > p.x
                          and player.x < p.x + p.w), None)
    if support_index is None:
        return (0, 0, 0, 0)
    target_index = game.world.platforms.index(target)
    options = launch_options(game.world.width, platforms, support_index, target_index)
    if not options:
        return (0, 0, 0, 0)
    x, charge = min(options, key=lambda option: (
        abs(option[0] - player.x), abs(option[1] - player.charge)))
    # Quantize only the diagnostic prediction, never the actual simulation.
    error, success = predict(game.world.width, platforms, target_index,
                             round(player.x / 2) * 2, feet, max(1, player.charge))
    return ((x - player.x) / game.world.width,
            charge / physics.MAX_CHARGE_FRAMES, error, float(success))

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Play LoneLeaper or watch evolving AI racers.",
        epilog="Race controls: Space pause, +/- speed, F follow leader, wheel or "
               "PageUp/PageDown scroll. --load starts a new population from one "
               "saved champion; it does not resume a full training session.",
    )
    parser.add_argument("--train", action="store_true", help="Run evolutionary training.")
    parser.add_argument("--headless", action="store_true", help="Train without opening a window.")
    parser.add_argument("--speed", type=int, default=1,
                        help="Target simulation speed, 1-254x (hardware permitting).")
    parser.add_argument("--generations", type=int, help="Stop after this many generations.")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--survival", type=float, default=5, help="Top percent retained.")
    parser.add_argument("--seconds", type=float, default=30, help="Game seconds per generation.")
    parser.add_argument("--seed", type=int, help="Reproducible starting population and mutations.")
    parser.add_argument("--checkpoint", default="champion.npz", help="Save best network here.")
    parser.add_argument("--load", help="Seed a fresh population from a saved champion.")
    args = parser.parse_args()
    if not 1 <= args.speed <= 254:
        parser.error("--speed must be between 1 and 254.")
    if args.generations is not None and args.generations < 1:
        parser.error("--generations must be positive.")
    if args.train or args.headless:
        from ai.train import Trainer
        Trainer(args.population, args.survival, args.seconds, args.seed,
                args.checkpoint, args.load).run(
                    headless=args.headless, speed=args.speed, generations=args.generations)
    else:
        if args.speed != 1 or args.generations is not None:
            parser.error("--speed and --generations require --train or --headless.")
        from game.game import Game
        Game().run()


if __name__ == "__main__":
    main()
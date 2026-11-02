import warnings

import numpy as np

LEGACY_INPUTS = 24
INPUTS = 33
HIDDEN = 12


class Network:
    """33 inputs -> 12 hidden units -> direction and jump hold.

    The first 24 inputs retain the original layout; four target inputs are appended.
    Four launch/prediction inputs follow the target inputs. Legacy checkpoints
    are padded with zero weights to preserve their policy. The final input is
    explicit target direction (-1 left, 0 centered/no target, +1 right).
    Mutated children use small changes 90% of the time, larger changes 10%.
    """

    def __init__(self, rng):
        self.weights = [
            rng.normal(0, 0.5, (INPUTS, HIDDEN)),
            rng.normal(0, 0.5, HIDDEN),
            rng.normal(0, 0.5, (HIDDEN, 2)),
            rng.normal(0, 0.5, 2),
        ]

    def act(self, observation):
        w1, b1, w2, b2 = self.weights
        hidden = np.tanh(np.asarray(observation) @ w1 + b1)
        direction, jump = np.tanh(hidden @ w2 + b2)
        return bool(direction < -0.25), bool(direction > 0.25), bool(jump > 0)

    def copy(self, rng, mutate=False, explore=False):
        child = Network.__new__(Network)
        child.weights = [w.copy() for w in self.weights]
        if mutate:
            scale = 0.35 if rng.random() < 0.1 or explore else 0.05
            for weight in child.weights:
                mask = rng.random(weight.shape) < 0.15
                weight += mask * rng.normal(0, scale, weight.shape)
        return child

    def save(self, path):
        with open(path, "wb") as file:
            np.savez(file, *self.weights)

    @classmethod
    def load(cls, path):
        shapes = ((INPUTS, HIDDEN), (HIDDEN,), (HIDDEN, 2), (2,))
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {f"arr_{i}" for i in range(4)}:
                raise ValueError("Invalid network checkpoint: expected four weight arrays.")
            weights = [np.asarray(data[f"arr_{i}"], dtype=float) for i in range(4)]
        if weights[0].shape in ((LEGACY_INPUTS, HIDDEN), (28, HIDDEN), (32, HIDDEN)):
            previous_inputs = weights[0].shape[0]
            weights[0] = np.pad(weights[0], ((0, INPUTS - previous_inputs), (0, 0)))
            warnings.warn(f"Migrated {previous_inputs}-input checkpoint: new guidance inputs initially "
                          "have zero weights; mutations can learn to use them.",
                          UserWarning, stacklevel=2)
        if any(w.shape != shape or not np.isfinite(w).all()
               for w, shape in zip(weights, shapes)):
            raise ValueError("Invalid network checkpoint: incompatible or non-finite weights.")
        network = cls.__new__(cls)
        network.weights = weights
        return network

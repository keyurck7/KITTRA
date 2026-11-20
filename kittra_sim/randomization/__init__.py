"""Domain randomization, sensor noise and actuator models (Phase 2).

``core`` is dependency-light (NumPy/OpenCV). ``wrappers`` needs Gymnasium and is imported lazily.
"""
from kittra_sim.randomization.core import (  # noqa: F401
    DelayLine,
    PhysicsRandomizer,
    VisualRandomizer,
    backlash_step,
)

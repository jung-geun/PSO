import tensorflow as tf
import os
from .optimizer import Optimizer as optimizer
from .particle import Particle as particle

__version__ = "1.0.5.1"

print("pso2keras version : " + __version__)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as r:
        print(r)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

__all__ = [
    "optimizer",
    "particle",
]

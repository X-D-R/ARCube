import os.path
from . import detection
from . import registration
from . import tracking
from . import utils

# Define the __all__ variable
__all__ = ["detection", "registration", "tracking", "utils"]

__version__ = '0.0.1'


MAIN_DIR = os.path.split(os.path.split(os.path.abspath("main.py"))[0])[0]
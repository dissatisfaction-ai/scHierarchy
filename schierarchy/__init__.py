"""scvi-tools-skeleton."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from .regression._reference_model import RegressionModel
from .regression._reference_module import RegressionBackgroundDetectionTechPyroModel

from .logistic._logistic_model import LogisticModel
from .logistic._logistic_module import HierarchicalLogisticPyroModel

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "schierarchy"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("scHierarchy: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = [
    "RegressionModel",
    "LogisticModel",
    "HierarchicalLogisticPyroModel",
    "RegressionBackgroundDetectionTechPyroModel",
]

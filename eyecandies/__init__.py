"""Top-level package for eyecandies."""

__author__ = "Eyecan.ai"
__email__ = "info@eyecan.ai"
__version__ = "1.0.1"

from eyecandies.commands.download import GetEyecandiesCommand
from eyecandies.commands.train import TrainCommand
from eyecandies.commands.test import TestCommand
from eyecandies.commands.predict import PredictCommand
from eyecandies.commands.metrics import ComputeMetricsCommand


def plmain():
    from pipelime.cli.main import run_with_extra_modules

    run_with_extra_modules("eyecandies")

"""Top-level package for eyecandies."""

__author__ = "Eyecan.ai"
__email__ = "info@eyecan.ai"
__version__ = "1.0.3"


def main():
    from pipelime.cli import PipelimeApp

    app = PipelimeApp(
        "eyecandies.commands", "eyecandies.stages", app_version=__version__
    )
    app()

import os.path

import click

from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.postprocessing import PostProcessWorkStation
from elara.benchmarking import BenchmarkWorkStation
from elara import factory


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config_path", type=click.Path(exists=True))
def cli(config_path):
    """
    Command line tool for processing a MATSim scenario events output.
    :param config_path: Configuration file path
    """
    config = Config(config_path)
    main(config)


def main(config):
    """
    Main logic:
        1) define workstations
        2) connect workstations (define dependencies)
        3) build all resulting graph requirements
    :param config: Session configuration object
    """
    # Create output folder if it does not exist
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # 1: Define Work Stations
    config_requirements = RequirementsWorkStation(config)
    postprocessing = PostProcessWorkStation(config)
    benchmarks = BenchmarkWorkStation(config)
    event_handlers = EventHandlerWorkStation(config)
    plan_handlers = PlanHandlerWorkStation(config)
    input_workstation = InputsWorkStation(config)
    paths = PathFinderWorkStation(config)

    # 2: Connect Workstations
    config_requirements.connect(
        managers=None,
        suppliers=[postprocessing, benchmarks, event_handlers, plan_handlers]
    )
    benchmarks.connect(
        managers=[config_requirements],
        suppliers=[event_handlers, plan_handlers],
    )
    postprocessing.connect(
        managers=[config_requirements],
        suppliers=[event_handlers, plan_handlers]
    )
    event_handlers.connect(
        managers=[postprocessing, benchmarks, config_requirements],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[config_requirements, benchmarks, postprocessing],
        suppliers=[input_workstation]
    )
    input_workstation.connect(
        managers=[event_handlers, plan_handlers],
        suppliers=[paths]
    )
    paths.connect(
        managers=[input_workstation],
        suppliers=None
    )

    # 3: Build all requirements
    factory.build(config_requirements, verbose=config.verbose)

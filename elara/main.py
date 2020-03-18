import os.path

import click
import logging
from pathlib import Path

from elara.helpers import PathPath, NaturalOrderGroup
from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.postprocessing import PostProcessWorkStation
from elara.benchmarking import BenchmarkWorkStation
from elara import factory


@click.group()
def cli():
    """
    Command line tool for processing a MATSim scenario events output.
    """
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path):
    """
    Run Elara using a config.toml, examples are included in the repo and readme.
    :param config_path: Configuration file path
    """
    config = Config(config_path)
    main(config)


def common_options(func):
    func = click.option('-d', '--debug', is_flag=True)(func)
    func = click.option('-n', '--name', type=click.STRING, default=str(Path(os.getcwd())).split('/')[-1])(func)
    func = click.option('-i', '--inputs_path', type=PathPath(exists=True), default=Path(os.getcwd()))(func)
    func = click.option('-o', '--outputs_path', type=PathPath(), default=Path(os.getcwd()) / 'elara_out')(func)
    func = click.option('-p', '--time_periods', type=click.INT, default=24)(func)
    func = click.option('-s', '--scale_factor', type=click.FLOAT, default=0.1)(func)
    func = click.option('-e', '--epsg', type=click.STRING, default="EPSG:27700")(func)
    func = click.option(
        '-f', '--full', is_flag=True, default=True,
        help="Option to disable output contracting.")(func)
    return func


@cli.group(name="event-handlers", cls=NaturalOrderGroup)
def event_handlers():
    """
    Access event handler output group.
    """
    pass


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def volume_counts(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a volume counts output for a given mode or modes. Example invocation for "car" and "bus"
    modes with name "test" and scale factor at 20% is:

    $ elara event-handlers volume-counts car bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["event_handlers"]["volume_counts"] = list(modes)
    config = Config(override=override)
    main(config)


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def passenger_counts(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a passenger counts output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers passenger-counts train bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["event_handlers"]["passenger_counts"] = list(modes)
    config = Config(override=override)
    main(config)


@event_handlers.command()
@common_options
def stop_interactions(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a stop interactions output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers stop-interactions train bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["event_handlers"]["stop_interactions"] = list(modes)
    config = Config(override=override)
    main(config)


@cli.group(name="plan-handlers")
def plan_handlers():
    """
    Access plan handler output group.
    """
    pass


@plan_handlers.command()
@click.argument('options', nargs=-1, type=click.STRING, required=True)
@common_options
def mode_share(
        options, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a mode share output for a given option. Example invocation for option "all" name
    "test" and scale factor at 20% is:

    $ elara plan-handlers mode-share all -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["plan_handlers"]["mode_share"] = list(options)
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('options', nargs=-1, type=click.STRING, required=True)
@common_options
def agent_logs(
        options, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create an agent logs output for a given option. Example invocation for option "all" name
    "test" and scale factor at 20% is:

    $ elara plan-handlers agent-logs all -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["plan_handlers"]["agent_logs"] = list(options)
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('options', nargs=-1, type=click.STRING, required=True)
@common_options
def agent_plans(
        options, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create an agent plans output for a given option. Example invocation for sub-populations "a",
    "b" and "c" with name "test" and scale factor at 20% is:

    $ elara plan-handlers agent-plans a b c -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["plan_handlers"]["agent_plans"] = list(options)
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('options', nargs=-1, type=click.STRING, required=True)
@common_options
def highway_distances(
        options, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a highways distances output for a given option. Example invocation for option "car" name
    "test" and scale factor at 20% is:

    $ elara plan-handlers highway-distances car -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["plan_handlers"]["highway_distances"] = list(options)
    config = Config(override=override)
    main(config)


@cli.group(name="post-processors")
def post_processors():
    """
    Access post processing output group.
    """
    pass


@post_processors.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def vkt(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a VKT output for a given mode or mnodes. Example invocation for mode "car", name
    "test" and scale factor at 20% is:

    $ elara plan-processors vkt car -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["post_processors"]["vkt"] = list(modes)
    config = Config(override=override)
    main(config)


@cli.group(name="benchmarks")
def benchmarks():
    """
    Access benchmarks output group.
    """
    pass


@benchmarks.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def london_central_cordon(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a london_central_cordon output for a given mode or modes. Example invocation for modes
    "car" and "bus", name "test" and scale factor at 20% is:

    $ elara benchmarks london-central-cordon car bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["benchmarks"]["london_central_cordon"] = list(modes)
    config = Config(override=override)
    main(config)


def main(config):
    """
    Main logic:
        1) define workstations
        2) connect workstations (define dependencies)
        3) build all resulting graph requirements
    :param config: Session configuration object
    """

    logging.basicConfig(
        level=config.logging,
        format='%(asctime)s %(name)-12s %(levelname)-3s %(message)s',
        datefmt='%m-%d %H:%M'
    )
    logger = logging.getLogger(__name__)

    logger.info('Starting')

    # Create output folder if it does not exist
    if not os.path.exists(config.output_path):
        logger.info(f'Creating new output directory {config.output_path}')
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
        suppliers=[input_workstation, event_handlers, plan_handlers]
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
        managers=[event_handlers, plan_handlers, postprocessing],
        suppliers=[paths]
    )
    paths.connect(
        managers=[input_workstation],
        suppliers=None
    )

    # 3: Build all requirements
    factory.build(config_requirements)

    logger.info('Done')


def common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, contract
):
    return {
        "scenario":
            {
                "name": name,
                "time_periods": time_periods,
                "scale_factor": scale_factor,
                "crs": epsg,
                "verbose": debug,
            },
        "inputs":
            {
                "events": inputs_path / "output_events.xml.gz",
                "network": inputs_path / "output_network.xml.gz",
                "transit_schedule": inputs_path / "output_transitSchedule.xml.gz",
                "transit_vehicles": inputs_path / "output_transitVehicles.xml.gz",
                "attributes": inputs_path / "output_personAttributes.xml.gz",
                "plans": inputs_path / "output_plans.xml.gz",
                "output_config_path": inputs_path / "output_config.xml",
            },
        "event_handlers":
            {
             },
        "plan_handlers":
            {
            },
        "post_processors":
            {
            },
        "benchmarks":
            {
            },
        "outputs":
            {
                "path": outputs_path,
                "contract": contract,
            },
    }


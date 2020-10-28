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


@click.group(cls=NaturalOrderGroup)
def cli():
    """
    Command line tool for processing a MATSim scenario events output.
    """
    pass


def common_options(func):
    func = click.option(
        '-d', '--debug', is_flag=True, help="Switch on debug verbosity."
    )(func)

    func = click.option(
        '-n', '--name', type=click.STRING, default=str(Path(os.getcwd())).split('/')[-1],
        help="Scenario name, defaults to root dir name."
    )(func)

    func = click.option(
        '-i', '--inputs_path', type=PathPath(exists=True), default=Path(os.getcwd()),
        help="Inputs path location, defaults to current root."
    )(func)

    func = click.option(
        '-o', '--outputs_path', type=PathPath(), default=Path(os.getcwd()) / 'elara_out',
        help="Outputs path, defaults to './elara_out'."
    )(func)

    func = click.option(
        '-p', '--time_periods', type=click.INT, default=24,
        help="Time period breakdown, defaults to 24 (hourly."
    )(func)

    func = click.option(
        '-s', '--scale_factor', type=click.FLOAT, default=0.1,
        help="Scale factor, defaults to 0.1 (10%)."
    )(func)

    func = click.option(
        '-e', '--epsg', type=click.STRING, default="EPSG:27700",
        help="EPSG string, defaults to 'EPSG:27700' (UK)."
    )(func)

    func = click.option(
        '-f', '--full', is_flag=True, default=True,
        help="Option to disable output contracting."
    )(func)

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
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def route_passenger_counts(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a route passenger counts output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers route-passenger-counts train bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["event_handlers"]["route_passenger_counts"] = list(modes)
    config = Config(override=override)
    main(config)


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
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


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def vehicle_interactions(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a vehicle interactions output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers vehicle-interactions train bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["event_handlers"]["vehicle_interactions"] = list(modes)
    config = Config(override=override)
    main(config)


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def stop_to_stop(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create stop to stop passenger counts output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers stop_to_stop train bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["event_handlers"]["passenger_stop_to_stop_loading"] = list(modes)
    config = Config(override=override)
    main(config)


@cli.group(name="plan-handlers", cls=NaturalOrderGroup)
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
def agent_highway_distances(
        options, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create an agent highways distances output for a given option. Example invocation for option "car" name
    "test" and scale factor at 20% is:

    $ elara plan-handlers highway-distances car -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["plan_handlers"]["agent_highway_distances"] = list(options)
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('options', nargs=-1, type=click.STRING, required=True)
@common_options
def trip_highway_distances(
        options, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a trip highways distances output for a given option. Example invocation for option "car" name
    "test" and scale factor at 20% is:

    $ elara plan-handlers highway-distances car -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["plan_handlers"]["trip_highway_distances"] = list(options)
    config = Config(override=override)
    main(config)


@cli.group(name="post-processors", cls=NaturalOrderGroup)
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
    Create a VKT output for a given mode or modes. Example invocation for mode "car", name
    "test" and scale factor at 20% is:

    $ elara plan-processors vkt car -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["post_processors"]["vkt"] = list(modes)
    config = Config(override=override)
    main(config)


@post_processors.command()
@click.argument('options', nargs=-1, type=click.STRING, required=True)
@common_options
def plan_summary(
        options, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a Plan Summary output. Example invocation for option "all", name
    "test" and scale factor at 20% is:

    $ elara plan-processors plan-summary all -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["post_processors"]["plan_summary"] = list(options)
    config = Config(override=override)
    main(config)


@cli.group(name="benchmarks", cls=NaturalOrderGroup)
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
    Create a london_central_cordon output for cars. Example invocation for mode
    "car", name "test" and scale factor at 20% is:

    $ elara benchmarks london-central-cordon car -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["benchmarks"]["london_central_cordon_car"] = list(modes)
    config = Config(override=override)
    main(config)


@benchmarks.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def london_modeshares(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a london modeshares benchmark. Example invocation for
    all modes, name "test" and scale factor at 20% is:

    $ elara benchmarks london-modeshares all -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["benchmarks"]["london_modeshares"] = list(modes)
    config = Config(override=override)
    main(config)


@benchmarks.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def ireland_highways(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a ireland highways output for a given mode or modes. Example invocation for modes
    "car" and "bus", name "test" and scale factor at 20% is:

    $ elara benchmarks london-central-cordon car bus -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["benchmarks"]["ireland_highways"] = list(modes)
    config = Config(override=override)
    main(config)


@benchmarks.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def london_rods_stops(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a london rods stops (boardings/alightings) output for a given mode or modes. Example invocation for mode
    "subway", name "test" and scale factor at 20% is:

    $ elara benchmarks london-central-cordon subway -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["benchmarks"]["london_board_alight_subway"] = list(modes)
    config = Config(override=override)
    main(config)


@benchmarks.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def london_rods_volumes(
        modes, debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
):
    """
    Create a london rods volumes (station to station) output for a given mode or modes. Example invocation for mode
    "subway", name "test" and scale factor at 20% is:

    $ elara benchmarks london-central-cordon subway -n test -s .2
    """
    override = common_override(
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
    )
    override["benchmarks"]["london_volume_subway"] = list(modes)
    config = Config(override=override)
    main(config)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--path_override", '-o', default=None)
def run(config_path, path_override):
    """
    Run Elara using a config.
    :param config_path: Configuration file path
    :param path_override: containing directory to update for [inputs], outputs.path in toml
    """

    config = Config(config_path)

    if path_override:
        config.override(path_override)
        
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
        debug, name, inputs_path, outputs_path, time_periods, scale_factor, epsg, full
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
                "contract": not full,
            },
    }


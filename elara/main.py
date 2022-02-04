from email.policy import default
import os.path
from attr import attributes

import click
import logging
from pathlib import Path
from elara import input_plan_handlers

from elara.helpers import PathPath, NaturalOrderGroup
from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.input_plan_handlers import InputPlanHandlerWorkstation
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


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--path_override", '-o', default=None)
@click.option("--root", '-r', default=None)
@click.option("--output_directory_override", default=None)
def run(config_path, path_override, root, output_directory_override):
    """
    Run Elara using a config.
    :param config_path: Configuration file path
    :param path_override: containing directory to update for [inputs], outputs.path in toml
    :param root: add root to all paths (assumes that paths in config are relative)
    :param output_directory_override: change output directory
    """
    if path_override and root:
        raise UserWarning(
            "Cannot run elara from config with both --path_override and --root options, please choose one."
            )

    config = Config(config_path)

    if path_override:
        config.override(path_override)
    
    if root:
        config.set_paths_root(root)

    if output_directory_override:
        config.output_directory_override(output_directory_override)
        
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

    # TODO
    config.experienced_plans_warning()

    # Create output folder if it does not exist
    if not os.path.exists(config.output_path):
        logger.info(f'Creating new output directory {config.output_path}')
        os.makedirs(config.output_path)

    # 1: Define Work Stations
    config_requirements = RequirementsWorkStation(config)
    postprocessing = PostProcessWorkStation(config)
    benchmarks = BenchmarkWorkStation(config)
    event_handlers = EventHandlerWorkStation(config)
    input_plan_handlers = InputPlanHandlerWorkstation(config)
    plan_handlers = PlanHandlerWorkStation(config)
    input_workstation = InputsWorkStation(config)
    paths = PathFinderWorkStation(config)

    # 2: Connect Workstations
    config_requirements.connect(
        managers=None,
        suppliers=[postprocessing, benchmarks, event_handlers, plan_handlers, input_plan_handlers]
    )
    benchmarks.connect(
        managers=[config_requirements],
        suppliers=[postprocessing, event_handlers, plan_handlers, input_plan_handlers],
    )
    postprocessing.connect(
        managers=[config_requirements, benchmarks],
        suppliers=[input_workstation, event_handlers, plan_handlers, input_plan_handlers]
    )
    event_handlers.connect(
        managers=[postprocessing, benchmarks, config_requirements],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[config_requirements, benchmarks, postprocessing],
        suppliers=[input_workstation]
    )
    input_plan_handlers.connect(
        managers=[config_requirements, benchmarks, postprocessing],
        suppliers=[input_workstation]
    )
    input_workstation.connect(
        managers=[event_handlers, plan_handlers, input_plan_handlers, postprocessing],
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
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    
    plans_file = "output_experienced_plans.xml.gz" if experienced_plans else "output_plans.xml.gz"
    attributes_file = "output_plans.xml.gz" if version >= 12 else "output_personAttributes.xml.gz"

    return {
        "scenario":
            {
                "name": name,
                "time_periods": time_periods,
                "scale_factor": scale_factor,
                "version": version,
                "crs": epsg,
                "verbose": debug,
            },
        "inputs":
            {
                "events": inputs_path / "output_events.xml.gz",
                "network": inputs_path / "output_network.xml.gz",
                "transit_schedule": inputs_path / "output_transitSchedule.xml.gz",
                "transit_vehicles": inputs_path / "output_transitVehicles.xml.gz",
                "attributes": inputs_path / attributes_file,
                "plans": inputs_path / plans_file,
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

def common_options(func):
    func = click.option(
        '-d', '--debug', is_flag=True, help="Switch on debug verbosity."
    )(func)

    func = click.option(
        '-n', '--name', type=click.STRING, default=str(Path(os.getcwd())).split('/')[-1],
        help="Scenario name, defaults to root dir name."
    )(func)

    func = click.option(
        '-x', '--experienced_plans', is_flag=True, default=True, 
        help="Switch for Using Experienced Plans, defaults to True"
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
        '-v', '--version', type=click.INT, default=12,
        help="MATSim version {11,12}"
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
def link_vehicle_counts(
        modes, debug, experienced_plans, name, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a volume counts output for a given mode or modes. Example invocation for "car" and "bus"
    modes with name "test" and scale factor at 20% is:

    $ elara event-handlers link-vehicle-counts car bus -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["event_handlers"]["link_vehicle_counts"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def link_passenger_counts(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a passenger counts output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers link-passenger-counts train bus -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["event_handlers"]["link_passenger_counts"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def route_passenger_counts(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a route passenger counts output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers route-passenger-counts train bus -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans,inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["event_handlers"]["route_passenger_counts"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def stop_passenger_counts(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a stop interactions output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers stop-passenger-counts train bus -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["event_handlers"]["stop_passenger_counts"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def stop_to_stop_passenger_counts(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create stop to stop passenger counts output for a given mode or modes. Example invocation for "train" and
    "bus" modes with name "test" and scale factor at 20% is:

    $ elara event-handlers stop_to_stop_passenger_counts train bus -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["event_handlers"]["stop_to_stop_passenger_counts"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)

@event_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def vehicle_link_log(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    create a vehicle link log

    $ elara event-handlers stop_to_stop_passenger_counts train bus -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["event_handlers"]["vehicle_link_log"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@cli.group(name="plan-handlers", cls=NaturalOrderGroup)
def plan_handlers():
    """
    Access plan handler output group.
    """
    pass


@plan_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def mode_shares(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a mode share output for a given option. Example invocation for option "all" and
     scale factor at 20% is:

    $ elara plan-handlers mode-shares all -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["mode_shares"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)

@plan_handlers.command()
@click.argument('destination_activity_filters', nargs=-1, type=click.STRING, required=True)
@common_options
def trip_destination_mode_share(
        debug, name, experienced_plans, inputs_path, outputs_path, destination_activity_filters, time_periods, scale_factor, version, epsg, full
):
    """
    Create a mode share output for a given option. Example invocation for option "work" and
     scale factor at 20% is:

    $ elara plan-handlers activity-mode-shares work -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["trip_destination_mode_share"] = {'modes': ['all'], 'destination_activity_filters':destination_activity_filters}
    config = Config(override=override)
    main(config)

@plan_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def leg_logs(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create an agent leg logs output. Example invocation for option "all" is:

    $ elara plan-handlers leg-logs all
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["leg_logs"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def trip_logs(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create an agent trip logs output. Example invocation for option "all" is:

    $ elara plan-handlers trip-logs all
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["trip_logs"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def utility_logs(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create an agent plan utility output. Example invocation for option "all" is:

    $ elara plan-handlers utility_logs all
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["utility_logs"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def plan_logs(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create an agent plans output for a given option. Example invocation for sub-populations "a",
    "b" and "c" with name "test" and scale factor at 20% is:

    $ elara plan-handlers plan_logs a b c -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["plan_logs"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def agent_highway_distance_logs(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create an agent highways distances output for a given option. Example invocation for option "car" name
    "test" and scale factor at 20% is:

    $ elara plan-handlers agent_highway_distance_logs car -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["agent_highway_distance_logs"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)


@plan_handlers.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def trip_highway_distance_logs(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a trip highways distances output for a given option. Example invocation for option "car" name
    "test" and scale factor at 20% is:

    $ elara plan-handlers trip_highway_distance_logs car -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["plan_handlers"]["trip_highway_distance_logs"] = {'modes': list(modes)}
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
        modes, debug, experienced_plans, name, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a VKT output for a given mode or modes. Example invocation for mode "car", name
    "test" and scale factor at 20% is:

    $ elara plan-processors vkt car -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["post_processors"]["vkt"] = list(modes)
    config = Config(override=override)
    main(config)


@post_processors.command()
@click.argument('modes', nargs=-1, type=click.STRING, required=True)
@common_options
def plan_summary(
        modes, debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
):
    """
    Create a Plan Summary output. Example invocation for option "all", name
    "test" and scale factor at 20% is:

    $ elara plan-processors plan-summary all -n test -s .2
    """
    override = common_override(
        debug, name, experienced_plans, inputs_path, outputs_path, time_periods, scale_factor, version, epsg, full
    )
    override["post_processors"]["plan_summary"] = {'modes': list(modes)}
    config = Config(override=override)
    main(config)

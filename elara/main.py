import os.path

import click
from halo import Halo

from elara.config import Config
from elara import inputs
from elara import event_handlers
from elara import plan_handlers
from elara import postprocessing
from elara import benchmarking


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
    Main logic.
    :param config: Session configuration object
    """
    # Create output folder if it does not exist
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # Prepare inputs
    with Halo(text="Preparing inputs...", spinner="dots") as spinner:
        spinner.text = "Preparing events input..."
        events = inputs.Events(config.events_path)
        spinner.text = "Preparing network input..."
        network = inputs.Network(config.network_path, config.crs)
        spinner.text = "Preparing schedule input..."
        transit_schedule = inputs.TransitSchedule(
            config.transit_schedule_path, config.crs
        )
        spinner.text = "Preparing transit vehicle input..."
        transit_vehicles = inputs.TransitVehicles(config.transit_vehicles_path)
        spinner.text = "Preparing Subpopulation Attribute input..."
        attributes = inputs.Attributes(config.attributes_path)
        spinner.text = "Preparing Plans input..."
        plans = inputs.Plans(config.plans_path, transit_schedule)

        spinner.succeed("Inputs prepared.")
        if config.verbose:
            print('--- Loading Summary ---')
            print('Event Handlers: {}'.format(config.event_handlers))
            print('Plan Handlers: {}'.format(config.plan_handlers))
            print('{} Vertexes Loaded'.format(len(network.node_gdf)))
            print('{} Edges Loaded'.format(len(network.link_gdf)))
            print('{} Transit Stops Loaded'.format(len(transit_schedule.stop_gdf)))
            print('Transit Capacities: {}'.format(transit_vehicles.veh_type_capacity_map))
            print('Transit Vehicles: {}'.format(transit_vehicles.transit_vehicle_counts))
            print('Sub-populations: {}'.format(attributes.attribute_count_map))
            print('Plan Activity Types: {}'.format(plans.activities))
            print('Plan Leg Modes: {}'.format(plans.modes))
            print('-----------------------')


    # Build event handlers
    with Halo(text="Building event handlers...", spinner="dots") as spinner:
        active_event_handlers = list()
        for handler_name, mode_list in config.event_handlers.items():
            for mode in mode_list:
                active_event_handlers.append(
                    event_handlers.EVENT_HANDLER_MAP[handler_name](
                        network,
                        transit_schedule,
                        transit_vehicles,
                        attributes,
                        mode,
                        config.time_periods,
                        config.scale_factor,
                    )
                )
        spinner.succeed(f"{len(active_event_handlers)} event handlers prepared.")

    # Build plan handlers
    with Halo(text="Building plan handlers...", spinner="dots") as spinner:
        active_plan_handlers = list()
        for handler_name, acts in config.plan_handlers.items():
            for act in acts:
                active_plan_handlers.append(
                    plan_handlers.PLAN_HANDLER_MAP[handler_name](
                        act,
                        plans,
                        transit_schedule,
                        attributes,
                        config.time_periods,
                        config.scale_factor,
                    )
                )
        spinner.succeed(f"{len(active_plan_handlers)} plan handlers prepared.")

    # Build post-processors
    with Halo(text="Building post-processors...", spinner="dots") as spinner:
        post_processors = list()
        for post_processor_name in config.post_processing:
            post_processor = postprocessing.POST_PROCESSOR_MAP[post_processor_name](
                config, network, transit_schedule, transit_vehicles
            )
            if not post_processor.check_prerequisites():
                raise Exception(
                    "Prerequisite handlers not met for {} post-processor".format(
                        post_processor_name
                    )
                )
            post_processors.append(post_processor)
        spinner.succeed("Post-processors prepared.")

    # Build benchmarks
    with Halo(text="Building benchmarks...", spinner="dots") as spinner:
        benchmarks = benchmarking.Benchmarks(config)
        spinner.succeed("Benchmarks prepared.")

    # Iterate through events
    with Halo(text="Processing events...", spinner="dots") as spinner:
        for i, event in enumerate(events.elems):
            if i % 12345:
                spinner.text = "Processed {:,} events...".format(i + 1)
            for event_handler in active_event_handlers:
                event_handler.process_event(event)
        spinner.succeed("Events processed!")

    # Iterate through plans
    with Halo(text="Processing plans...", spinner="dots") as spinner:
        for i, plan in enumerate(plans.elems):
            if i % 123:
                spinner.text = "Processed {:,} plans...".format(i + 1)
            for plan_handler in active_plan_handlers:
                plan_handler.process_event(plan)
        spinner.succeed("Plans processed!")

    # Generate event file outputs
    with Halo(text="Generating event outputs...", spinner="dots") as spinner:
        for event_handler in active_event_handlers:
            event_handler.finalise()
            if config.contract:
                event_handler.contract_results()

            for name, gdf in event_handler.result_gdfs.items():
                csv_name = "{}_{}.csv".format(config.name, name)
                geojson_name = "{}_{}.geojson".format(config.name, name)
                csv_path = os.path.join(config.output_path, csv_name)
                geojson_path = os.path.join(config.output_path, geojson_name)

                # File exports
                spinner.text = "Writing {}".format(csv_name)
                gdf.drop("geometry", axis=1).to_csv(csv_path)
                spinner.text = "Writing {}".format(geojson_name)
                export_geojson(gdf, geojson_path)

        spinner.succeed("Event outputs generated!")

    # Generate file outputs
    with Halo(text="Generating plan outputs...", spinner="dots") as spinner:
        for event_handler in active_plan_handlers:
            event_handler.finalise()

            for name, result in event_handler.results.items():
                csv_name = "{}_{}.csv".format(config.name, name)
                csv_path = os.path.join(config.output_path, csv_name)

                # File exports
                spinner.text = "Writing {}".format(csv_name)
                result.to_csv(csv_path)

        spinner.succeed("Plan outputs generated!")

    # Run post-processing modules
    with Halo(text="Running post-processing...", spinner="dots") as spinner:
        for post_processor in post_processors:
            post_processor.run()
        spinner.succeed("Post-processing complete!")

    # Run benchmarking
    with Halo(text="Running benchmarking...", spinner="dots") as spinner:
        meta_score = benchmarks.score()
        spinner.succeed("Benchmark completed with METASCORE of {} and breakdown as follows:".format(meta_score))

    print(benchmarks.scores_df)


def export_geojson(gdf, path):
    """
    Given a geodataframe, export geojson representation to specified path.
    :param gdf: Input geodataframe
    :param path: Output path
    """
    with open(path, "w") as file:
        file.write(gdf.to_json())

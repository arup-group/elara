import os.path

import click
from halo import Halo

from elara.config import Config
from elara import inputs
from elara import handlers

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
        events = inputs.Events(config.events_path)
        network = inputs.Network(config.network_path)
        spinner.succeed("Inputs prepared.")

    # Build handlers
    with Halo(text="Building event handlers...", spinner="dots") as spinner:
        event_handlers = list()
        for handler_name, mode_list in config.handlers.items():
            for mode in mode_list:
                event_handlers.append(
                    handlers.HANDLER_MAP[handler_name](
                        network, mode, config.time_periods, config.scale_factor
                    )
                )
        spinner.succeed("Event handlers prepared.")

    # Iterate through events
    with Halo(text="Processing events...", spinner="dots") as spinner:
        for i, event in enumerate(events.events):
            if i % 12345:
                spinner.text = "Processed {:,} events...".format(i + 1)
            for event_handler in event_handlers:
                event_handler.process_event(event)
        spinner.succeed("Events processed!")

    # Generate file outputs
    with Halo(text="Generating outputs...", spinner="dots") as spinner:
        for event_handler in event_handlers:
            event_handler.generate_results()
            for name, df in event_handler.final_tables.items():
                spinner.text = "Writing {}".format(name)
                path = os.path.join(
                    config.output_path, "{}_{}.csv".format(config.name, name)
                )
                df.to_csv(path)
        spinner.succeed("Outputs generated!")

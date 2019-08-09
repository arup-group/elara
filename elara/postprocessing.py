import os
import geopandas
import pandas as pd

from elara.handlers.agent_plan_handlers import *
from elara.handlers.network_event_handlers import *
from elara import ConfigPostProcessorError, PostProcessorPrerequisiteError


class PostProcessor:

    requirements = ['volume_counts']

    handler_prerequisite = None

    def __init__(self, config, network, transit_schedule, transit_vehicles):
        self.config = config
        self.network = network
        self.transit_schedule = transit_schedule
        self.transit_vehicles = transit_vehicles

    def check_prerequisites(self):
        return NotImplementedError

    def run(self):
        return NotImplementedError

    def check_handler_prerequisite(self, handler_manager):
        if self.handler_prerequisite and not sum(
                [isinstance(h, self.handler_prerequisite) for h in handler_manager.all]
        ):
            raise PostProcessorPrerequisiteError(
                f'Missing handler: {self.handler_prerequisite} for postprocessing')


class VKT(PostProcessor):

    handler_prerequisite = VolumeCounts

    def __init__(self, config, network, transit_schedule, transit_vehicles):
        super().__init__(config, network, transit_schedule, transit_vehicles)

    def check_prerequisites(self):
        return (
            "volume_counts" in self.config.event_handlers
            and len(self.config.event_handlers["volume_counts"]) > 0
        )

    def run(self):
        for mode in self.config.event_handlers["volume_counts"]:
            file_name = "{}_volume_counts_{}_total.geojson".format(self.config.name, mode)
            file_path = os.path.join(self.config.output_path, file_name)
            volumes_gdf = geopandas.read_file(file_path)

            # Calculate VKT
            period_headers = generate_period_headers(self.config.time_periods)
            volumes = volumes_gdf[period_headers]
            link_lengths = volumes_gdf["length"].values / 1000  # Conversion to metres
            vkt = volumes.multiply(link_lengths, axis=0)
            vkt_gdf = pd.concat([volumes_gdf.drop(period_headers, axis=1), vkt], axis=1)

            # Export results
            csv_path = os.path.join(
                self.config.output_path, "{}_vkt_{}.csv".format(self.config.name, mode)
            )
            geojson_path = os.path.join(
                self.config.output_path,
                "{}_vkt_{}.geojson".format(self.config.name, mode),
            )
            vkt_gdf.drop("geometry", axis=1).to_csv(csv_path)
            export_geojson(vkt_gdf, geojson_path)


def generate_period_headers(time_periods):
    """
    Generate a list of strings corresponding to the time period headers in a typical
    data output.
    :param time_periods: Number of time periods across the day
    :return: List of time period numbers as strings
    """
    return list(map(str, range(time_periods)))


def export_geojson(gdf, path):
    """
    Given a geodataframe, export geojson representation to specified path.
    :param gdf: Input geodataframe
    :param path: Output path
    """
    with open(path, "w") as file:
        file.write(gdf.to_json())


# Dictionary used to map configuration string to post-processor type
POST_PROCESSOR_MAP = {"vkt": VKT}

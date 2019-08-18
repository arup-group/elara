import os
import geopandas
import pandas as pd

from elara.factory import WorkStation, Tool


class PostProcessor(Tool):
    options_enabled = True

    def check_prerequisites(self):
        return NotImplementedError

    def build(self, resource):
        return NotImplementedError


class VKT(PostProcessor):
    requirements = ['volume_counts']
    valid_options = ['car', 'bus', 'train', 'subway', 'ferry']

    # def __init__(self, config, option=None):
    #     super().__init__(config, option)

    def build(self, resource: dict):
        super().build(resource)

        mode = self.option

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


class PostProcessWorkStation(WorkStation):

    tools = {
        'vkt': VKT,
    }


import os

import geopandas


class PostProcessor:
    def __init__(self, config, network, transit_schedule, transit_vehicles):
        self.config = config
        self.network = network
        self.transit_schedule = transit_schedule
        self.transit_vehicles = transit_vehicles

    def check_prerequisites(self):
        return NotImplementedError

    def run(self):
        return NotImplementedError


class VKT(PostProcessor):
    def __init__(self, config, network, transit_schedule, transit_vehicles):
        super().__init__(config, network, transit_schedule, transit_vehicles)

    def check_prerequisites(self):
        return (
            "volume_counts" in self.config.handlers
            and len(self.config.handlers["volume_counts"]) > 0
        )

    def run(self):
        link_lengths = self.network.link_gdf["length"] / 1000  # Conversion to km
        for mode in self.config.handlers["volume_counts"]:
            file_name = "{}_volume_counts_{}.geojson".format(self.config.name, mode)
            file_path = os.path.join(self.config.output_path, file_name)
            gdf = geopandas.read_file(file_path)

            # Calculate VKT
            print(file_path)


# Dictionary used to map configuration string to post-processor type
POST_PROCESSOR_MAP = {"vkt": VKT}

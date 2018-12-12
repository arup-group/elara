from math import floor

import numpy as np
import pandas as pd


class Handler:
    def __init__(
        self, network, mode, handler_type="link", periods=24, scale_factor=1.0
    ):
        """
        Generic handler for events.
        :param network: Network object
        :param mode: Mode of transport string
        :param handler_type: Handler type ('link' or 'node')
        :param periods: Number of time periods per 24 hours
        :param scale_factor: Scenario run sample size
        """

        # Network attributes
        if handler_type == "link":
            self.elems = network.links
            self.elem_ids = network.link_ids
        elif handler_type == "node":
            self.elems = network.nodes
            self.elem_ids = network.node_ids
        else:
            raise Exception(
                "Unknown handler type encountered ({})".format(handler_type)
            )
        self.elem_indices = {
            key: value
            for (key, value) in zip(self.elem_ids, range(0, len(self.elem_ids)))
        }

        # Other attributes
        self.mode = mode
        self.periods = periods
        self.scale_factor = scale_factor

        # Initialise results storage
        self.node_gdf = network.node_gdf
        self.link_gdf = network.link_gdf
        self.result_gdfs = dict()  # Result geodataframes ready to export

    def table_position(self, elem_id, time):
        """
        Calculate the result table coordinates from a given a element ID and timestamp.
        :param elem_id: Element ID string
        :param time: Timestamp of event
        :return: (row, col) tuple to index results table
        """
        row = self.elem_indices[elem_id]
        col = floor(time / (86400.0 / self.periods)) % self.periods
        return row, col

    def vehicle_mode(self, vehicle_id):
        """
        Given a vehicle's ID, return its mode type.
        :param vehicle_id: Vehicle ID string
        :return: Vehicle mode type string
        """
        # TODO: Implement actual mode determination logic
        return "car"

    def finalise(self):
        """
        Transform accumulated results during event processing into final dataframes
        ready for exporting.
        """
        return NotImplementedError


class VolumeCounts(Handler):
    def __init__(self, network, mode, periods=24, scale_factor=1.0):
        super().__init__(network, mode, "link", periods, scale_factor)

        # Initialise volume count table
        self.counts = np.zeros((len(self.elem_indices), periods))

    def process_event(self, elem):
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
        :param elem: Event XML element
        """
        event_type = elem.get("type")
        if (event_type == "vehicle enters traffic") or (event_type == "entered link"):
            veh_mode = self.vehicle_mode(elem.get("vehicle"))
            if veh_mode == self.mode:
                link = elem.get("link")
                time = float(elem.get("time"))
                row, col = self.table_position(link, time)
                self.counts[row, col] += 1

    def finalise(self):
        """
        Following event processing, the raw events table will contain counts by link
        by time slice. The only thing left to do is scale by the sample size and
        create dataframes.
        """

        # Scale final counts
        self.counts *= 1.0 / self.scale_factor

        # Create volume counts output
        counts_df = pd.DataFrame(
            data=self.counts, index=self.elem_ids, columns=range(0, self.periods)
        ).sort_index()
        self.result_gdfs["volume_counts_{}".format(self.mode)] = self.link_gdf.join(
            counts_df, how="left"
        )

        # Create volume/capacity ratio output
        ratios_df = counts_df.divide(self.link_gdf["capacity"].values, axis=0).fillna(
            value=0
        )
        self.result_gdfs["vc_ratios_{}".format(self.mode)] = self.link_gdf.join(
            ratios_df, how="left"
        )


# Dictionary used to map configuration string to handler type
HANDLER_MAP = {"volume_counts": VolumeCounts}

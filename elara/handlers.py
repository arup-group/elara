from math import floor

import numpy as np
import pandas as pd


class Handler:
    def __init__(self, network, transit_vehicles, mode, periods=24, scale_factor=1.0):
        """
        Generic handler for events.
        :param network: Network object
        :param transit_vehicles: Transit vehicles object
        :param mode: Mode of transport string
        :param periods: Number of time periods per 24 hours
        :param scale_factor: Scenario run sample size
        """

        self.network = network
        self.transit_vehicles = transit_vehicles
        self.mode = mode
        self.periods = periods
        self.scale_factor = scale_factor

        # Initialise results storage
        self.result_gdfs = dict()  # Result geodataframes ready to export

    @staticmethod
    def generate_elem_ids(elem_gdf):
        """
        Generate element ID list and index dictionary from given geodataframe.
        :param elem_gdf: Element geodataframe
        :return: (element IDs, element indices)
        """
        elem_ids = elem_gdf.index.tolist()
        elem_indices = {
            key: value for (key, value) in zip(elem_ids, range(0, len(elem_ids)))
        }
        return elem_ids, elem_indices

    def vehicle_mode(self, vehicle_id):
        """
        Given a vehicle's ID, return its mode type.
        :param vehicle_id: Vehicle ID string
        :return: Vehicle mode type string
        """
        if vehicle_id in self.transit_vehicles.veh_id_veh_type_map.keys():
            return self.transit_vehicles.veh_type_mode_map[
                self.transit_vehicles.veh_id_veh_type_map[vehicle_id]
            ]
        else:
            return "car"

    def remove_empty_rows(self, df):
        """
        Remove rows from given results dataframe if time period columns only contain
        zeroes.
        :param df: Results dataframe
        :return: Contracted results dataframe
        """
        cols = [h for h in range(self.periods)]
        return df.loc[df[cols].sum(axis=1) > 0]

    def finalise(self):
        """
        Transform accumulated results during event processing into final dataframes
        ready for exporting.
        """
        return NotImplementedError

    def contract_results(self):
        """
        Remove zero-sum rows from all results dataframes.
        """
        self.result_gdfs = {
            k: self.remove_empty_rows(df) for (k, df) in self.result_gdfs.items()
        }


class VolumeCounts(Handler):
    def __init__(self, network, transit_vehicles, mode, periods=24, scale_factor=1.0):
        super().__init__(network, transit_vehicles, mode, periods, scale_factor)

        # Initialise element attributes
        self.elem_gdf = self.network.link_gdf
        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

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
                row, col = table_position(self.elem_indices, self.periods, link, time)
                self.counts[row, col] += 1

    def finalise(self):
        """
        Following event processing, the raw events table will contain counts by link
        by time slice. The only thing left to do is scale by the sample size and
        create dataframes.
        """

        # Overwrite the scale factor for public transport vehicles (these do not need to
        # be expanded.
        if self.mode != "car":
            self.scale_factor = 1.0

        # Scale final counts
        self.counts *= 1.0 / self.scale_factor

        # Create volume counts output
        counts_df = pd.DataFrame(
            data=self.counts, index=self.elem_ids, columns=range(0, self.periods)
        ).sort_index()
        self.result_gdfs["volume_counts_{}".format(self.mode)] = self.elem_gdf.join(
            counts_df, how="left"
        )

        # Create volume/capacity ratio output
        capacity_factor = 24 / self.periods
        ratios_df = counts_df.divide(
            self.elem_gdf["capacity"].values * capacity_factor, axis=0
        ).fillna(value=0)
        self.result_gdfs["vc_ratios_{}".format(self.mode)] = self.elem_gdf.join(
            ratios_df, how="left"
        )


class PassengerCounts(Handler):
    def __init__(self, network, transit_vehicles, mode, periods=24, scale_factor=1.0):
        super().__init__(network, transit_vehicles, mode, periods, scale_factor)

        # Initialise element attributes
        self.elem_gdf = self.network.link_gdf
        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

        # Initialise passenger count table
        self.counts = np.zeros((len(self.elem_indices), periods))

        # Initialise vehicle occupancy mapping
        self.veh_occupancy = dict()  # vehicle_id : occupancy

    def process_event(self, elem):
        """
        Iteratively aggregate 'PersonEntersVehicle' and 'PersonLeavesVehicle'
        events to determine passenger volumes by link.
        :param elem: Event XML element
        """
        event_type = elem.get("type")
        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                if self.veh_occupancy.get(veh_id, None) is None:
                    self.veh_occupancy[veh_id] = 1
                else:
                    self.veh_occupancy[veh_id] += 1
        elif event_type == "PersonLeavesVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                if (
                    self.veh_occupancy.get(veh_id, None) == 0
                    or self.veh_occupancy.get(veh_id, None) is None
                ):
                    pass
                else:
                    self.veh_occupancy[veh_id] -= 1
        elif event_type == "left link" or event_type == "vehicle leaves traffic":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.mode:
                # Increment link passenger volumes
                time = float(elem.get("time"))
                link = elem.get("link")
                row, col = table_position(self.elem_indices, self.periods, link, time)
                self.counts[row, col] += self.veh_occupancy.get(veh_id, 0)

    def finalise(self):
        """
        Following event processing, the raw events table will contain passenger
        counts by link by time slice. The only thing left to do is scale by the
        sample size and create dataframes.
        """

        # Scale final counts
        self.counts *= 1.0 / self.scale_factor

        # Create passenger counts output
        counts_df = pd.DataFrame(
            data=self.counts, index=self.elem_ids, columns=range(0, self.periods)
        ).sort_index()
        self.result_gdfs["passenger_counts_{}".format(self.mode)] = self.elem_gdf.join(
            counts_df, how="left"
        )


def table_position(elem_indices, periods, elem_id, time):
    """
    Calculate the result table coordinates from a given a element ID and timestamp.
    :param elem_indices: Element index list
    :param periods: Number of time periods across the day
    :param elem_id: Element ID string
    :param time: Timestamp of event
    :return: (row, col) tuple to index results table
    """
    row = elem_indices[elem_id]
    col = floor(time / (86400.0 / periods)) % periods
    return row, col


# Dictionary used to map configuration string to handler type
HANDLER_MAP = {"volume_counts": VolumeCounts, "passenger_counts": PassengerCounts}

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
        self.result_dfs = dict()  # Result dataframes ready to export to CSV

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

        # Save results in the result_dfs dictionary
        self.result_dfs["volume_counts_{}".format(self.mode)] = pd.DataFrame(
            data=self.counts, index=self.elem_ids, columns=range(0, self.periods)
        )


def safe_array_divide(a, b):
    """
    Safely divide two arrays, if divide by zero is encountered output zero.
    :param a: Numpy array 1
    :param b: Numpy array 2
    :return: Output numpy array
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        results = np.true_divide(a, b)
        results[results == np.inf] = 0
        results = np.nan_to_num(results)
    return results


# Dictionary used to map configuration string to handler type
HANDLER_MAP = {"volume_counts": VolumeCounts}

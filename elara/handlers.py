from math import floor

import numpy as np
import pandas as pd


class Handler:
    def __init__(
        self, network, mode, handler_type="link", periods=24, scale_factor=1.0
    ):
        """
        Generic handler for link events.
        :param network: Network object
        :param mode: Mode of transport string
        :param handler_type: Handler type ('link' or 'node')
        :param periods: Number of time periods per 24 hours
        :param scale_factor: Scenario run sample size
        """
        # Handler name
        self.name = None  # To be set by children

        # Network attributes
        if handler_type == "link":
            self.elems = network.links
        elif handler_type == "node":
            self.elems = network.nodes
        else:
            raise Exception(
                "Unknown handler type encountered ({})".format(handler_type)
            )
        self.elem_indices = {
            key: value
            for (key, value) in zip(self.elems.keys(), range(0, len(self.elems)))
        }

        # Other attributes
        self.mode = mode
        self.periods = periods
        self.scale_factor = scale_factor

        # Initialise results storage
        self.results = dict()  # Raw numpy array results
        self.final_tables = dict()  # Formatted pandas dataframes

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

    def generate_table(self, arr):
        """
        Generate a formatted pandas dataframe from raw numpy results with row
        and column labels.
        :param arr: Results array to be processed
        :return: Dataframe
        """
        return pd.DataFrame(data=arr, index=self.elems, columns=range(0, self.periods))

    def process_event(self, elem):
        """
        Process event function stub, to be overwritten with actual implementation
        in handler children.
        :param elem: XML element representing event
        """
        return NotImplementedError

    def generate_results(self):
        """
        Transform the numpy array created during event iteration into the actual
        results table, to be overwritten with actual implementation in handler
        children.
        """
        return NotImplementedError


class VolumeCounts(Handler):
    def __init__(self, network, mode, periods=24, scale_factor=1.0):
        super().__init__(network, mode, "link", periods, scale_factor)
        self.name = "volume_counts_{}".format(mode)
        self.results = {self.name: np.zeros((len(self.elem_indices), periods))}

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
                self.results[self.name][row, col] += 1

    def generate_results(self):
        """
        Following event processing, the raw events table will contain counts by link
        by time slice. The only thing left to do is scale by the sample size and
        create dataframes.
        """
        self.results[self.name] *= 1.0 / self.scale_factor
        self.final_tables[self.name] = self.generate_table(self.results[self.name])


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

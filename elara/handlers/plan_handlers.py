import numpy as np
from math import floor
import pandas as pd
import os

from elara.factory import Tool, WorkStation

__all__ = [
    'Activities',
    'Legs',
    'ModeShare',
]


class PlanHandler(Tool):
    results = dict()

    def __init__(self, config, option=None):
        super().__init__(config, option)

        # TODO
        if self.option != "all":
            raise NotImplementedError(f'Not implemented option: {self.option} for modeshare')


class Activities(PlanHandler):

    def __init__(self):
        raise NotImplementedError


class Legs(PlanHandler):
    def __init__(self):
        raise NotImplementedError


class ModeShare(PlanHandler):

    requirements = [
        'plans',
        'transit_schedule',
        'attributes',
        'mode_map',
        'mode_hierarchy'
    ]
    valid_options = ['all']

    def __init__(self, config, option=None):
        super().__init__(config, option)
        self.modes = None
        self.mode_indices = None
        self.classes = None
        self.class_indices = None
        self.activities = None
        self.activity_indices = None
        self.mode_counts = None
        self.results = None

        # Initialise results storage
        self.results = dict()  # Result geodataframes ready to export

    def build(self, resources):
        super().build(resources)

        # Initialise mode classes
        self.modes, self.mode_indices = self.generate_id_map(self.resources['plans'].modes)

        # Initialise class classes
        self.classes, self.class_indices = self.generate_id_map(
            self.resources['attributes'].classes
        )

        # Initialise activity classes
        self.activities, self.activity_indices = self.generate_id_map(
            self.resources['plans'].activities
        )
        # TODO - don't need to keep pt interactions or/for modeshare

        # Initialise mode count table
        self.mode_counts = np.zeros((len(self.modes),
                                    len(self.classes),
                                    len(self.activities),
                                    self.config.time_periods))

        self.results = dict()

    def process_plan(self, elem):

        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
        :param elem: Plan XML element
        """
        if elem.get('selected') == 'yes':

            ident = elem.getparent().get('id')
            attribute_class = self.resources['attributes'].map.get(ident, 'not found')

            end_time = None
            modes = []

            for stage in elem:

                if stage.tag == 'leg':
                    mode = stage.get('mode')
                    if mode == 'pt':
                        route = stage.xpath('route')[0].text.split('===')[-2]
                        mode = self.resources['transit_schedule'].mode_map.get(route)
                    modes.append(mode)

                elif stage.tag == 'activity':
                    activity = stage.get('type')

                    if activity == 'pt interaction':  # ignore pt interaction activities
                        continue

                    # only add activity modes when there has been previous activity
                    # (ie trip start time)
                    if end_time:
                        mode = self.resources['mode_hierarchy'].get(modes)
                        x, y, z, w = self.table_position(
                            mode,
                            attribute_class,
                            activity,
                            end_time
                        )

                        self.mode_counts[x, y, z, w] += 1

                    # update endtime for next activity
                    end_time = convert_time(stage.get('end_time'))

                    # reset modes
                    modes = []

    def finalise(self):
        """
        Following plan processing, the raw mode share table will contain counts by mode,
        population attribute class, activity and period (where period is based on departure
        time).
        Finalise aggregates these results as required and creates a dataframe.
        """

        # Scale final counts
        self.mode_counts *= 1.0 / self.config.scale_factor

        names = ['mode', 'class', 'activity', 'hour']
        indexes = [self.modes, self.classes, self.activities, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.mode_counts.flatten(), index=index)[0]

        # mode counts breakdown output
        counts_df = counts_df.unstack(level='mode').sort_index()
        key = "mode_counts_{}_breakdown".format(self.option)
        self.results[key] = counts_df

        # mode counts totals output
        total_counts_df = counts_df.sum(0)
        key = "mode_counts_{}_total".format(self.option)
        self.results[key] = total_counts_df

        # convert to mode shares
        total = self.mode_counts.sum()

        # mode shares breakdown output
        key = "mode_shares_{}_breakdown".format(self.option)
        self.results[key] = counts_df / total

        # mode shares totals output
        key = "mode_shares_{}_total".format(self.option)
        self.results[key] = total_counts_df / total

    @staticmethod
    def generate_id_map(list_in):
        """
        Generate element ID list and index dictionary from given list.
        :param list_in: list
        :return: (list, list_indices_map)
        """
        list_indices_map = {
            key: value for (key, value) in zip(list_in, range(0, len(list_in)))
        }
        return list_in, list_indices_map

    def table_position(
        self,
        mode,
        attribute_class,
        activity,
        time
    ):
        """
        Calculate the result table coordinates from given maps.
        :param mode: String, mode id
        :param attribute_class: String, class id
        :param activity: String, activity id
        :param time: Timestamp of event
        :return: (x, y, z, w) tuple of integers to index results table
        """
        x = self.mode_indices[mode]
        y = self.class_indices[attribute_class]
        z = self.activity_indices[activity]
        w = floor(time / (86400.0 / self.config.time_periods)) % self.config.time_periods
        return x, y, z, w


class PlanHandlerStation(WorkStation):

    tools = {
        # "activities": Activities,
        # "legs": Legs,
        "modeshare": ModeShare,
    }

    def build(self):
        # build tools
        super().build()

        # iterate through plans
        plans = self.supplier_resources['plans']
        for i, plan in enumerate(plans.elems):
            for event_handler in self.resources.values():
                event_handler.process_plan(plan)

        # finalise
        # Generate event file outputs
        for handler in self.resources.values():
            handler.finalise()
            # if self.config.contract:
            #     handler.contract_results()

            for name, result in handler.results.items():
                csv_name = "{}_{}.csv".format(self.config.name, name)
                csv_path = os.path.join(self.config.output_path, csv_name)

                # File exports
                result.to_csv(csv_path, header=True)


def convert_time(t: str) -> int:
    """
    Convert MATSim output plan times into seconds

    :param t: MATSim str time
    :return: seconds int
    """
    if not t:
        return None
    t = t.split(":")
    return ((int(t[0]) * 60) + int(t[1])) * 60 + int(t[2])


def export_geojson(gdf, path):
    """
    Given a geodataframe, export geojson representation to specified path.
    :param gdf: Input geodataframe
    :param path: Output path
    """
    with open(path, "w") as file:
        file.write(gdf.to_json())

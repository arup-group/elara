import numpy as np
from math import floor
import pandas as pd

class ModeShare:

    def __init__(
        self,
        act,
        plans,
        transit_schedule,
        attributes,
        time_periods=24,
        scale_factor=1.0,

    ):
        self.act = act
        if self.act != "all":
            raise NotImplementedError(f'Not implemented filtering with {self.act} for modeshare')

        self.transit_schedule = transit_schedule
        self.attributes = attributes
        self.periods = time_periods
        self.scale_factor = scale_factor

        # Initialise mode classes
        self.modes, self.mode_indices = self.generate_id_map(plans.modes)

        # Initialise class classes
        self.classes, self.class_indices = self.generate_id_map(attributes.classes)

        # Initialise activity classes
        self.activities, self.activity_indices = self.generate_id_map(plans.activities)

        # Initialise mode count table
        self.mode_counts = np.zeros((len(self.modes),
                                    len(self.classes),
                                    len(self.activities),
                                    time_periods))

        self.results = dict()

        self.hierarchy = [
            'ferry',
            'rail',
            'tram',
            'bus',
            'car',
            'bike',
            'walk',
            'transit_walk'
        ]

    def process_event(self, elem):
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
        :param elem: Plan XML element
        """
        if elem.get('selected') == 'yes':

            ident = elem.getparent().get('id')
            attribute_class = self.attributes.map.get(ident, 'missing')

            end_time = None
            modes = []

            for stage in elem:

                if stage.tag == 'leg':
                    mode = stage.get('mode')
                    if mode == 'pt':
                        route = stage.xpath('route')[0].text.split('===')[-2]
                        mode = self.transit_schedule.mode_map.get(route)
                    modes.append(mode)

                elif stage.tag == 'activity':
                    activity = stage.get('type')

                    if activity == 'pt interaction':  # ignore pt interaction activities
                        continue

                    # only add activity modes when there has been previous activity (ie trip start time)
                    if end_time:
                        mode = self.select_hierarchy(modes)
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
        self.mode_counts *= 1.0 / self.scale_factor

        names = ['mode', 'class', 'activity', 'hour']
        indexes = [self.modes, self.classes, self.activities, range(self.periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.mode_counts.flatten(), index=index)[0]

        # mode counts breakdown output
        counts_df = counts_df.unstack(level='mode').sort_index()
        key = "mode_counts_{}_breakdown".format(self.act)
        self.results[key] = counts_df

        # mode counts totals output
        total_counts_df = counts_df.sum(0)
        key = "mode_counts_{}_total".format(self.act)
        self.results[key] = total_counts_df

        # convert to mode shares
        total = self.mode_counts.sum()

        # mode shares breakdown output
        key = "mode_shares_{}_breakdown".format(self.act)
        self.results[key] = counts_df / total

        # mode shares totals output
        key = "mode_shares_{}_total".format(self.act)
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

    def select_hierarchy(self, modes):
        for h in self.hierarchy:
            for mode in modes:
                if h == mode:
                    return mode
        raise LookupError(f'not found modes in mode [{modes}] hierarchy')

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
        w = floor(time / (86400.0 / self.periods)) % self.periods
        return x, y, z, w


def convert_time(t):
    if not t:
        return None
    t = t.split(":")
    return ((int(t[0]) * 60) + int(t[1])) * 60 + int(t[2])


PLAN_HANDLER_MAP = {
    "mode_share": ModeShare,
}

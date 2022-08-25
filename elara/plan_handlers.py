import numpy as np
from math import floor
import pandas as pd
import geopandas
from datetime import datetime, timedelta
from typing import Optional
import logging
import json
import uuid

from elara.factory import Tool, WorkStation


class PlanHandlerTool(Tool):
    """
    Base Tool class for Plan Handling.
    """
    options_enabled = True

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

    def build(
            self,
            resources: dict,
            write_path=None) -> None:

        super().build(resources, write_path)

    def extract_mode_from_route_elem(self, leg_mode, route_elem):
        """
        Extract mode and route identifieers from a MATSim route xml element.
        """
        if self.config.version == 11:
            if leg_mode == "pt":
                return self.extract_mode_from_v11_route_elem(route_elem)
            return leg_mode

        return leg_mode

    def extract_mode_from_v11_route_elem(self, route_elem):
        """
        Extract mode and route identifieers from a MATSim v11 route xml element.
        """
        route = route_elem.text.split('===')[-2]
        mode = self.resources['transit_schedule'].route_to_mode_map.get(route)
        return mode

    def extract_routeid_from_v12_route_elem(self, route_elem):
        """
        Extract mode and route identifieers from a MATSim v11 route xml element.
        """
        route_dict = json.loads(route_elem.text.strip())
        route = route_dict["transitRouteId"]
        return route

    @staticmethod
    def get_furthest_mode(modes):
        """
        Return key with greatest value. Note that in the case of join max, the first is returned only.
        """
        if len(modes) > 2 and 'transit_walk' in modes:
            del modes['transit_walk']
        return max(modes, key=modes.get)

    @staticmethod
    def generate_id_map(list_in):
        """
        Generate element ID list and index dictionary from given list.
        :param list_in: list
        :return: (list, list_indices_map)
        """
        if not len(set(list_in)) == len(list_in):
            raise UserWarning("non unique mode list found")

        list_indices_map = {
            key: value for (key, value) in zip(list_in, range(0, len(list_in)))
        }
        return list_in, list_indices_map

    @staticmethod
    def generate_indices_map(list_in):
        """
        Generate element ID list and index dictionary from given list.
        :param list_in: list
        :return: (list, list_indices_map)
        """
        list_indices_map = {
            key: value for (key, value) in zip(list_in, range(0, len(list_in)))
        }
        return list_in, list_indices_map

    def mode_table_position(
        self,
        mode,
        attribute_class,
        time
    ):
        """
        Calculate the result table coordinates from given maps.
        :param mode: String, mode id
        :param attribute_class: String, class id
        :param time: Timestamp of event
        :return: (x, y, z) tuple of integers to index results table
        """
        x = self.mode_indices[mode]
        y = self.class_indices[attribute_class]
        z = floor(time / (86400.0 / self.config.time_periods)) % self.config.time_periods
        return x, y, z


class ModeShares(PlanHandlerTool):
    """
    Extract Mode Share from Plans.
    """

    requirements = [
        'plans',
        'attributes',
        'transit_schedule',
        'output_config',
    ]
    valid_modes = ['all']

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        """
        Initiate Handler.
        :param config: Config
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode  # todo options not implemented
        self.groupby_person_attribute = groupby_person_attribute
        self.destination_activity_filters = kwargs.get("destination_activity_filters")

        self.modes = None
        self.mode_indices = None
        self.classes = None
        self.class_indices = None
        self.mode_counts = None
        self.results = None

        # Initialise results storage
        self.results = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """
        super().build(resources, write_path=write_path)

        modes = list(set(self.resources['output_config'].modes + self.resources['transit_schedule'].modes))
        self.logger.debug(f'modes = {modes}')

        # Initialise mode classes
        self.modes, self.mode_indices = self.generate_id_map(modes)

        if self.groupby_person_attribute:
            self.attributes = self.resources["attributes"]
            availability = self.attributes.attribute_key_availability(self.groupby_person_attribute)
            self.logger.debug(f'availability of attribute {self.groupby_person_attribute} = {availability*100}%')
            if availability < 1:
                self.logger.warning(f'availability of attribute {self.groupby_person_attribute} = {availability*100}%')
            found_attributes = self.resources['attributes'].attribute_values(self.groupby_person_attribute) | {None}
        else:
            self.attributes = {}
            found_attributes = [None]
        self.logger.debug(f'attributes = {found_attributes}')

        # Initialise class classes
        self.classes, self.class_indices = self.generate_id_map(found_attributes)

        # Initialise mode count table
        self.mode_counts = np.zeros((
            len(self.modes),
            len(self.classes),
            self.config.time_periods))

        self.results = dict()

    def finalise(self):
        """
        Following plan processing, the raw mode share table will contain counts by mode,
        population attribute class, activity and period (where period is based on departure
        time).
        Finalise aggregates these results as required and creates a dataframe.
        """

        # Scale final counts
        self.mode_counts *= 1.0 / self.config.scale_factor

        names = ['mode', 'class', 'hour']
        indexes = [self.modes, self.classes, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.mode_counts.flatten(), index=index)
        counts_df.index = pd.MultiIndex.from_frame(
            counts_df.index.to_frame().fillna("None")
        )
        counts_df.columns = ["count"]
        key = f"{self.name}_detailed_counts"
        self.results[key] = counts_df

        # mode counts totals by attribute
        if self.groupby_person_attribute:
            total_grouped_counts_df = counts_df.groupby(["mode", "class"]).sum()
            key = f"{self.name}_{self.groupby_person_attribute}_counts"
            self.results[key] = total_grouped_counts_df

        # mode counts totals output
        total_counts_df = counts_df.groupby(["mode"]).sum()
        key = f"{self.name}_counts"
        self.results[key] = total_counts_df

        # convert to mode shares
        total = self.mode_counts.sum()

        # mode shares breakdown output
        key = f"{self.name}_detailed_shares"
        shares_df = counts_df / total
        shares_df.columns = ["share"]
        self.results[key] = shares_df

        # mode shares totals by attribute
        if self.groupby_person_attribute:
            total_grouped_shares_df = shares_df.groupby(["mode", "class"]).sum()
            key = f"{self.name}_{self.groupby_person_attribute}_shares"
            self.results[key] = total_grouped_shares_df

        # mode shares totals output
        key = f"{self.name}_shares"
        total_shares_df = shares_df.groupby(["mode"]).sum()
        self.results[key] = total_shares_df


class TripModes(ModeShares):

    def process_plans(self, elem):
        """
        :param elem: Plan XML element
        """
        for plan in elem.xpath(".//plan"):
            if plan.get('selected') != 'no':

                ident = elem.get('id')
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute)

                end_time = None
                trip_modes = {}

                for stage in plan:

                    if stage.tag == 'leg':
                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)
                        distance = float(route_elem.get("distance", 0))
                        # ignore access and egress walk
                        mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)
                        trip_modes[mode] = trip_modes.get(mode, 0) + distance

                    elif stage.tag == 'activity':
                        if stage.get('type') == 'pt interaction':  # ignore pt interaction activities
                            continue

                        # only add activity modes when there has been previous activity
                        # (ie trip start time)
                        if end_time:
                            mode = self.get_furthest_mode(trip_modes)
                            x, y, z = self.mode_table_position(
                                mode,
                                attribute_class,
                                end_time
                            )

                            self.mode_counts[x, y, z] += 1

                        # update endtime for next activity
                        end_time = convert_time_to_seconds(stage.get('end_time'))

                        # reset modes
                        trip_modes = {}


class PlanModes(ModeShares):

    def process_plans(self, elem):
        """
        """
        for plan in elem.xpath(".//plan"):
            if plan.get('selected') != 'no':

                ident = elem.get('id')
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute)

                plan_modes = {}

                for stage in plan:

                    if stage.tag == 'leg':
                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)
                        distance = float(route_elem.get("distance", 0))
                        # ignore access and egress walk
                        mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)
                        plan_modes[mode] = plan_modes.get(mode, 0) + distance

                if plan_modes: # stay-home agents have no legs/modes
                    mode = self.get_furthest_mode(plan_modes)
                    x, y, z = self.mode_table_position(
                        mode,
                        attribute_class,
                        0
                    )

                    self.mode_counts[x, y, z] += 1


class TripActivityModes(ModeShares):
    """
    Extract mode shares for specified activities from plans.
    This handler takes a list of activities and computes the mode shares for each independent activity trip.
    """
    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        if not kwargs.get("destination_activity_filters"):
            raise UserWarning(
                "You must configure 'destination_activity_filters' for the 'trip_activity_modes' tool, eg '['work']."
                )
        if not isinstance(kwargs.get("destination_activity_filters"), list):
            raise UserWarning(
                "You must configure 'destination_activity_filters' for the 'trip_activity_modes' tool as a list, eg '['work']."
                )

    def process_plans(self, elem):
        """
        Iterate through the plans and produce counts / mode shares for trips to
        the destination activities specified in the list.

        This handler counts the longest travelling mode of the trip leg leading to each instance of the
        destination activity(ies) specified

        e.g. if the destination acitivity list consists of ['work_a', work_b]
        and a plan consists of the trips [home] --> (bus,11km) --> [work_a] --> (train, 10km) --> [work_b],
        the resulting counts will see: (bus) +1 & (train) + 1.
        :param elem: Plan XML element
        """
        for plan in elem.xpath(".//plan"):
            if plan.get('selected') != 'no':

                ident = elem.get('id')
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute)

                end_time = None
                trip_modes = {}

                for stage in plan:
                    if stage.tag == 'leg':
                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)
                        distance = float(route_elem.get("distance", 0))
                        # ignore access and egress walk
                        mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)
                        trip_modes[mode] = trip_modes.get(mode, 0) + distance

                    elif stage.tag == 'activity':
                        activity = stage.get('type')

                        # only add activity modes when there has been previous activity
                        # (ie trip start time) AND the activity is in specified list
                        if end_time:
                            if activity in self.destination_activity_filters:
                                mode = self.get_furthest_mode(trip_modes)
                                x, y, z = self.mode_table_position(
                                    mode,
                                    attribute_class,
                                    end_time
                                )
                                self.mode_counts[x, y, z] += 1
                                # reset modes
                                trip_modes = {}
                        if not activity == 'pt interaction':  # reset modes at end of trip
                            trip_modes = {}
                            # update endtime for next activity
                            end_time = convert_time_to_seconds(stage.get('end_time'))


class PlanActivityModes(ModeShares):

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        if not kwargs.get("destination_activity_filters"):
            raise UserWarning(
                "You must configure 'destination_activity_filters' for the 'plan_activity_modes' tool, eg '['work']."
                )
        if not isinstance(kwargs.get("destination_activity_filters"), list):
            raise UserWarning(
                "You must configure 'destination_activity_filters' for the 'plan_activity_modes' tool as a list, eg '['work']."
                )

    def process_plans(self, elem):
        """
        """
        for plan in elem.xpath(".//plan"):
            if plan.get('selected') != 'no':

                ident = elem.get('id')
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute)

                end_time = None
                trip_modes = {}
                plan_modes = {}

                for stage in plan:
                    if stage.tag == 'leg':
                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)
                        distance = float(route_elem.get("distance", 0))
                        # ignore access and egress walk
                        mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)
                        trip_modes[mode] = trip_modes.get(mode, 0) + distance

                    elif stage.tag == 'activity':
                        activity = stage.get('type')

                        # only add activity modes when there has been previous activity
                        # (ie trip start time) AND the activity is in specified list
                        if end_time:
                            if activity in self.destination_activity_filters:
                                # add modes and distances to plan_modes
                                for mode, distance in trip_modes.items():
                                    plan_modes[mode] = plan_modes.get(mode, 0) + distance
                            # reset modes
                            trip_modes = {}
                        if not activity == 'pt interaction':  # reset modes at end of trip
                            trip_modes = {}
                            # update endtime for next activity
                            end_time = convert_time_to_seconds(stage.get('end_time'))

                if plan_modes:
                    mode = self.get_furthest_mode(plan_modes)
                    x, y, z = self.mode_table_position(
                        mode,
                        attribute_class,
                        0
                    )
                    self.mode_counts[x, y, z] += 1


class LegLogs(PlanHandlerTool):

    requirements = ['plans', 'transit_schedule', 'attributes']
    valid_modes = ['all']

    # todo make it so that 'all' option not required (maybe for all plan handlers)

    """
    Note that MATSim plan output plans display incorrect 'dep_time' (they show departure time of
    original init plan) and do not show activity start time. As a result, we use leg 'duration'
    to calculate the start of the next activity. This results in time waiting to enter
    first link as being activity time. Therefore activity durations are slightly over reported
    and leg duration under reported.
    """

    def __init__(self, config, mode="all", groupby_person_attribute="subpopulation", **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        :param attributes: str, attribute key defaults to 'subpopulation'
        """

        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.groupby_person_attribute = groupby_person_attribute
        self.start_datetime = datetime.strptime("2020:4:1-00:00:00", '%Y:%m:%d-%H:%M:%S')

        self.activities_log = None
        self.legs_log = None

        # Initialise results storage
        self.results = dict()  # Result dataframes ready to export

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """
        super().build(resources, write_path=write_path)

        self.attributes = self.resources["attributes"]

        activity_csv_name = f"{self.name}_activities.csv"
        legs_csv_name = f"{self.name}_legs.csv"

        self.activities_log = self.start_csv_chunk_writer(activity_csv_name, write_path=write_path, compression=self.compression)
        self.legs_log = self.start_csv_chunk_writer(legs_csv_name, write_path=write_path, compression=self.compression)

    def process_plans(self, elem):

        """
        Build list of leg and activity logs (dicts) for each selected plan.

        Note that this assumes that first stage of a plan is ALWAYS an activity.
        Note that activity wrapping is not dealt with here.

        :return: Tuple[List[dict]]
        """
        ident = elem.get('id')

        for plan in elem.xpath(".//plan"):

            if plan.get('selected') != 'no':

                activities = []
                legs = []

                attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)

                leg_seq_idx = 0
                trip_seq_idx = 0
                act_seq_idx = 0

                arrival_dt = self.start_datetime
                activity_end_dt = None
                x = None
                y = None

                for stage in plan:

                    if stage.tag == 'activity':
                        act_seq_idx += 1

                        act_type = stage.get('type')

                        if act_type != 'pt interaction' or stage.get('end_time'):
                            end_time_str = stage.get('end_time', '23:59:59')
                            activity_end_dt = matsim_time_to_datetime(
                                arrival_dt, end_time_str, self.logger, idx=ident
                            )
                        else:
                            activity_end_dt = arrival_dt # zero duration for pt interactions without an end_time attribute

                        if act_type != 'pt interaction':
                            trip_seq_idx += 1  # increment for a new trip idx
                        
                        duration = activity_end_dt - arrival_dt  

                        x = stage.get('x')
                        y = stage.get('y')

                        activities.append(
                            {
                                'agent': ident,
                                'attribute': attribute,
                                'seq': act_seq_idx,
                                'act': act_type,
                                'x': x,
                                'y': y,
                                'start': arrival_dt.time(),
                                'end': activity_end_dt.time(),
                                'end_day': activity_end_dt.day,
                                # 'duration': duration,
                                'start_s': self.get_seconds(arrival_dt),
                                'end_s': self.get_seconds(activity_end_dt),
                                'duration_s': duration.total_seconds()
                            }
                        )

                    elif stage.tag == 'leg':
                        leg_seq_idx += 1

                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)
                        mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)

                        trav_time = stage.get('trav_time')
                        h, m, s = trav_time.split(":")
                        td = timedelta(hours=int(h), minutes=int(m), seconds=int(s))

                        arrival_dt = activity_end_dt + td

                        legs.append(
                            {
                                'agent': ident,
                                'attribute': attribute,
                                'seq': leg_seq_idx,
                                'trip': trip_seq_idx,
                                'mode': mode,
                                'ox': x,
                                'oy': y,
                                'dx': None,
                                'dy': None,
                                'o_act': act_type,
                                'd_act': None,
                                'start': activity_end_dt.time(),
                                'end': arrival_dt.time(),
                                'end_day': arrival_dt.day,
                                # 'duration': td,
                                'start_s': self.get_seconds(activity_end_dt),
                                'end_s': self.get_seconds(arrival_dt),
                                'duration_s': td.total_seconds(),
                                'distance': route_elem.get('distance')
                            }
                        )

                for idx, leg in enumerate(legs):
                    # back-fill leg destinations
                    leg['dx'] = activities[idx + 1]['x']
                    leg['dy'] = activities[idx + 1]['y']
                    # back-fill destination activities for legs
                    leg['d_act'] = activities[idx + 1]['act']

                self.activities_log.add(activities)
                self.legs_log.add(legs)

    def finalise(self):
        """
        Finalise aggregates and joins these results as required and creates a dataframe.
        """
        self.activities_log.finish()
        self.legs_log.finish()

    @staticmethod
    def get_seconds(dt: datetime) -> int:
        """
        Extract time of day in seconds from datetime.
        :param dt: datetime
        :return: int, seconds
        """
        d = dt.day
        h = dt.hour
        m = dt.minute
        s = dt.second
        return s + (60 * (m + (60 * (h + ((d-1) * 24)))))


class TripLogs(PlanHandlerTool):

    requirements = ['plans', 'transit_schedule', 'attributes']
    # mode and purpose options need to be enabled for post-processing cross tabulation w euclidian distance
    valid_modes = ['all']

    # TODO make it so that 'all' option not required (maybe for all plan handlers)

    """
    Note that MATSim plan output plans display incorrect 'dep_time' (they show departure time of
    original init plan) and do not show activity start time. As a result, we use leg 'duration'
    to calculate the start of the next activity. This results in time waiting to enter
    first link as being activity time. Therefore activity durations are slightly over reported
    and leg duration under reported.
    """

    def __init__(self, config, mode="all", groupby_person_attribute="subpopulation", **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        :param attributes: str, attribute key defaults to 'subpopulation'
        """

        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.groupby_person_attribute = groupby_person_attribute
        self.start_datetime = datetime.strptime("2020:4:1-00:00:00", '%Y:%m:%d-%H:%M:%S')

        self.activities_log = None
        self.trips_log = None

        # Initialise results storage
        self.results = dict()  # Result dataframes ready to export

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """
        super().build(resources, write_path=write_path)

        self.attributes = self.resources["attributes"]

        activity_csv_name = f"{self.name}_activities.csv"
        trips_csv_name = f"{self.name}_trips.csv"

        self.activities_log = self.start_csv_chunk_writer(activity_csv_name, write_path=write_path, compression=self.compression)
        self.trips_log = self.start_csv_chunk_writer(trips_csv_name, write_path=write_path, compression=self.compression)

    def process_plans(self, elem):

        """
        Build list of trip and activity logs (dicts) for each selected plan.

        Note that this assumes that first stage of a plan is ALWAYS an activity.
        Note that activity wrapping is not dealt with here.

        :return: Tuple[List[dict]]
        """
        ident = elem.get('id')

        for plan in elem.xpath(".//plan"):

            attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)

            if plan.get('selected') != 'no':

                # check that plan starts with an activity
                if not plan[0].tag == 'activity':
                    raise UserWarning('Plan does not start with activity.')
                if plan[0].get('type') == 'pt interaction':
                    raise UserWarning('Plan cannot start with activity type "pt interaction".')

                activities = []
                trips = []
                act_seq_idx = 0

                activity_start_dt = self.start_datetime
                activity_end_dt = self.start_datetime
                # todo replace this start datetime with a real start datetime using config

                x = None
                y = None
                modes = {}
                trip_distance = 0

                for stage in plan:

                    if stage.tag == 'activity':
                        act_type = stage.get('type')

                        if not act_type == 'pt interaction':

                            act_seq_idx += 1  # increment for a new trip idx
                            trip_duration = activity_start_dt - activity_end_dt

                            end_time_str = stage.get('end_time', '23:59:59')

                            activity_end_dt = matsim_time_to_datetime(
                                activity_start_dt, end_time_str, self.logger, idx=ident
                            )

                            activity_duration = activity_end_dt - activity_start_dt

                            x = stage.get('x')
                            y = stage.get('y')

                            if modes:  # add to trips log

                                trips.append(
                                    {
                                        'agent': ident,
                                        'attribute': attribute,
                                        'seq': act_seq_idx-1,
                                        'mode': self.get_furthest_mode(modes),
                                        'ox': activities[-1]['x'],
                                        'oy': activities[-1]['y'],
                                        'dx': x,
                                        'dy': y,
                                        'o_act': activities[-1]['act'],
                                        'd_act': act_type,
                                        'start': activities[-1]['end'],
                                        'start_day': activities[-1]['end_day'],
                                        'end': activity_start_dt.time(),
                                        'end_day': activity_start_dt.day,
                                        'start_s': activities[-1]['end_s'],
                                        'end_s': self.get_seconds(activity_start_dt),
                                        'duration': trip_duration,
                                        'duration_s': trip_duration.total_seconds(),
                                        'distance': trip_distance,
                                    }
                                )

                                modes = {}  # reset for next trip
                                trip_distance = 0  # reset for next trip

                            activities.append(
                                {
                                    'agent': ident,
                                    'attribute': attribute,
                                    'seq': act_seq_idx,
                                    'act': act_type,
                                    'x': x,
                                    'y': y,
                                    'start': activity_start_dt.time(),
                                    'start_day': activity_start_dt.day,
                                    'end': activity_end_dt.time(),
                                    'end_day': activity_end_dt.day,
                                    'start_s': self.get_seconds(activity_start_dt),
                                    'end_s': self.get_seconds(activity_end_dt),
                                    'duration': activity_duration,
                                    'duration_s': activity_duration.total_seconds()
                                }
                            )

                            activity_start_dt = activity_end_dt

                        # if a 'pt interaction' activity has duration (ie it has an 'end_time' attribute)
                        # then advance the next activity start time accordingly   
                        elif stage.get('end_time'):
                            end_time_str = stage.get('end_time')

                            activity_start_dt = matsim_time_to_datetime(
                                activity_start_dt, end_time_str, self.logger, idx=ident
                            )

                    elif stage.tag == 'leg':

                        leg_mode = stage.get('mode')

                        # check for route elements. these are used to infer modes when analyzing output plans
                        # routes do not exist when analysing input plans (except when they are also simulation outputs)
                        route_elem = stage.xpath('route')
                        if route_elem:  # is not [], use route info
                            route_elem = route_elem[0]

                            distance = float(route_elem.get("distance", 0))

                            mode = self.extract_mode_from_route_elem(leg_mode, route_elem)

                            mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)
                            trip_distance += distance
                        else:  # use leg info
                            mode = leg_mode
                            distance = 0  # don't know distances for unrouted trips

                        # update mode dictionary with leg or route information
                        modes[mode] = modes.get(mode, 0) + distance

                        trav_time = stage.get('trav_time')
                        h, m, s = trav_time.split(":")
                        td = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                        activity_start_dt += td

                self.activities_log.add(activities)
                self.trips_log.add(trips)

    def finalise(self):
        """
        Finalise aggregates and joins these results as required and creates a dataframe.
        """
        self.activities_log.finish()
        self.trips_log.finish()

    @staticmethod
    def get_seconds(dt: datetime) -> int:
        """
        Extract time of day in seconds from datetime.
        :param dt: datetime
        :return: int, seconds
        """
        d = dt.day
        h = dt.hour
        m = dt.minute
        s = dt.second
        return s + (60 * (m + (60 * (h + ((d-1) * 24)))))

class SeeTripLogs(PlanHandlerTool):
    
    requirements = ['plans', 'transit_schedule', 'attributes']
    valid_modes = ['all']

    # todo make it so that 'all' option not required (maybe for all plan handlers)

    """
    Note that MATSim plan output plans display incorrect 'dep_time' (they show departure time of 
    original init plan) and do not show activity start time. As a result, we use leg 'duration' 
    to calculate the start of the next activity. This results in time waiting to enter 
    first link as being activity time. Therefore activity durations are slightly over reported 
    and leg duration under reported.
    """

    def __init__(self, config, mode="all", groupby_person_attribute="subpopulation",see=None, **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        :param attributes: str, attribute key defaults to 'subpopulation'
        """

        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.groupby_person_attribute = groupby_person_attribute
        self.start_datetime = datetime.strptime("2020:4:1-00:00:00", '%Y:%m:%d-%H:%M:%S')

        self.see_trips_log = None
        self.see = see

        # Initialise results storage
        self.results = dict()  # Result dataframes ready to export
        self.results['SeeAllPlansGdf'] = geopandas.GeoDataFrame()
        self.results['SeeUnSelectedPlansCarSelectedGdf'] = geopandas.GeoDataFrame()


    def build(self, resources: dict, write_path=None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """

        print("""\                                                                                                                                  
                                                                                                                                  
  .--.--.       ,---,.    ,---,.           ,---,                                ,--,                                              
 /  /    '.   ,'  .' |  ,'  .' |          '  .' \                             ,--.'|                          ,--,                
|  :  /`. / ,---.'   |,---.'   |         /  ;    '.          ,---,            |  | :                        ,--.'|                
;  |  |--`  |   |   .'|   |   .'        :  :       \     ,--. /  |           :  : '              .--.--.   |  |,      .--.--.    
|  :  ;_    :   :  |-,:   :  |-,        :  |   /\   \   ,--.'|'   |  ,--.--.  |  ' |        .--, /  /    '  `--'_     /  /    '   
 \  \    `. :   |  ;/|:   |  ;/|        |  :  ' ;.   : |   |  ,"' | /       \ '  | |      /_ ./||  :  /`./  ,' ,'|   |  :  /`./   
  `----.   \|   :   .'|   :   .'        |  |  ;/  \   \|   | /  | |.--.  .-. ||  | :   , ' , ' :|  :  ;_    '  | |   |  :  ;_     
  __ \  \  ||   |  |-,|   |  |-,        '  :  | \  \ ,'|   | |  | | \__\/: . .'  : |__/___/ \: | \  \    `. |  | :    \  \    `.  
 /  /`--'  /'   :  ;/|'   :  ;/|        |  |  '  '--'  |   | |  |/  ," .--.; ||  | '.'|.  \  ' |  `----.   \'  : |__   `----.   \ 
'--'.     / |   |    \|   |    \        |  :  :        |   | |--'  /  /  ,.  |;  :    ; \  ;   : /  /`--'  /|  | '.'| /  /`--'  / 
  `--'---'  |   :   .'|   :   .'        |  | ,'        |   |/     ;  :   .'   \  ,   /   \  \  ;'--'.     / ;  :    ;'--'.     /  
            |   | ,'  |   | ,'          `--''          '---'      |  ,     .-./---`-'     :  \  \ `--'---'  |  ,   /   `--'---'   
            `----'    `----'                                       `--`---'                \  ' ;            ---`-'               
                                                                                            `--`                                  """)

        super().build(resources, write_path=write_path)

        self.attributes = self.resources["attributes"]
        see_trips_csv_name = f"{self.name}_see_trips.csv"

        # writes the SEE specific trips log 
        self.see_trips_log = self.start_csv_chunk_writer(see_trips_csv_name, write_path=write_path)

    def process_plans(self, elem):

        """
        Build list of trip and activity logs (dicts) for all plans.

        Note that this uses ALL plans, selected and unselected (for subsequent SEE analysis).
        Note that this assumes that first stage of a plan is ALWAYS an activity.
        Note that activity wrapping is not dealt with here.

        :return: Tuple[List[dict]]
        """
        ident = elem.get('id')

        summary = []

        for plan in elem.xpath(".//plan"):

            attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)

            # check that plan starts with an activity
            if not plan[0].tag == 'activity':
                raise UserWarning('Plan does not start with activity.')
            if plan[0].get('type') == 'pt interaction':
                raise UserWarning('Plan cannot start with activity type "pt interaction".')

            activities = []
            trips = []
            act_seq_idx = 0

            activity_start_dt = self.start_datetime
            activity_end_dt = self.start_datetime
            # todo replace this start datetime with a real start datetime using config

            x = None
            y = None
            modes = {}
            trip_distance = 0

            for stage in plan:

                # we give each plan an id (to permit comparisons)
                innovation_hash = str(uuid.uuid4())
            
                if stage.tag == 'activity':
                    act_type = stage.get('type')

                    if not act_type == 'pt interaction':

                        act_seq_idx += 1  # increment for a new trip idx
                        trip_duration = activity_start_dt - activity_end_dt

                        end_time_str = stage.get('end_time', '23:59:59')

                        activity_end_dt = matsim_time_to_datetime(
                                activity_start_dt, end_time_str, self.logger, idx=ident
                        )

                        activity_duration = activity_end_dt - activity_start_dt

                        x = stage.get('x')
                        y = stage.get('y')

                        if modes:  # add to trips log

                            trips.append(
                                {
                                    'agent': ident,
                                    'attribute': attribute,
                                    'seq': act_seq_idx-1,
                                    'mode': self.get_furthest_mode(modes),
                                    'ox': float(activities[-1]['x']),
                                    'oy': float(activities[-1]['y']),
                                    'dx': float(x),
                                    'dy': float(y),
                                    'o_act': activities[-1]['act'],
                                    'd_act': act_type,
                                    'start': activities[-1]['end'],
                                    'start_day': activities[-1]['end_day'],
                                    'end': activity_start_dt.time(),
                                    'end_day': activity_start_dt.day,
                                    'start_s': activities[-1]['end_s'],
                                    'end_s': self.get_seconds(activity_start_dt),
                                    'duration': trip_duration,
                                    'duration_s': trip_duration.total_seconds(),
                                    'distance': trip_distance,
                                    "utility" : float(plan.get("score")),
                                    "selected" : plan.get("selected"),
                                    "innovation_hash" : innovation_hash
                                }
                            )

                            modes = {}  # reset for next trip
                            trip_distance = 0  # reset for next trip

                        activities.append(
                            {
                                'agent': ident,
                                'attribute': attribute,
                                'seq': act_seq_idx,
                                'act': act_type,
                                'x': x,
                                'y': y,
                                'start': activity_start_dt.time(),
                                'start_day': activity_start_dt.day,
                                'end': activity_end_dt.time(),
                                'end_day': activity_end_dt.day,
                                'start_s': self.get_seconds(activity_start_dt),
                                'end_s': self.get_seconds(activity_end_dt),
                                'duration': activity_duration,
                                'duration_s': activity_duration.total_seconds()
                            }
                        )

                        activity_start_dt = activity_end_dt
                    
                    # if a 'pt interaction' activity has duration (ie it has an 'end_time' attribute)
                    # then advance the next activity start time accordingly   
                    elif stage.get('end_time'):
                        end_time_str = stage.get('end_time')

                        activity_start_dt = matsim_time_to_datetime(
                            activity_start_dt, end_time_str, self.logger, idx=ident
                        )

                elif stage.tag == 'leg':

                    leg_mode = stage.get('mode')
                    route_elem = stage.xpath('route')[0]
                    distance = float(route_elem.get("distance", 0))

                    mode = self.extract_mode_from_route_elem(leg_mode, route_elem)

                    mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)  # ignore access and egress walk
                    modes[mode] = modes.get(mode, 0) + distance
                    trip_distance = distance

                    trav_time = stage.get('trav_time')
                    h, m, s = trav_time.split(":")
                    td = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                    activity_start_dt += td

            self.see_trips_log.add(trips)
            summary.extend(trips)

        def get_dominant(group):
            group['dominantModeDistance'] = group['distance'].max()
            group['dominantTripMode'] = group.loc[group['distance'].idxmax(), 'mode'] # this might return a series if multiple same max values ?
            return group

        def get_relativeUtilityToSelected(group):
            selected_utility = group[group['selected']=='yes']['utility'].values[0]
            group['relativeDisUtilityToSelected'] = group['utility'] - selected_utility
            return group

        trips = pd.DataFrame(summary).groupby(['agent',"innovation_hash"]).apply(get_dominant)

        # for each agent and each innovation, we want to define the dominantMode (by frequency)
        trips['dominantModeFreq'] = trips.groupby(['agent','innovation_hash'])['mode'].transform('max')
        trips = trips.drop_duplicates(subset=['agent','innovation_hash'],keep='first')

        # we use the innovation_hash to identify a summary, per agent
        plans = trips.groupby(['agent',"innovation_hash","utility","dominantTripMode"],as_index=False)['mode','selected'].agg(lambda x: ','.join(x.unique()))

        # calculate relative disutility of a plan to the selected plan, gives indication of relative proximity to that chosen/selected
        # Remember, the chosen plan may not always be the top score due to randomness in MATSim process
        # in the case of the selected plan, the relative disutility is 0
        plans = plans.groupby(['agent']).apply(get_relativeUtilityToSelected)
        plans['relativeDisUtilityToSelectedPerc'] = plans['relativeDisUtilityToSelected'] / plans['utility'] * -100.0

        # We find the home locations, based on the origin activity (home)
        # we use home locations for visulisation purposes
        homes = trips[trips.o_act=="home"][['ox','oy','agent']]
        homes.drop_duplicates(subset=['agent'],keep='first')

        # merge this table into the plans, giving us the ox and oy
        plans = plans.merge(homes,on='agent',how='left')

        # todo. Add a warning if many home locations are not found
        gdf = geopandas.GeoDataFrame(plans, geometry=geopandas.points_from_xy(plans.ox, plans.oy))

        # dropping some columns
        # we've used ox/oy to build geometry
        # we remove mode as it is not used, we now use the dominantMode
        gdf = gdf.drop(['ox', 'oy', 'mode'], axis=1)

        # British east/northing
        # need to inherit this from elara config
        gdf.crs = {'init': 'epsg:27700'}

        # re-project to 4326
        gdf['geometry'] = gdf['geometry'].to_crs(epsg=4326)

        # sort by utility
        gdf = gdf.sort_values("utility")
        # flatten, one row per innovation (removes duplicates from lazy processes above)
        gdf = gdf.drop_duplicates(subset=['innovation_hash'],keep='first')

        # Kepler weirdness. Moving from yes/no (matsim lingo) to a bool for whether or not it was selected
        # enables this to be toggled on/off on kepler
        gdf['selected'] = gdf['selected'].map({'yes':True ,'no':False})

        # let's sort and label them in order (i.e. 1st (selected),  2nd (least worst etc), per plan
        gdf = gdf.sort_values('utility')
        gdf['scoreRank'] = gdf.groupby(['agent'])['utility'].rank(method='dense',ascending=False).astype(int)

        # subselecting them into 2 different dfs
        selectedPlans = gdf[gdf.selected==True]
        unSelectedPlans = gdf[gdf.selected==False]

        # Often we will have time/route innovations across n innovation strategies
        # Since we care about mode shift, we can pick the best innovation per mode. 
        # this is the 'best' option for a given mode
        # since we have sorted based on utility, we can remove duplicates 

        unSelectedPlans = unSelectedPlans.sort_values("utility")
        unSelectedPlans = unSelectedPlans.drop_duplicates(['agent','dominantTripMode'], keep='last')

        # zip them back together again
        gdf = pd.concat([selectedPlans,unSelectedPlans])

        self.results['SeeAllPlansGdf'] = self.results['SeeAllPlansGdf'].append(gdf)

        # creation of a df where car is selected
        # but PT exists in their unchosen plans
        # "mode shift opportunity" gdf
        PlanAgentsSel = gdf[gdf.selected==True]
        carPlanAgentsSel = PlanAgentsSel[PlanAgentsSel.dominantTripMode=='car']

        unSelectedPlansCarSelected = unSelectedPlans[unSelectedPlans.agent.isin(carPlanAgentsSel.agent.unique())]

        # This finds all modes that aren't car, generally bike, walk, pt etc
        unSelectedPlansCarSelected = unSelectedPlans[~unSelectedPlans.dominantTripMode.isin(['car'])]

        self.results['SeeUnSelectedPlansCarSelectedGdf'] = self.results['SeeUnSelectedPlansCarSelectedGdf'].append(unSelectedPlansCarSelected)

    def finalise(self):
        """
        Finalise aggregates and joins these results as required and creates a dataframe.
        """
        self.see_trips_log.finish()
        # self.allPlansGdf.finish()
        # self.unSelectedPlansCarSelectedGdf.finish()

    @staticmethod
    def get_seconds(dt: datetime) -> int:
        """
        Extract time of day in seconds from datetime.
        :param dt: datetime
        :return: int, seconds
        """
        d = dt.day
        h = dt.hour
        m = dt.minute
        s = dt.second
        return s + (60 * (m + (60 * (h + ((d-1) * 24)))))

class UtilityLogs(PlanHandlerTool):

    requirements = ['plans']
    valid_modes = ['all']

    # todo make it so that 'all' option not required (maybe for all plan handlers)

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        """

        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.utility_log = None
        # Initialise results storage
        self.results = dict()  # Result dataframes ready to export

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """
        super().build(resources, write_path=write_path)

        utility_csv_name = f"{self.name}.csv"

        self.utility_log = self.start_csv_chunk_writer(utility_csv_name, write_path=write_path, compression=self.compression)

    def process_plans(self, elem):

        """
        Build list of the utility of the selected plan for each agent.

        :return: Tuple[List[dict]]
        """

        ident = elem.get('id')

        for plan in elem.xpath(".//plan"):

            if plan.get('selected') != 'no':

                score = plan.get('score')
                utilities = [{'agent': ident, 'score': score}]
                self.utility_log.add(utilities)

                return None

    def finalise(self):
        """
        Finalise aggregates and joins these results as required and creates a dataframe.
        """
        self.utility_log.finish()


class PlanLogs(PlanHandlerTool):
    """
    Write log of all plans, including selection and score.
    Format will we mostly duplicate of legs log.
    """

    """
    Note that MATSim plan output plans display incorrect 'dep_time' (they show departure time of
    original init plan) and do not show activity start time. As a result, we use leg 'duration'
    to calculate the start of the next activity. This results in time waiting to enter
    first link as being activity time. Therefore activity durations are slightly over reported
    and leg duration under reported.
    """

    requirements = ['plans', 'attributes']

    def __init__(self, config, mode="all", groupby_person_attribute="subpopulation", **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        """

        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.groupby_person_attribute = groupby_person_attribute

        self.plans_log = None

        # Initialise results storage
        self.results = dict()  # Results will remain empty as using writer

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        self.attributes = self.resources["attributes"]

        csv_name = f"{self.name}.csv"
        self.plans_log = self.start_csv_chunk_writer(csv_name, write_path=write_path, compression=self.compression)

    def process_plans(self, elem):

        """
        Build list of leg logs (dicts) for each selected plan.

        Note that this assumes that first stage of a plan is ALWAYS an activity.
        Note that activity wrapping is not dealt with here.

        :return: Tuple[List[dict]]
        """

        ident = elem.get('id')
        attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)

        if not self.mode == "all" and not attribute == self.mode:
            return None

        for pidx, plan in enumerate(elem.xpath(".//plan")):

            selected = str(plan.get('selected'))
            score = float(plan.get('score', 0))

            trip_records = []
            trip_seq = 0
            arrival_dt = None
            in_transit = False
            trip_start_time = None
            prev_mode = "NA"
            prev_act = "NA"
            leg_mode = None
            prev_x = None
            prev_y = None

            for stage in plan:

                if stage.tag == 'activity':
                    act_type = stage.get('type')

                    if not act_type == 'pt interaction':

                        end_time_str = stage.get('end_time', '23:59:59')

                        activity_end_dt = matsim_time_to_datetime(
                            arrival_dt, end_time_str, self.logger, idx=ident
                        )

                        duration, arrival_dt = safe_duration(arrival_dt, activity_end_dt)

                        # MATSim BUG: first activity location may not be recorded
                        # Use NaN to allow distance() -> NaN
                        x = float(stage.get('x', np.NaN))
                        y = float(stage.get('y', np.NaN))

                        if trip_start_time is not None:  # ignores first activity

                            trip_seq += 1

                            if in_transit:
                                trip_mode = "pt"
                            else:
                                trip_mode = leg_mode

                            # record previous trip
                            trip_records.append(
                                {
                                    "pid": ident,
                                    "subpop": attribute,
                                    # "license": license,
                                    "plan": pidx,
                                    "seq": trip_seq,
                                    "start": self.get_seconds(trip_start_time),
                                    "distance": distance(x, y, prev_x, prev_y),
                                    "mode": trip_mode,
                                    "prev_mode": prev_mode,
                                    "origin_activity": prev_act,
                                    "destination_activity": act_type,
                                    "act_duration": duration.seconds,
                                    "selected": selected,
                                    "score": score
                                }
                            )

                            prev_mode = trip_mode

                        prev_act = act_type
                        in_transit = False
                        prev_x = x
                        prev_y = y
                        arrival_dt = activity_end_dt

                    # if a 'pt interaction' activity has duration (ie it has an 'end_time' attribute)
                    # then advance the trip arrival time accordingly   
                    elif stage.get('end_time'):
                        end_time_str = stage.get('end_time')

                        arrival_dt = matsim_time_to_datetime(
                            arrival_dt, end_time_str, self.logger, idx=ident
                        )

                elif stage.tag == 'leg':

                    leg_start_time = activity_end_dt
                    if not in_transit:
                        trip_start_time = leg_start_time

                    leg_mode = stage.get('mode')
                    if leg_mode == "pt":
                        in_transit = True

                    trav_time = stage.get('trav_time')
                    h, m, s = trav_time.split(":")
                    td = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                    arrival_dt = arrival_dt + td

            total_trips = len(trip_records)
            total_duration = sum([trip['act_duration'] for trip in trip_records])
            total_distance = sum([trip['distance'] for trip in trip_records])

            for trip in trip_records:
                trip["total_trips"] = total_trips
                trip["total_duration"] = total_duration
                trip["total_distance"] = total_distance

            self.plans_log.add(trip_records)

    def finalise(self):
        """
        Finalise aggregates and joins these results as required and creates a dataframe.
        """
        self.plans_log.finish()

    @staticmethod
    def get_seconds(dt: datetime) -> int:
        """
        Extract time of day in seconds from datetime.
        :param dt: datetime
        :return: int, seconds
        """
        h = dt.hour
        m = dt.minute
        s = dt.second
        return s + (60 * (m + (60 * h)))


class AgentTollsPaidFromRPConfig(PlanHandlerTool):
    """
    Deprecated in favour of AgentTollLogs Event Handler

    Produces summaries of tolls paid by extracting routes from plans and
    comparing them to the road_pricing.xml config file.

    When using factored tolling -- either capped or differential --
    the actual amount charged to individual agents may be different.
    """

    requirements = [
        'plans',
        'attributes',
        'road_pricing'
        ]
    valid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute="subpopulation", **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        :param attribute: str, attribute key defaults to subpopulation
        """

        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.groupby_person_attribute = groupby_person_attribute
        self.roadpricing = None
        self.agents_ids = None
        self.results = dict()  # Result dataframes ready to export

        # deprecation warning
        self.logger.warning("""
            Plan tolls may differ from actual tolls paid. Use EventHandler.
        """)

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build Handler.
        :param resources: dict, resources from suppliers
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        self.attributes = resources['attributes']
        self.roadpricing = resources['road_pricing']
        self.toll_log = pd.DataFrame(
            columns=["agent", "subpopulation", "tollname", "link", "time", "toll"]
        )  # df for results

    def process_plans(self, elem):
        """
        Iteratively check whether agent pays toll as part of their car trip and append to log.
        :param elem: Plan XML element
        """
        ident = elem.get('id')
        attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)
        agent_in_tolled_space = [0, 0]  # {trip marker, whether last link was tolled}

        def apply_toll(agent_in_tolled_space, current_link_tolled, start_time):
            if current_link_tolled and agent_in_tolled_space[1] is False:
                return True  # enter into tolled space from non-tolled space
            elif current_link_tolled and agent_in_tolled_space[1] is True and agent_in_tolled_space[0] != start_time:
                return True  # already in a tolled space but starting new trip
            else:
                return False

        def get_toll(link, time):  # assumes prices fed in chronological order
            for elem in self.roadpricing.links[link]:
                if time < elem.get('end_time'):
                    return elem.get('amount')

        for plan in elem.xpath(".//plan"):
            if plan.get('selected') != 'no':

                for stage in plan:
                    if stage.tag == 'leg':
                        mode = stage.get('mode')
                        start_time = stage.get('dep_time')
                        if not mode == self.mode:
                            continue
                        route = stage.xpath('route')[0].text.split(' ')

                        for i, link in enumerate(route):
                            if link in self.roadpricing.links:
                                current_link_tolled = True
                            else:
                                current_link_tolled = False

                            # append results to dictionary if toll applies
                            if apply_toll(agent_in_tolled_space, current_link_tolled, start_time):
                                toll_dictionary = {
                                                "agent": ident,
                                                "subpopulation": attribute,
                                                "tollname": self.roadpricing.tollnames[link],
                                                "link": link,
                                                "time": start_time,
                                                "toll": get_toll(link, start_time)}
                                self.toll_log = self.toll_log.append(toll_dictionary, ignore_index=True)

                            # update memory of last link
                            if link in self.roadpricing.links:
                                agent_in_tolled_space[1] = True
                            else:
                                agent_in_tolled_space[1] = False
                            agent_in_tolled_space[0] = start_time  # use start time as a marker of unique leg

    def finalise(self):
        """
        Following plan processing, we now have a log of all tolls levvied on agents.
        We write two versions out - the whole log and some aggregate statistics by subpopulation.
        """
        # log of individual toll events
        toll_log_df = self.toll_log
        toll_log_df['toll'] = pd.to_numeric(toll_log_df['toll'])
        key = "tolls_paid_log"
        self.results[key] = toll_log_df

        # total amount paid by each agent
        aggregate_df = toll_log_df.groupby(by=['agent', 'subpopulation'])['toll'].sum()
        key = "tolls_paid_total_by_agent"
        self.results[key] = aggregate_df

        # average amount paid by each agent within subpopulation
        aggregate_df = aggregate_df.reset_index().groupby(by=['subpopulation'])['toll'].mean()
        key = "tolls_paid_average_by_subpopulation"
        self.results[key] = aggregate_df


class AgentHighwayDistanceLogs(PlanHandlerTool):
    """
    Extract modal distances from agent plans.
    todo road only will not require transit schedule... maybe split road and pt
    """

    requirements = [
        'plans',
        'subpopulations',
        'osm_ways'
        ]
    valid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.osm_ways = None
        self.agents_ids = None
        self.ways = None
        self.agent_indices = None
        self.ways_indices = None
        self.distances = None

        # Initialise results storage
        self.results = dict()  # Result dataframes ready to export

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build Handler.
        :param resources: dict, resources from suppliers
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        self.osm_ways = resources['osm_ways']
        self.subpopulations = resources['subpopulations'].map

        # Initialise agent indices
        self.agent_ids, self.agent_indices = self.generate_indices_map(list(self.subpopulations))

        # Initialise way indices
        self.ways, self.ways_indices = self.generate_indices_map(self.osm_ways.classes)

        # Initialise results array
        self.distances = np.zeros((
            len(self.agent_ids),
            len(self.ways))
        )

    def process_plans(self, elem):
        """
        Iteratively aggregate distance on highway distances from legs of selected plans.
        :param elem: Plan XML element
        """
        ident = elem.get('id')

        for plan in elem.xpath(".//plan"):
            if plan.get('selected') != 'no':

                for stage in plan:

                    if stage.tag == 'leg':
                        mode = stage.get('mode')
                        if not mode == self.mode:
                            continue

                        route = stage.xpath('route')[0].text.split(' ')
                        length = len(route)
                        for i, link in enumerate(route):
                            way = str(self.osm_ways.ways.get(link, None))
                            distance = float(self.osm_ways.lengths.get(link, 0))
                            if i == 0 or i == length - 1:  # halve first and last link lengths
                                distance /= 2

                            x, y = self.table_position(
                                ident,
                                way,
                            )

                            self.distances[x, y] += distance

    def finalise(self):
        """
        Following plan processing, the raw distance table will contain agent travel distance by way.
        Finalise aggregates and joins these results as required and creates a dataframe.
        """

        names = ['agent', 'way']
        indexes = [self.agent_ids, self.ways]
        index = pd.MultiIndex.from_product(indexes, names=names)
        distance_df = pd.DataFrame(self.distances.flatten(), index=index)[0]

        # mode counts breakdown output
        distance_df = distance_df.unstack(level='way').sort_index()

        # calculate agent total distance
        distance_df['total'] = distance_df.sum(1)

        # calculate summary
        total_df = distance_df.sum(0)
        key = f"{self.name}_totals"
        self.results[key] = total_df

        # join with agent attributes
        key = f"{self.name}"
        distance_df['attribute'] = distance_df.index.map(self.subpopulations)
        self.results[key] = distance_df

    def table_position(
        self,
        ident,
        way
    ):
        """
        Calculate the result table coordinates from given maps.
        :param ident: String, agent id
        :param way: String, way id
        :return: (x, y) tuple of integers to index results table
        """
        x = self.agent_indices[ident]
        y = self.ways_indices[way]
        return x, y


class TripHighwayDistanceLogs(PlanHandlerTool):
    """
    Extract modal distances from agent car trips.
    """

    requirements = [
        'plans',
        'osm_ways',
        'attributes'
        ]
    valid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute="subpopulation", **kwargs):
        """
        Initiate handler.
        :param config: config
        :param mode: str, mode option
        :param attribute: str, attribute key defaults to 'subpopulation'
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode
        self.groupby_person_attribute = groupby_person_attribute
        self.osm_ways = None
        self.ways = None

        # Initialise results storage
        self.distances_log = None
        self.results = dict()  # Results will remain empty as using writer

    def build(self, resources: dict, write_path=None) -> None:
        """
        Build Handler.
        :param resources: dict, resources from suppliers
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        self.osm_ways = resources['osm_ways']
        self.attributes = resources['attributes']

        # Initialise ways
        self.ways = self.osm_ways.classes

        # Initialise results writer
        csv_name = f"{self.name}.csv"
        self.distances_log = self.start_csv_chunk_writer(csv_name, write_path=write_path, compression=self.compression)

    def process_plans(self, elem):
        """
        Iteratively aggregate distance on highway distances from legs of selected plans.
        :param elem: Plan XML element
        """
        ident = elem.get('id')
        attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)

        for plan in elem.xpath(".//plan"):
            if plan.get('selected') != 'no':

                trips = []
                trip_counter = None
                trip_seq_idx = 0

                for stage in plan:

                    if stage.tag == 'activity':
                        if not stage.get('type') == 'pt interaction':

                            if trip_counter is not None:
                                # record previous counts and move idx
                                # this 'works' because plans must end with an activity
                                trips.append(trip_counter)

                            trip_seq_idx += 1

                            # set counter
                            trip_counter = {
                                'agent': ident,
                                'subpop': attribute,
                                'seq': trip_seq_idx
                            }
                            trip_counter.update(
                                {k: 0 for k in self.ways}
                            )

                    if stage.tag == 'leg':
                        mode = stage.get('mode')
                        if not mode == self.mode:
                            continue

                        route = stage.xpath('route')[0].text.split(' ')
                        length = len(route)
                        for i, link in enumerate(route):
                            way = str(self.osm_ways.ways.get(link, None))
                            distance = float(self.osm_ways.lengths.get(link, 0))
                            if i == 0 or i == length - 1:  # halve first and last link lengths
                                distance /= 2
                            trip_counter[way] += distance

                self.distances_log.add(trips)

    def finalise(self):
        """
        Finish writer.
        """
        self.distances_log.finish()


class PlanHandlerWorkStation(WorkStation):
    """
    Work Station class for collecting and building Plans Handlers.
    """
    # dict key to hand to supplier.resources
    # allows handler to be subclassed by overriding
    plans_resource = 'plans'

    tools = {
        "trip_modes": TripModes,
        "trip_activity_modes": TripActivityModes,
        "plan_modes": PlanModes,
        "plan_activity_modes": PlanActivityModes,
        "leg_logs": LegLogs,
        "trip_logs": TripLogs,
        "see_trip_logs" : SeeTripLogs,
        "plan_logs": PlanLogs,
        "utility_logs": UtilityLogs,
        "agent_highway_distance_logs": AgentHighwayDistanceLogs,
        "trip_highway_distance_logs": TripHighwayDistanceLogs,
        "toll_logs": AgentTollsPaidFromRPConfig
    }

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def build(self, write_path=None):
        """
        Build all required handlers, then finalise and save results.
        :return: None
        """

        if not self.resources:
            self.logger.warning(f'{str(self)} has no resources, build returning None.')
            return None

        # build tools
        super().build()

        # iterate through plans
        plans = self.supplier_resources[self.plans_resource]
        self.logger.info(' *** Commencing Plans Iteration ***')
        base = 1

        for i, person in enumerate(plans.persons):

            if not (i+1) % base:
                self.logger.info(f'parsed {i + 1} persons plans')
                base *= 2

            for plan_handler in self.resources.values():
                plan_handler.process_plans(person)

        self.logger.info('***Completed Plan Iteration***')

        # finalise
        # Generate event file outputs
        self.logger.debug(f'{str(self)} .resources = {self.resources}')

        for handler_name, handler in self.resources.items():
            self.logger.debug(f'Finalising {str(handler)}')
            handler.finalise()

            self.logger.debug(f'{len(handler.results)} result_dfs at {str(handler)}')

            if handler.results:
                output_path = handler.config.output_path
                self.logger.info(f'Writing results from {str(handler)} to {output_path}')

                for name, result in handler.results.items():
                    csv_name = "{}.csv".format(name)
                    self.write_csv(result, csv_name, write_path=write_path, compression=handler.compression)
                    # hacky - trying to catch geojson results. How is this handled elsewhere? Currently assumes everything is csv
                    if "gdf" in name.lower():
                        self.logger.debug(f'writing to {output_path + f"{name}.geojson"}')
                        export_geojson(result,output_path+f"{name}.geojson")
                    
                    else:
                        csv_name = "{}.csv".format(name)
                        self.write_csv(result, csv_name, write_path=write_path)

                    del result


def convert_time_to_seconds(t: str) -> Optional[int]:
    """
    Convert MATSim output plan times into seconds.
    If t is None, must return None.
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


def matsim_time_to_datetime(
        current_time: datetime,
        new_time_str: str,
        logger=None,
        idx=None,
        base_year=2020,
        base_month=4,
        base_day=1,) -> datetime:
    """
    Function to convert matsim time strings (hours,  minutes and seconds since start)
    to datetime objects (days, hours, minutes, seconds).
    Raises a warning for backward time steps. To remmove backward time steps, consider
    using the 'non_wrapping_datetime' method which has the same arguments.
    :param current_time: datetime from previous event
    :param new_time_str: new time string
    :param logger: optional logger
    :param idx: optional idx
    :param base_year: int default=2020,
    :param base_month: int defaul=4,
    :param base_day: int defaul=1,
    :return: time string, day
    """
    start_of_day = datetime(year=base_year, month=base_month, day=base_day, hour=0)
    if current_time is None:
        current_time = start_of_day

    h, m, s = (int(i) for i in new_time_str.split(":"))
    new_time = start_of_day + timedelta(hours=h, minutes=m, seconds=s)

    if logger is not None:
        if h > 23:
            logger.debug(f'Bad time str: {new_time_str}, outputting: {new_time}, idx: {idx}')
        if new_time < current_time:
            logger.warning(f'Time Wrapping (new time:{new_time} < previous time:{current_time}), idx: {idx}')

    return new_time


def safe_duration(start_time, end_time):
    """
    Duration calculation that can cope with None as starting time. In which case assumes start time at start
    of day (00:00:00).
    Returns tuple(datetime.timedelta, datetime.datetime)
    """
    if start_time is None:
        start_time = datetime(
            year=end_time.year,
            month=end_time.month,
            day=end_time.day,
            hour=0
            )
        return timedelta(hours=end_time.hour, minutes=end_time.minute, seconds=end_time.second), start_time
    return end_time - start_time, start_time


def distance(x, y, prev_x, prev_y):
    dx = x - prev_x
    dy = y - prev_y
    return np.sqrt(dx*dx + dy*dy)

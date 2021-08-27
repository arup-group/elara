import numpy as np
from math import floor
import pandas as pd
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

    def __init__(self, config, mode=None, groupby_person_attribute=None, **kwargs):
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
        Extract mode and route identifieers from a MATSim v12 route xml element.
        """
        route = route_elem.text.split('===')[-2]
        mode = self.resources['transit_schedule'].route_to_mode_map.get(route)
        return mode

    def extract_routeid_from_v12_route_elem(self, route_elem):
        """
        Extract mode and route identifieers from a route xml element.
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

    def __init__(self, config, mode=None, groupby_person_attribute=None, **kwargs) -> None:
        """
        Initiate Handler.
        :param config: Config
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode  # todo options not implemented
        self.groupby_person_attribute = groupby_person_attribute

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

    def process_plans(self, elem):
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
        :param elem: Plan XML element
        """
        for plan in elem.xpath(".//plan"):
            if plan.get('selected') == 'yes':

                ident = elem.get('id')
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute)

                end_time = None
                modes = {}

                for stage in plan:

                    if stage.tag == 'leg':
                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)
                        distance = float(route_elem.get("distance", 0))
                        
                        mode = {"egress_walk":"walk", "access_walk":"walk"}.get(mode, mode)  # ignore access and egress walk
                        modes[mode] = modes.get(mode, 0) + distance

                    elif stage.tag == 'activity':
                        if stage.get('type') == 'pt interaction':  # ignore pt interaction activities
                            continue

                        # only add activity modes when there has been previous activity
                        # (ie trip start time)
                        if end_time:
                            mode = self.get_furthest_mode(modes)
                            x, y, z = self.table_position(
                                mode,
                                attribute_class,
                                end_time
                            )

                            self.mode_counts[x, y, z] += 1

                        # update endtime for next activity
                        end_time = convert_time_to_seconds(stage.get('end_time'))

                        # reset modes
                        modes = {}

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
        counts_df = pd.DataFrame(self.mode_counts.flatten(), index=index)[0]
        # mode counts breakdown output
        counts_df = counts_df.unstack(level='mode').sort_index()
        if self.groupby_person_attribute:
            key = f"{self.name}_{self.groupby_person_attribute}_counts"
            self.results[key] = counts_df

        # mode counts totals output
        total_counts_df = counts_df.sum(0)
        key = f"{self.name}_counts"
        self.results[key] = total_counts_df

        # convert to mode shares
        total = self.mode_counts.sum()

        # mode shares breakdown output
        if self.groupby_person_attribute:
            key = f"{self.name}_{self.groupby_person_attribute}"
            self.results[key] = counts_df / total

        # mode shares totals output
        key = f"{self.name}"
        self.results[key] = total_counts_df / total

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

    def table_position(
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
        :return: (x, y, z, w) tuple of integers to index results table
        """
        x = self.mode_indices[mode]
        y = self.class_indices[attribute_class]
        z = floor(time / (86400.0 / self.config.time_periods)) % self.config.time_periods
        return x, y, z

class TripDestinationModeShare(PlanHandlerTool):
    """
    Extract mode shares for specified activities from plans. This handler takes a list of activities and computes the mode shares for each
    independant activity trip. 
    """

    requirements = [
        'plans',
        'attributes',
        'transit_schedule',
        'output_config',
    ]
    valid_modes = ['all']

    def __init__(self, config, mode=None, groupby_person_attribute=None,destination_activity_filters=None, **kwargs) -> None:
        """
        Initiate Handler.
        :param config: Config
        :param mode: str, mode
        :param groupby_person_attribute: list, attributes
        :param destination_activity_filters: list, activities
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode  # todo options not implemented
        self.groupby_person_attribute = groupby_person_attribute
        self.destination_activity_filters = destination_activity_filters
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

        # Initialise mode count table, if have list of activities then initiallise as dictionary of tables
 
        self.mode_counts = np.zeros((
            len(self.modes),
            len(self.classes),
            self.config.time_periods))

        self.results = dict()

    def process_plans(self, elem):
        """
        Iterate through the plans and produce counts / mode shares for trips to the destination activities specified in the list. 
        This handler counts the longest travelling mode of the trip leg leading to each instance of the destination activity(ies) specified 
        e.g. if the destination acitivity list consists of ['work_a', work_b] and a plan consists of the trips 
        [home] --> (bus,11km) --> [work_a] --> (train, 10km) --> [work_b], the resulting counts will see (bus) +1 & (train) + 1.
        :param elem: Plan XML element
        """
        for plan in elem.xpath(".//plan"):
            if plan.get('selected') == 'yes':

                ident = elem.get('id')
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute)

                end_time = None
                modes = {}

                for stage in plan:
                    if stage.tag == 'leg':
                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)
                        distance = float(route_elem.get("distance", 0))
                        
                        mode = {"egress_walk":"walk", "access_walk":"walk"}.get(mode, mode)  # ignore access and egress walk
                        modes[mode] = modes.get(mode, 0) + distance

                    elif stage.tag == 'activity':
                        activity = stage.get('type')
                        if activity == 'pt interaction':  # ignore pt interaction activities
                            continue

                        # only add activity modes when there has been previous activity
                        # (ie trip start time) AND the activity is in specified list
                        if end_time and (activity in self.destination_activity_filters):
                            mode = self.get_furthest_mode(modes)
                            x, y, z = self.table_position(
                                mode,
                                attribute_class,
                                end_time
                            )
                            self.mode_counts[x, y, z] += 1 
                        # update endtime for next activity
                        end_time = convert_time_to_seconds(stage.get('end_time'))

                        # reset modes
                        modes = {}

    def finalise(self):
        """
        Following plan processing, the raw mode share table will contain counts by mode,
        population attribute class, activity and period (where period is based on departure
        time).
        Finalise aggregates these results as required and creates a dataframe.
        """

        # Scale final counts
        self.mode_counts *= 1.0 / self.config.scale_factor
        activity_filter_name = '_'.join(self.destination_activity_filters)
        names = ['mode', 'class', 'hour']
        indexes = [self.modes, self.classes, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.mode_counts.flatten(), index=index)[0]
        # mode counts breakdown output
        counts_df = counts_df.unstack(level='mode').sort_index()
        counts_df = counts_df.reset_index().drop("hour", axis=1)
        counts_df = counts_df.groupby(counts_df["class"]).sum()
        # this removes the breakdown by hour which no one has been using

        if self.groupby_person_attribute:
            key = f"{self.name}_{self.groupby_person_attribute}_{activity_filter_name}_counts"
            self.results[key] = counts_df

        # mode counts totals output
        total_counts_df = counts_df.sum(0)
        key = f"{self.name}_{activity_filter_name}_counts"
        self.results[key] = total_counts_df

        # convert to mode shares
        total = self.mode_counts.sum()

        # mode shares breakdown output
        if self.groupby_person_attribute:
            key = f"{self.name}_{self.groupby_person_attribute}_{activity_filter_name}"
            self.results[key] = counts_df / total

        # mode shares totals output
        key = f"{self.name}_{activity_filter_name}"
        self.results[key] = total_counts_df / total

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

    def table_position(
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
        :return: (x, y, z, w) tuple of integers to index results table
        """
        x = self.mode_indices[mode]
        y = self.class_indices[attribute_class]
        z = floor(time / (86400.0 / self.config.time_periods)) % self.config.time_periods
        return x, y, z



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

    def __init__(self, config, mode=None, groupby_person_attribute="subpopulation", **kwargs):
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

        self.activities_log = self.start_chunk_writer(activity_csv_name, write_path=write_path)
        self.legs_log = self.start_chunk_writer(legs_csv_name, write_path=write_path)

    def process_plans(self, elem):

        """
        Build list of leg and activity logs (dicts) for each selected plan.

        Note that this assumes that first stage of a plan is ALWAYS an activity.
        Note that activity wrapping is not dealt with here.

        :return: Tuple[List[dict]]
        """
        ident = elem.get('id')

        for plan in elem.xpath(".//plan"):

            if plan.get('selected') == 'yes':

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

                        if not act_type == 'pt interaction':

                            trip_seq_idx += 1  # increment for a new trip idx

                            end_time_str = stage.get('end_time', '23:59:59')

                            activity_end_dt = matsim_time_to_datetime(
                                arrival_dt, end_time_str, self.logger, idx=ident
                            )

                            duration = activity_end_dt - arrival_dt

                        else:
                            activity_end_dt = arrival_dt
                            duration = arrival_dt - arrival_dt  # zero duration

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
                        mode = {"egress_walk":"walk", "access_walk":"walk"}.get(mode, mode)

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
                                # 'duration': td,
                                'end_day': arrival_dt.day,
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
    valid_modes = ['all'] #mode and purpose options need to be enabled for post-processing cross tabulation w euclidian distance

    # todo make it so that 'all' option not required (maybe for all plan handlers)

    """
    Note that MATSim plan output plans display incorrect 'dep_time' (they show departure time of 
    original init plan) and do not show activity start time. As a result, we use leg 'duration' 
    to calculate the start of the next activity. This results in time waiting to enter 
    first link as being activity time. Therefore activity durations are slightly over reported 
    and leg duration under reported.
    """

    def __init__(self, config, mode=None, groupby_person_attribute="subpopulation", see=None, **kwargs):
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
        self.see = see

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

        self.activities_log = self.start_chunk_writer(activity_csv_name, write_path=write_path)
        self.trips_log = self.start_chunk_writer(trips_csv_name, write_path=write_path)

    def process_plans(self, elem):

        """
        Build list of trip and activity logs (dicts) for each selected plan.

        Note that this assumes that first stage of a plan is ALWAYS an activity.
        Note that activity wrapping is not dealt with here.

        :return: Tuple[List[dict]]
        """
        ident = elem.get('id')

        # this toggles on the keeping of unselected plans
        if self.see:
            plans_to_keep = ['yes','no']
        
        else:
            plans_to_keep = ['yes']

        for plan in elem.xpath(".//plan"):

            attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)

            if plan.get('selected') in plans_to_keep:

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

                # This is used to track different output plans, and therefore compare unselected plans
                innovation_hash = str(uuid.uuid4())

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

                                trip_record = {
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
                                        "utility" : plan.get("score")
                                    }
                                
                                if self.see:
                                    
                                    # add the extra required SEE fields
                                    trip_record["selected"] = plan.get('selected')
                                    trip_record['innovation_hash'] = innovation_hash

                                trips.append(trip_record)

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
                                    'duration_s': activity_duration.total_seconds(),
                                    # "utility": plan.get("score"),
                                    # "selected": plan.get('selected'),
                                    # "innovation_hash" : innovation_hash,
                                }
                            )

                            activity_start_dt = activity_end_dt

                    elif stage.tag == 'leg':

                        leg_mode = stage.get('mode')
                        route_elem = stage.xpath('route')[0]
                        distance = float(route_elem.get("distance", 0))

                        mode = self.extract_mode_from_route_elem(leg_mode, route_elem)

                        mode = {"egress_walk": "walk", "access_walk": "walk"}.get(mode, mode)  # ignore access and egress walk
                        modes[mode] = modes.get(mode, 0) + distance
                        trip_distance += distance

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


class UtilityLogs(PlanHandlerTool):

    requirements = ['plans']
    valid_modes = ['all']

    # todo make it so that 'all' option not required (maybe for all plan handlers)

    def __init__(self, config, mode=None, groupby_person_attribute=None, **kwargs):
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

        self.utility_log = self.start_chunk_writer(utility_csv_name, write_path=write_path)

    def process_plans(self, elem):

        """
        Build list of the utility of the selected plan for each agent.

        :return: Tuple[List[dict]]
        """

        ident = elem.get('id')

        for plan in elem.xpath(".//plan"):

            if plan.get('selected') == 'yes':

                score = plan.get('score')
                utilities = [{'agent': ident,'score': score}]
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

    def __init__(self, config, mode=None, groupby_person_attribute="subpopulation", **kwargs):
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
        self.plans_log = self.start_chunk_writer(csv_name, write_path=write_path)

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
                        x = float(stage.get('x'))
                        y = float(stage.get('y'))

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
                    arrival_dt = activity_end_dt + td

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

class AgentTollsPaid(PlanHandlerTool):
    """
    Extract where and when agents paid tolls and produce summaries by agent and subpopulation.
    """

    requirements = [
        'plans',
        'attributes',
        'road_pricing'
        ]
    valid_modes = ['car']

    def __init__(self, config, mode=None, groupby_person_attribute="subpopulation", **kwargs):
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
        self.toll_log = pd.DataFrame(columns = ["agent","subpopulation","tollname","link","time","toll"]) #df for results
        
    def process_plans(self, elem):
        """
        Iteratively check whether agent pays toll as part of their car trip and append to log.
        :param elem: Plan XML element
        """
        ident = elem.get('id')
        attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)
        agent_in_tolled_space = [0,0] # {trip marker, whether last link was tolled}

        def apply_toll(agent_in_tolled_space, current_link_tolled, start_time):
            if current_link_tolled and agent_in_tolled_space[1] == False: #entering into tolled space from non-tolled space
                return True
            elif current_link_tolled and agent_in_tolled_space[1] == True and agent_in_tolled_space[0] != start_time: #already in a tolled space but starting new trip
                return True
            else:
                return False
        
        def get_toll(link,time): #assumes prices fed in chronological order
            for elem in self.roadpricing.links[link]:
                if time < elem.get('end_time'):
                    return elem.get('amount')

        for plan in elem.xpath(".//plan"):
            if plan.get('selected') == 'yes':

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
                                                "tollname":self.roadpricing.tollnames[link],
                                                "link": link,
                                                "time": start_time,
                                                "toll": get_toll(link,start_time)}
                                self.toll_log = self.toll_log.append(toll_dictionary, ignore_index=True)
                            
                            #update memory of last link
                            if link in self.roadpricing.links:
                                agent_in_tolled_space[1] = True
                            else:
                                agent_in_tolled_space[1] = False
                            agent_in_tolled_space[0] = start_time # use start time as a marker of unique leg

    def finalise(self):
        """
        Following plan processing, we now have a log of all tolls levvied on agents.
        We write two versions out - the whole log and some aggregate statistics by subpopulation.
        """
        # log of individual toll events
        toll_log_df = self.toll_log
        toll_log_df['toll'] = pd.to_numeric(toll_log_df['toll'])
        key = f"tolls_paid_log"
        self.results[key] = toll_log_df

        #total amount paid by each agent
        aggregate_df = toll_log_df.groupby(by=['agent','subpopulation'])['toll'].sum()
        key = f"tolls_paid_total_by_agent"
        self.results[key] = aggregate_df

        # average amount paid by each agent within subpopulation
        aggregate_df = aggregate_df.reset_index().groupby(by=['subpopulation'])['toll'].mean()
        key = f"tolls_paid_average_by_subpopulation"
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

    def __init__(self, config, mode=None, groupby_person_attribute=None, **kwargs):
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
            if plan.get('selected') == 'yes':

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

    def __init__(self, config, mode=None, groupby_person_attribute="subpopulation", **kwargs):
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
        self.distances_log = self.start_chunk_writer(csv_name, write_path=write_path)

    def process_plans(self, elem):
        """
        Iteratively aggregate distance on highway distances from legs of selected plans.
        :param elem: Plan XML element
        """
        ident = elem.get('id')
        attribute = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)

        for plan in elem.xpath(".//plan"):
            if plan.get('selected') == 'yes':
                
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

    tools = {
        "mode_shares": ModeShares,
        "trip_destination_mode_share": TripDestinationModeShare,
        "leg_logs": LegLogs,
        "trip_logs": TripLogs,
        "plan_logs": PlanLogs,
        "utility_logs": UtilityLogs,
        "agent_highway_distance_logs": AgentHighwayDistanceLogs,
        "trip_highway_distance_logs": TripHighwayDistanceLogs,
        "toll_logs": AgentTollsPaid
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
            self.logger.warning(f'{self.__str__} has no resources, build returning None.')
            return None

        # build tools
        super().build()

        # iterate through plans
        plans = self.supplier_resources['plans']
        self.logger.info(f' *** Commencing Plans Iteration ***')
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
        self.logger.debug(f'{self.__str__()} .resources = {self.resources}')

        for handler_name, handler in self.resources.items():
            self.logger.info(f'Finalising {handler.__str__()}')
            handler.finalise()

            self.logger.debug(f'{len(handler.results)} result_dfs at {handler.__str__()}')

            if handler.results:
                self.logger.info(f'Writing results from {handler.__str__()}')

                for name, result in handler.results.items():
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
        base_day=1,
    ) -> datetime:
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
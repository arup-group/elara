import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Union, Tuple, Optional
import logging
import os
import networkx as nx
from shapely.geometry import LineString
from math import floor
from elara.factory import WorkStation, Tool


class EventHandlerTool(Tool):
    """
    Base Tool class for Event Handling.
    """
    result_dfs = dict()
    options_enabled = True

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.groupby_person_attribute = None

    def build(
            self,
            resources: dict,
            write_path=None) -> None:

        super().build(resources, write_path)

    @staticmethod
    def generate_elem_ids(
            elems_in: Union[list, gpd.GeoDataFrame]
    ) -> Tuple[list, dict]:
        """
        Generate element ID list and index dictionary from given geodataframe.
        :param elems_in: elements
        :return: (element IDs, element indices)
        """
        if isinstance(elems_in, pd.DataFrame) or isinstance(elems_in, pd.Series):
            elems_in = elems_in.index.tolist()

        elem_indices = {
            key: value for (key, value) in zip(elems_in, range(0, len(elems_in)))
        }
        return elems_in, elem_indices

    def vehicle_mode(self, vehicle_id: str) -> str:
        """
        Given a vehicle's ID, return its mode type.
        :param vehicle_id: Vehicle ID string
        :return: Vehicle mode type string
        """
        if vehicle_id in self.resources['transit_schedule'].veh_to_mode_map.keys():
            return self.resources['transit_schedule'].veh_to_mode_map[vehicle_id]
        else:
            return "car"

    def vehicle_route(self, vehicle_id: str) -> str:
        """
        Given a vehicle's ID, return the ID of the route it belongs to.
        :param vehicle_id: Vehicle ID string
        :return: ID of the parent route
        """
        if vehicle_id in self.resources['transit_schedule'].veh_to_route_map.keys():
            return self.resources['transit_schedule'].veh_to_route_map.get(vehicle_id)
        else:
            return 'unknown_route'

    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows from given results dataframe if time period columns only contain
        zeroes.
        :param df: Results dataframe
        :return: Contracted results dataframe
        """

        # TODO This is a hack for backwards compaitbility
        # only try to contract if column indices are hours from config
        if set(range(self.config.time_periods)).issubset(set(df.columns)):
            cols = [h for h in range(self.config.time_periods)]
            return df.loc[df[cols].sum(axis=1) > 0]
        else:
            return df

    def finalise(self):
        """
        Transform accumulated results during event processing into final dataframes
        ready for exporting.
        """
        return NotImplementedError

    def contract_results(self) -> None:
        """
        Remove zero-sum rows from all results dataframes.
        """
        self.result_dfs = {
            k: self.remove_empty_rows(df) for (k, df) in self.result_dfs.items()
        }

    def extract_attribute_values(self, attributes, attribute_key) -> set:
        """
        For given attributes input (elara.inputs.Attributes) and attributes key, eg "subpopulations".
        Return available set of attribute values, inclusing None, eg {"old", "young", None}.
        """
        availability = attributes.attribute_key_availability(attribute_key)
        self.logger.debug(f'availability of attribute {attribute_key} = {availability*100}%')
        if availability < 1:
            self.logger.warning(f'availability of attribute {attribute_key} = {availability*100}%')
        return attributes.attribute_values(attribute_key) | {None}

    def extract_attributes(self) -> Tuple:
        """
        Get attributes input and find available attribute values based on self.attribute_key.
        If key is None, return empty attribute dictionary and {None}.

        Returns:
            Tuple: (elara.inputs.Attributes, found_attributes)
        """
        if self.groupby_person_attribute:
            attributes = self.resources['attributes']
            found_attributes = self.extract_attribute_values(attributes, self.groupby_person_attribute)
        else:
            attributes = {}
            found_attributes = {None}
        self.logger.debug(f'found attributes = {found_attributes}')

        return attributes, found_attributes


class VehiclePassengerGraph(EventHandlerTool):
    """
    Extract a graph of interactions (where interactions are passengers sharing some time in a vehicle).
    """

    requirements = [
        'events',
        'transit_vehicles',
        'attributes',
    ]

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        :param attribute: str, atribute key defaults to None
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.groupby_person_attribute = groupby_person_attribute
        self.veh_occupancy = dict()

        # Initialise results storage
        self.graph = nx.Graph()

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        self.agent_attributes = self.resources["attributes"]

        if self.groupby_person_attribute:
            for person, attribs in self.agent_attributes.items():
                self.graph.add_node(person, attribute=attribs.get(self.groupby_person_attribute, None))
        else:
            for person in self.agent_attributes.keys():
                self.graph.add_node(person)

    def process_event(self, elem) -> None:
        """
        :param elem: Event XML element
        """
        event_type = elem.get("type")

        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")

            if agent_id not in self.agent_attributes:
                return None

            veh_ident = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_ident)

            if veh_mode == 'car':  # can ignore cars
                return None

            # update veh occupancy
            if not self.veh_occupancy.get(veh_ident):
                self.veh_occupancy[veh_ident] = [agent_id]
            else:
                for passenger in self.veh_occupancy[veh_ident]:
                    self.graph.add_edge(
                        agent_id,
                        passenger,
                        mode=veh_mode,
                        occupancy=len(self.veh_occupancy[veh_ident])
                    )

                self.veh_occupancy[veh_ident].append(agent_id)

            return None

    def finalise(self):
        del self.veh_occupancy
        name = f"{str(self)}.pkl"
        path = os.path.join(self.config.output_path, name)
        nx.write_gpickle(self.graph, path)
        del self.graph


class StopPassengerWaiting(EventHandlerTool):
    """
    Extract agent waiting times at stops.
    """

    requirements = [
        'events',
        'transit_schedule',
        'attributes',
    ]

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.groupby_person_attribute = groupby_person_attribute
        self.agent_status = dict()
        self.veh_waiting_occupancy = dict()
        self.waiting_time_log = None

        # Initialise results storage
        self.results = dict()

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        self.agent_attributes, _ = self.extract_attributes()

        csv_name = f"{str(self)}.csv"
        self.waiting_time_log = self.start_chunk_writer(csv_name, write_path=write_path)

    def process_event(self, elem) -> None:
        """
        :param elem: Event XML element
        """
        event_type = elem.get("type")
        if event_type == 'waitingForPt':
            time = int(float(elem.get("time")))
            agent_id = elem.get("agent")

            """Note the use of 'agent' above - not 'person' as per usual"""

            if agent_id not in self.agent_status:
                self.agent_status[agent_id] = [time, None]

            else:  # agent is in transit therefore this is an interchange, so update time only
                self.agent_status[agent_id][0] = time
            return None

        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")
            veh_ident = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_ident)

            if veh_mode == 'car':  # can ignore cars
                return None

            # update veh occupancy
            if not self.veh_waiting_occupancy.get(veh_ident):
                self.veh_waiting_occupancy[veh_ident] = [agent_id]
            else:
                self.veh_waiting_occupancy[veh_ident].append(agent_id)

        if event_type == "VehicleDepartsAtFacility":
            veh_ident = elem.get("vehicle")

            if veh_ident not in self.veh_waiting_occupancy:  # ignore if no-one waiting in vehicle
                return None

            vehicle_waiting_report = []

            veh_mode = self.vehicle_mode(veh_ident)
            time = int(float(elem.get("time")))
            stop = elem.get("facility")

            # get loc
            point = self.resources['transit_schedule'].stop_gdf.loc[stop, 'geometry']
            x, y = point.x, point.y

            for agent_id in self.veh_waiting_occupancy[veh_ident]:

                if agent_id not in self.agent_status:  # ignore (ie driver)
                    continue

                agent_attribute = self.agent_attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                start_time, previous_mode = self.agent_status[agent_id]

                waiting_time = time - start_time

                # first waiting
                vehicle_waiting_report.append(
                    {
                        'agent_id': agent_id,
                        'prev_mode': previous_mode,
                        'mode': veh_mode,
                        'stop': stop,
                        'x': x,
                        'y': y,
                        'duration': waiting_time,
                        'departure': time,
                        self.groupby_person_attribute: agent_attribute
                    }
                )

                # update agent status for 'previous mode'
                self.agent_status[agent_id] = [time, veh_mode]

            # clear veh_waiting_occupancy
            self.veh_waiting_occupancy.pop('veh_ident', None)

            self.waiting_time_log.add(vehicle_waiting_report)

            return None

        if event_type == "actstart":

            if elem.get("type") == "pt interaction":  # ignore pt interactions
                return None

            agent_id = elem.get("person")
            self.agent_status.pop(agent_id, None)  # agent has finished transit - remove record

    def finalise(self):
        del self.agent_status
        del self.veh_waiting_occupancy
        self.waiting_time_log.finish()


class LinkVehicleCounts(EventHandlerTool):
    """
    Extract Volume Counts for mode on given network.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.groupby_person_attribute = groupby_person_attribute
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Initialise class attributes
        self.attributes, found_attributes = self.extract_attributes()
        self.classes, self.class_indices = self.generate_elem_ids(found_attributes)
        self.logger.debug(f'available population {self.groupby_person_attribute} values = {self.classes}')

        # generate index and map for network link dimension
        self.elem_gdf = self.resources['network'].link_gdf

        links = resources['network'].mode_to_links_map.get(self.mode)
        if links is None:
            self.logger.warning(
                f"""
                No viable links found for mode:{self.mode} in Network,
                this may be because the Network modes do not match the configured
                modes. Elara will continue with all links found in network.
                """
                )
        else:
            self.logger.debug(f'Selecting links for mode:{self.mode}.')
            self.elem_gdf = self.elem_gdf.loc[links, :]

        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

        # Initialise volume count table
        self.counts = np.zeros((len(self.elem_indices),
                                len(self.classes),
                                self.config.time_periods))

    def process_event(self, elem) -> None:
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
        :param elem: Event XML element
        """
        event_type = elem.get("type")
        if (event_type == "vehicle enters traffic") or (event_type == "entered link"):
            ident = elem.get("vehicle")
            veh_mode = self.vehicle_mode(ident)
            if veh_mode == self.mode:
                # look for attribute_class, if not found assume pt and use mode
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)
                link = elem.get("link")
                time = float(elem.get("time"))
                x, y, z = table_position(
                    self.elem_indices,
                    self.class_indices,
                    self.config.time_periods,
                    link,
                    attribute_class,
                    time
                )
                self.counts[x, y, z] += 1

    def finalise(self) -> None:
        """
        Following event processing, the raw events table will contain counts by link
        by time slice. The only thing left to do is scale by the sample size and
        create dataframes.
        """

        # Overwrite the scale factor for public transport vehicles (these do not need to
        # be expanded.
        scale_factor = self.config.scale_factor
        if self.mode != "car":
            scale_factor = 1.0

        # Scale final counts
        self.counts *= 1.0 / scale_factor

        if self.groupby_person_attribute:
            names = ['elem', self.groupby_person_attribute, 'hour']
            indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
            index = pd.MultiIndex.from_product(indexes, names=names)
            counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]
            counts_df = counts_df.unstack(level='hour').sort_index()
            counts_df = counts_df.reset_index().set_index(['elem', self.groupby_person_attribute])

            counts_df['total'] = counts_df.sum(1)
            counts_df = counts_df.reset_index().set_index('elem')

            key = f"{self.name}_{self.groupby_person_attribute}"
            counts_df = self.elem_gdf.join(
                counts_df, how="left"
            )
            self.result_dfs[key] = counts_df

        # calc sum across all recorded attribute classes
        self.counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=self.counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()
        totals_df['total'] = totals_df.sum(1)

        del self.counts

        key = f"{self.name}"
        totals_df = self.elem_gdf.join(
            totals_df, how="left"
        )
        self.result_dfs[key] = totals_df


class LinkVehicleSpeeds(EventHandlerTool):
    """
    Extract Volume Counts for mode on given network.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.groupby_person_attribute = groupby_person_attribute
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Initialise class attributes
        self.attributes, found_attributes = self.extract_attributes()
        self.classes, self.class_indices = self.generate_elem_ids(found_attributes)
        self.logger.debug(f'available population {self.groupby_person_attribute} values = {self.classes}')

        # generate index and map for network link dimension
        self.elem_gdf = self.resources['network'].link_gdf

        links = resources['network'].mode_to_links_map.get(self.mode)
        if links is None:
            self.logger.warning(
                f"""
                No viable links found for mode:{self.mode} in Network,
                this may be because the Network modes do not match the configured
                modes. Elara will continue with all links found in network.
                """
                )
        else:
            self.logger.debug(f'Selecting links for mode:{self.mode}.')
            self.elem_gdf = self.elem_gdf.loc[links, :]

        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

        # Initialise volume count table
        self.counts = np.zeros((len(self.elem_indices),
                                len(self.classes),
                                self.config.time_periods))

        # Initialise duration cummulative sum table
        self.inverseduration_sum = np.zeros((len(self.elem_indices), len(self.classes), self.config.time_periods))
        self.duration_min = np.zeros((len(self.elem_indices), len(self.classes), self.config.time_periods))
        self.duration_max = np.zeros((len(self.elem_indices), len(self.classes), self.config.time_periods))

        self.link_tracker = dict()  # {(agent,link):start_time}

    def process_event(self, elem) -> None:
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle leaves traffic'
        events to determine average time spent on link.
        :param elem: Event XML element

        The events of interest to this handler look like:

          <event time="300.0"
                 type="vehicle enters traffic"
                 person="nick"
                 link ="1-2"
                 vehicle="nick"
                 networkMode="car"
                 relativePosition="1.0"/>
          <event time="600.0"
                 type="vehicle leaves traffic"
                 person="nick"
                 link ="1-2"
                 vehicle="nick"
                 networkMode="car"
                 relativePosition="1.0"/>

        """

        event_type = elem.get("type")
        if event_type == "entered link":
            ident = elem.get("vehicle")
            veh_mode = self.vehicle_mode(ident)
            if veh_mode == self.mode:
                start_time = float(elem.get("time"))
                self.link_tracker[ident] = (event_type, start_time)

        elif event_type == "left link":
            ident = elem.get("vehicle")
            veh_mode = self.vehicle_mode(ident)
            if veh_mode == self.mode:
                # look for attribute_class, if not found assume pt and use mode
                attribute_class = self.attributes.get(ident, {}).get(self.groupby_person_attribute, None)
                link = elem.get("link")
                end_time = float(elem.get("time"))

                # if person not in link tracker, this means they've entered
                # link via "vehicle enters traffic event" and should be ignored.
                if ident in self.link_tracker:
                    start_time = self.link_tracker[ident][1]
                    # start_event_type = self.link_tracker[ident][0]
                    x, y, z = table_position(
                        self.elem_indices,
                        self.class_indices,
                        self.config.time_periods,
                        link,
                        attribute_class,
                        start_time
                    )

                    duration = end_time - start_time

                    self.counts[x, y, z] += 1

                    if duration != 0:
                        self.inverseduration_sum[x, y, z] += 1/duration

                    self.duration_max[x, y, z] = max(duration, self.duration_max[x, y, z])

                    # needs this condition or else the minimum duration would ever budge from zero
                    if self.duration_min[x, y, z] == 0:
                        self.duration_min[x, y, z] = duration
                    else:
                        self.duration_min[x, y, z] = min(duration, self.duration_min[x, y, z])

    def finalise(self) -> None:
        """
        Following event processing, the raw events table will contain counts by link
        by time slice. The only thing left to do is scale by the sample size and
        create dataframes.
        """

        def calc_av_matrices(self):
            counts_pop = self.counts.sum(1)
            duration_pop = self.inverseduration_sum.sum(1)
            av_pop = np.divide(duration_pop, counts_pop, out=np.zeros_like(counts_pop), where=duration_pop != 0)
            return av_pop

        def calc_max_matrices(self):
            unit_matrix = np.ones((len(self.elem_indices), len(self.classes), self.config.time_periods))
            max_subpop = np.divide(
                unit_matrix, self.duration_min, out=np.zeros_like(unit_matrix), where=self.duration_max != 0
            )
            max_pop = max_subpop.max(1)
            return [max_subpop, max_pop]

        def calc_min_matrices(self):
            unit_matrix = np.ones((len(self.elem_indices), len(self.classes), self.config.time_periods))
            min_subpop = np.divide(
                unit_matrix, self.duration_max, out=np.zeros_like(unit_matrix), where=self.duration_max != 0
            )
            min_pop = min_subpop
            min_pop[min_pop == 0] = np.inf
            min_pop = min_pop.min(1)
            min_subpop[min_subpop == np.inf] = 0
            min_pop[min_pop == np.inf] = 0
            return [min_subpop, min_pop]

        def flatten_subpops(self, subpop_matrix):
            names = ['elem', 'class', 'hour']
            indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
            index = pd.MultiIndex.from_product(indexes, names=names)
            df = pd.DataFrame(subpop_matrix.flatten(), index=index)[0]
            df = df.unstack(level='hour').sort_index()
            df = df.reset_index().set_index('elem')
            return df

        def calc_speeds(self, df):  # converts 1/duration matrix into speeds by multiplying through by length
            for i in range(self.config.time_periods):
                df[i] = df[i] * df["length"]
            return df

        if self.groupby_person_attribute:
            # Calc average at subpop level
            key = f"{self.name}_average_{self.groupby_person_attribute}"
            average_speeds = flatten_subpops(self, self.inverseduration_sum)
            average_speeds = self.elem_gdf.join(average_speeds, how="left")
            average_speeds = calc_speeds(self, average_speeds)
            average_speeds.index.name = "id"
            self.result_dfs[key] = average_speeds

        # Calc average at pop level
        key = f"{self.name}_average"
        average_speeds = calc_av_matrices(self)
        average_speeds = pd.DataFrame(
                data=average_speeds, index=self.elem_ids, columns=range(0, self.config.time_periods)
            ).sort_index()
        average_speeds = self.elem_gdf.join(average_speeds, how="left")
        average_speeds = calc_speeds(self, average_speeds)
        self.result_dfs[key] = average_speeds

        if self.groupby_person_attribute:
            # Calc max at subpop level
            key = f"{self.name}_max_{self.groupby_person_attribute}"
            max_speeds = flatten_subpops(self, calc_max_matrices(self)[0])
            max_speeds = self.elem_gdf.join(max_speeds, how="left")
            max_speeds = calc_speeds(self, max_speeds)
            max_speeds.index.name = "id"
            self.result_dfs[key] = max_speeds

        # Calc max at pop level
        key = f"{self.name}_max"
        max_speeds = calc_max_matrices(self)[1]
        max_speeds = pd.DataFrame(
                data=max_speeds, index=self.elem_ids, columns=range(0, self.config.time_periods)
            ).sort_index()
        max_speeds = self.elem_gdf.join(max_speeds, how="left")
        max_speeds = calc_speeds(self, max_speeds)
        self.result_dfs[key] = max_speeds

        if self.groupby_person_attribute:
            # Calc min at subpop level
            key = f"{self.name}_min_{self.groupby_person_attribute}"
            min_matrix = calc_min_matrices(self)[0]
            min_speeds = flatten_subpops(self, min_matrix)
            min_speeds = self.elem_gdf.join(min_speeds, how="left")
            min_speeds = calc_speeds(self, min_speeds)
            min_speeds.index.name = "id"
            self.result_dfs[key] = min_speeds

        # Calc max at pop level
        key = f"{self.name}_min"
        min_speeds = calc_min_matrices(self)[1]
        min_speeds = pd.DataFrame(
                data=min_speeds, index=self.elem_ids, columns=range(0, self.config.time_periods)
            ).sort_index()
        min_speeds = self.elem_gdf.join(min_speeds, how="left")
        min_speeds = calc_speeds(self, min_speeds)
        self.result_dfs[key] = min_speeds


class LinkPassengerCounts(EventHandlerTool):
    """
    Build Passenger Counts on links for given mode in mode vehicles.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]
    invalid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        self.groupby_person_attribute = groupby_person_attribute
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None
        self.total_counts = None
        self.veh_occupancy = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Check for car
        if self.mode == 'car':
            raise ValueError("Passenger Counts Handlers not intended for use with mode type = car")

        # Initialise class attributes
        self.attributes, found_attributes = self.extract_attributes()
        self.classes, self.class_indices = self.generate_elem_ids(found_attributes)
        self.logger.debug(f'available population {self.groupby_person_attribute} values = {self.classes}')

        # Initialise element attributes
        self.elem_gdf = resources['network'].link_gdf

        links = resources['network'].mode_to_links_map.get(self.mode)
        if links is None:
            self.logger.warning(
                f"""
                No viable links found for mode:{self.mode} in Network,
                this may be because the Network modes do not match the configured
                modes. Elara will continue with all links found in network.
                """
                )
        else:
            self.logger.debug(f'Selecting links for mode:{self.mode}.')
            self.elem_gdf = self.elem_gdf.loc[links, :]

        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

        # Initialise passenger count table
        self.counts = np.zeros((len(self.elem_ids),
                                len(self.classes),
                                self.config.time_periods))

        # Initialise vehicle occupancy mapping
        self.veh_occupancy = dict()  # vehicle_id : occupancy
        self.total_counts = None

    def process_event(self, elem):
        """
        Iteratively aggregate 'PersonEntersVehicle' and 'PersonLeavesVehicle'
        events to determine passenger volumes by link.
        :param elem: Event XML element

        The events of interest to this handler look like:

          <event time="300.0"
                 type="PersonEntersVehicle"
                 person="pt_veh_41173_bus_Bus"
                 vehicle="veh_41173_bus"/>
          <event time="600.0"
                 type="PersonLeavesVehicle"
                 person="pt_veh_41173_bus_Bus"
                 vehicle="veh_41173_bus"/>
          <event time="25656.0"
                 type="left link"
                 vehicle="veh_41173_bus"
                 link="2-3"  />
          <event time="67360.0"
                 type="vehicle leaves traffic"
                 person="pt_bus4_Bus"
                 link="2-1"
                 vehicle="bus4"
                 networkMode="car"
                 relativePosition="1.0"  />
        """
        event_type = elem.get("type")
        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if self.veh_occupancy.get(veh_id, None) is None:
                    self.veh_occupancy[veh_id] = {attribute_class: 1}
                elif not self.veh_occupancy[veh_id].get(attribute_class, None):
                    self.veh_occupancy[veh_id][attribute_class] = 1
                else:
                    self.veh_occupancy[veh_id][attribute_class] += 1

        elif event_type == "PersonLeavesVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if not self.veh_occupancy[veh_id][attribute_class]:
                    pass
                else:
                    self.veh_occupancy[veh_id][attribute_class] -= 1
                    if not self.veh_occupancy[veh_id]:
                        self.veh_occupancy.pop(veh_id, None)

        elif event_type == "left link" or event_type == "vehicle leaves traffic":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.mode:
                # Increment link passenger volumes
                time = float(elem.get("time"))
                link = elem.get("link")
                occupancy_dict = self.veh_occupancy.get(veh_id, {})

                for attribute_class, occupancy in occupancy_dict.items():
                    x, y, z = table_position(
                        self.elem_indices,
                        self.class_indices,
                        self.config.time_periods,
                        link,
                        attribute_class,
                        time
                    )
                    self.counts[x, y, z] += occupancy

    def finalise(self):
        """
        Following event processing, the raw events table will contain passenger
        counts by link by time slice. The only thing left to do is scale by the
        sample size and create dataframes.
        """

        # Scale final counts
        self.counts *= 1.0 / self.config.scale_factor

        if self.groupby_person_attribute:
            names = ['elem', self.groupby_person_attribute, 'hour']
            indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
            index = pd.MultiIndex.from_product(indexes, names=names)
            counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]
            counts_df = counts_df.unstack(level='hour').sort_index()
            counts_df = counts_df.reset_index().set_index(['elem', self.groupby_person_attribute])
            counts_df['total'] = counts_df.sum(1)
            counts_df = counts_df.reset_index().set_index('elem')
            key = f"{self.name}_{self.groupby_person_attribute}"
            counts_df = self.elem_gdf.join(
                counts_df, how="left"
            )
            self.result_dfs[key] = counts_df

        # calc sum across all recorded attribute classes
        self.counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=self.counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()
        totals_df['total'] = totals_df.sum(1)

        del self.counts

        key = f"{self.name}"
        totals_df = self.elem_gdf.join(
            totals_df, how="left"
        )
        self.result_dfs[key] = totals_df


class RoutePassengerCounts(EventHandlerTool):
    """
    Build Passenger Counts per transit route for given mode.
    """
    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]
    invalid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        self.groupby_person_attribute = groupby_person_attribute
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None
        self.total_counts = None
        self.route_occupancy = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Initialise class attributes
        self.attributes, found_attributes = self.extract_attributes()
        self.classes, self.class_indices = self.generate_elem_ids(found_attributes)
        self.logger.debug(f'available population {self.groupby_person_attribute} values = {self.classes}')

        # # Initialise element attributes
        # all_routes = set(resources['transit_schedule'].veh_to_route_map.values())
        # get routes used by this mode
        self.logger.debug(f'Selecting routes for mode:{self.mode}.')
        routes = resources['transit_schedule'].mode_to_routes_map.get(self.mode)
        if routes is None:
            self.logger.warning(
                f"""
                No viable routes found for mode:{self.mode} in TransitSchedule,
                this may be because the Schedule modes do not match the configured
                modes. Elara will continue with all routes found in schedule.
                """
                )

        self.elem_ids, self.elem_indices = self.generate_elem_ids(list(routes))

        # Initialise passenger count table
        self.counts = np.zeros((len(self.elem_ids),
                                len(self.classes),
                                self.config.time_periods))

        self.route_occupancy = dict()

    def process_event(self, elem):
        """
        Iteratively aggregate 'PersonEntersVehicle' and 'PersonLeavesVehicle'
        events to determine passenger volumes by route.
        :param elem: Event XML element

        The events of interest to this handler look like:

          <event time="300.0"
                 type="PersonEntersVehicle"
                 person="pt_veh_41173_bus_Bus"
                 vehicle="veh_41173_bus"/>
          <event time="600.0"
                 type="PersonLeavesVehicle"
                 person="pt_veh_41173_bus_Bus"
                 vehicle="veh_41173_bus"/>
          <event time="25656.0"
                 type="left link"
                 vehicle="veh_41173_bus"
                 link="2-3"  />
          <event time="67360.0"
                 type="vehicle leaves traffic"
                 person="pt_bus4_Bus"
                 link="2-1"
                 vehicle="bus4"
                 networkMode="car"
                 relativePosition="1.0"  />
        """
        event_type = elem.get("type")
        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)
            veh_route = self.vehicle_route(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if self.route_occupancy.get(veh_route, None) is None:
                    self.route_occupancy[veh_route] = {attribute_class: 1}
                elif not self.route_occupancy[veh_route].get(attribute_class, None):
                    self.route_occupancy[veh_route][attribute_class] = 1
                else:
                    self.route_occupancy[veh_route][attribute_class] += 1
        elif event_type == "PersonLeavesVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)
            veh_route = self.vehicle_route(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if self.route_occupancy[veh_route][attribute_class]:
                    self.route_occupancy[veh_route][attribute_class] -= 1
                    if not self.route_occupancy[veh_route]:
                        self.route_occupancy.pop(veh_route, None)
        elif event_type == "left link" or event_type == "vehicle leaves traffic":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)
            veh_route = self.vehicle_route(veh_id)

            if veh_mode == self.mode:
                time = float(elem.get("time"))
                occupancy_dict = self.route_occupancy.get(veh_route, {})

                for attribute_class, occupancy in occupancy_dict.items():
                    x, y, z = table_position(
                        self.elem_indices,
                        self.class_indices,
                        self.config.time_periods,
                        veh_route,
                        attribute_class,
                        time
                    )
                    self.counts[x, y, z] += occupancy

    def finalise(self):
        """
        Following event processing, the raw events table will contain passenger
        counts by route by time slice. The only thing left to do is scale by the
        sample size and create dataframes.
        """

        del self.route_occupancy

        # Scale final counts
        self.counts *= 1.0 / self.config.scale_factor

        if self.groupby_person_attribute:
            names = ['elem', self.groupby_person_attribute, 'hour']
            indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
            index = pd.MultiIndex.from_product(indexes, names=names)
            counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]
            counts_df = counts_df.unstack(level='hour').sort_index()
            counts_df = counts_df.reset_index().set_index('elem')

            # Create volume counts output
            key = f"{self.name}_{self.groupby_person_attribute}"
            self.result_dfs[key] = counts_df

        # calc sum across all recorded attribute classes
        self.counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=self.counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()

        del self.counts

        key = f"{self.name}"
        self.result_dfs[key] = totals_df


class StopPassengerCounts(EventHandlerTool):
    """
    Stop alightings and boardings handler for given mode.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]
    invalid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        self.groupby_person_attribute = groupby_person_attribute
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.boardings = None
        self.alightings = None
        self.agent_status = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Check for car
        if self.mode == 'car':
            raise ValueError("Stop Interaction Handlers not intended for use with mode type = car")

        # Initialise class attributes
        self.attributes, found_attributes = self.extract_attributes()
        self.classes, self.class_indices = self.generate_elem_ids(found_attributes)
        self.logger.debug(f'available population {self.groupby_person_attribute} values = {self.classes}')

        # Initialise element attributes
        self.elem_gdf = resources['transit_schedule'].stop_gdf
        # get stops used by this mode
        viable_stops = resources['transit_schedule'].mode_to_stops_map.get(self.mode)

        if viable_stops is None:
            self.logger.warning(
                f"""
                No viable stops found for mode:{self.mode} in TransitSchedule,
                this may be because the Schedule modes do not match the configured
                modes. Elara will continue with all stops found in schedule.
                """
                )
        else:
            self.logger.debug(f'Filtering stops for mode:{self.mode}.')
            self.elem_gdf = self.elem_gdf.loc[viable_stops, :]

        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

        # Initialise results tables
        self.boardings = np.zeros((len(self.elem_indices),
                                   len(self.class_indices),
                                   self.config.time_periods))
        self.alightings = np.zeros((len(self.elem_indices),
                                    len(self.class_indices),
                                    self.config.time_periods))

        # Initialise agent status mapping
        self.agent_status = dict()  # agent_id : [origin_stop, destination_stop]

    def process_event(self, elem):
        """
        Iteratively aggregate 'waitingForPt', 'PersonEntersVehicle' and
        'PersonLeavesVehicle' events to determine stop boardings and alightings.
        :param elem: Event XML element
        """
        event_type = elem.get("type")

        if event_type == "waitingForPt":
            agent_id = elem.get("agent")
            origin_stop = elem.get("atStop")
            destination_stop = elem.get("destinationStop")
            self.agent_status[agent_id] = [origin_stop, destination_stop]

        elif event_type == "PersonEntersVehicle":
            veh_mode = self.vehicle_mode(elem.get("vehicle"))

            if veh_mode == self.mode:
                agent_id = elem.get("person")

                if self.agent_status.get(agent_id, None) is not None:
                    time = float(elem.get("time"))
                    attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)
                    origin_stop = self.agent_status[agent_id][0]
                    x, y, z = table_position(
                        self.elem_indices,
                        self.class_indices,
                        self.config.time_periods,
                        origin_stop,
                        attribute_class,
                        time
                    )
                    self.boardings[x, y, z] += 1

        elif event_type == "PersonLeavesVehicle":
            veh_mode = self.vehicle_mode(elem.get("vehicle"))

            if veh_mode == self.mode:
                agent_id = elem.get("person")

                if self.agent_status.get(agent_id, None) is not None:
                    time = float(elem.get("time"))
                    attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)
                    destination_stop = self.agent_status[agent_id][1]
                    x, y, z = table_position(
                        self.elem_indices,
                        self.class_indices,
                        self.config.time_periods,
                        destination_stop,
                        attribute_class,
                        time
                    )
                    self.alightings[x, y, z] += 1
                    self.agent_status.pop(agent_id, None)

    def finalise(self):
        """
        Following event processing, the raw events table will contain boardings
        and alightings by link by time slice. The only thing left to do is scale
        by the sample size and create dataframes.
        """
        del self.agent_status

        # Scale final counts
        self.boardings *= 1.0 / self.config.scale_factor
        self.alightings *= 1.0 / self.config.scale_factor

        # Create passenger counts output
        for data, direction in zip([self.boardings, self.alightings], ['boardings', 'alightings']):

            if self.groupby_person_attribute:
                names = ['elem', self.groupby_person_attribute, 'hour']
                indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
                index = pd.MultiIndex.from_product(indexes, names=names)
                counts_df = pd.DataFrame(data.flatten(), index=index)[0]
                counts_df = counts_df.unstack(level='hour').sort_index()
                counts_df = counts_df.reset_index().set_index(['elem', self.groupby_person_attribute])
                counts_df['total'] = counts_df.sum(1)
                counts_df = counts_df.reset_index().set_index('elem')

                # Create volume counts output
                key = f"{self.name}_{direction}_{self.groupby_person_attribute}"
                counts_df = self.elem_gdf.join(
                    counts_df, how="left"
                )
                self.result_dfs[key] = counts_df

            # calc sum across all recorded attribute classes
            data = data.sum(1)

            totals_df = pd.DataFrame(
                data=data, index=self.elem_ids, columns=range(0, self.config.time_periods)
            ).sort_index()
            totals_df['total'] = totals_df.sum(1)

            del data

            key = f"{self.name}_{direction}"
            totals_df = self.elem_gdf.join(
                totals_df, how="left"
            )
            self.result_dfs[key] = totals_df


class StopToStopPassengerCounts(EventHandlerTool):
    """
    Build Passenger Counts between stops for given mode in mode vehicles.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]
    invalid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        self.groupby_person_attribute = groupby_person_attribute
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None
        self.veh_occupancy = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Check for car
        if self.mode in ['car', 'walk', 'bike']:
            raise ValueError(f"Passenger Counts Handlers not intended for use with mode type = {self.mode}")

        # Initialise class attributes
        self.attributes, found_attributes = self.extract_attributes()
        self.classes, self.class_indices = self.generate_elem_ids(found_attributes)
        self.logger.debug(f'available population {self.groupby_person_attribute} values = {self.classes}')

        # Initialise element attributes
        self.elem_gdf = resources['transit_schedule'].stop_gdf
        # get stops used by this mode
        viable_stops = resources['transit_schedule'].mode_to_stops_map.get(self.mode)
        if viable_stops is None:
            self.logger.warning(
                f"""
                No viable stops found for mode:{self.mode} in TransitSchedule,
                this may be because the Schedule modes do not match the configured
                modes. Elara will continue with all stops found in schedule.
                """
                )
        else:
            self.logger.debug(f'Filtering stops for mode:{self.mode}.')
            self.elem_gdf = self.elem_gdf.loc[viable_stops, :]

        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

        # Initialise results tablescd
        self.counts = np.zeros((
            len(self.elem_indices),
            len(self.elem_indices),
            len(self.class_indices),
            self.config.time_periods
            ))

        # Initialise agent status mapping
        self.veh_occupancy = dict()  # {veh_id : {attribute_class: COUNT}}
        self.veh_tracker = dict()  # {veh_id: last_stop}

    def process_event(self, elem):
        """
        Iteratively aggregate 'PersonEntersVehicle' and 'PersonLeavesVehicle'
        events to determine passenger volumes by stop interactions.
        :param elem: Event XML element

        The events of interest to this handler look like:

            <event time="300.0"
                type="PersonEntersVehicle"
                person="pt_veh_41173_bus_Bus"
                vehicle="veh_41173_bus"/>
            <event time="600.0"
                type="PersonLeavesVehicle"
                person="pt_veh_41173_bus_Bus"
                vehicle="veh_41173_bus"/>
            <event time="27000.0"
                type="VehicleArrivesAtFacility"
                vehicle="bus1"
                facility="home_stop_out"
                delay="Infinity"/>
        """
        event_type = elem.get("type")

        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if self.veh_occupancy.get(veh_id, None) is None:
                    self.veh_occupancy[veh_id] = {attribute_class: 1}
                elif not self.veh_occupancy[veh_id].get(attribute_class, None):
                    self.veh_occupancy[veh_id][attribute_class] = 1
                else:
                    self.veh_occupancy[veh_id][attribute_class] += 1

        elif event_type == "PersonLeavesVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if self.veh_occupancy[veh_id][attribute_class]:
                    self.veh_occupancy[veh_id][attribute_class] -= 1
                    if not self.veh_occupancy[veh_id]:
                        self.veh_occupancy.pop(veh_id, None)

        elif event_type == "VehicleArrivesAtFacility":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.mode:
                stop_id = elem.get('facility')
                prev_stop_id = self.veh_tracker.get(veh_id, None)
                self.veh_tracker[veh_id] = stop_id

                if prev_stop_id is not None:
                    time = float(elem.get("time"))
                    occupancy_dict = self.veh_occupancy.get(veh_id, {})

                    for attribute_class, occupancy in occupancy_dict.items():
                        o, d, y, z = table_position_4d(
                            self.elem_indices,
                            self.elem_indices,
                            self.class_indices,
                            self.config.time_periods,
                            prev_stop_id,
                            stop_id,
                            attribute_class,
                            time
                        )
                        self.counts[o, d, y, z] += occupancy

    def finalise(self):
        """
        Following event processing, the raw events table will contain passenger
        counts by od pair, attribute class and time slice. The only thing left to do is scale by the
        sample size and create dataframes.
        """
        # TODO this is a mess. requires some forcing to string hacks for None. The pd ops are forcing None to np.nan
        del self.veh_occupancy

        # Scale final counts
        self.counts *= 1.0 / self.config.scale_factor

        names = ['origin', 'destination', str(self.groupby_person_attribute), 'hour']
        self.classes = [str(c) for c in self.classes]
        indexes = [self.elem_ids, self.elem_ids, self.classes, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]

        del self.counts

        counts_df = counts_df.unstack(level='hour').sort_index()

        # Join stop data and build geometry
        for n in ("origin", "destination"):
            counts_df = counts_df.reset_index().set_index(n)
            stop_info = self.elem_gdf.copy()
            stop_info.columns = [f"{n}_{c}" for c in stop_info.columns]
            counts_df = counts_df.join(
                    stop_info, how="left"
                )

            counts_df.index.name = n

        counts_df = counts_df.reset_index().set_index(['origin', 'destination', str(self.groupby_person_attribute)])
        counts_df['total'] = counts_df.sum(1)

        counts_df['geometry'] = [
            LineString([o, d]) for o, d in zip(counts_df.origin_geometry, counts_df.destination_geometry)
        ]
        counts_df.drop('origin_geometry', axis=1, inplace=True)
        counts_df.drop('destination_geometry', axis=1, inplace=True)
        counts_df = gpd.GeoDataFrame(counts_df, geometry='geometry')

        if self.groupby_person_attribute:
            key = f"{self.name}_{self.groupby_person_attribute}"
            self.result_dfs[key] = counts_df

        # TODO make below pandas ops more efficient

        # calc sum across all recorded attribute classes
        totals_df = counts_df.reset_index().groupby(
            ['origin', 'destination']
            ).sum().reset_index().set_index(['origin', 'destination'])

        # Join stop data and build geometry
        for n in ("origin", "destination"):
            totals_df = totals_df.reset_index().set_index(n)
            stop_info = self.elem_gdf.copy()
            stop_info.columns = [f"{n}_{c}" for c in stop_info.columns]
            totals_df = totals_df.join(
                    stop_info, how="left"
                )
            totals_df.index.name = n

        totals_df = totals_df.reset_index().set_index(['origin', 'destination'])

        totals_df['geometry'] = [
            LineString([o, d]) for o, d in zip(totals_df.origin_geometry, totals_df.destination_geometry)
        ]
        totals_df.drop('origin_geometry', axis=1, inplace=True)
        totals_df.drop('destination_geometry', axis=1, inplace=True)
        totals_df = gpd.GeoDataFrame(totals_df, geometry='geometry')

        totals_df = gpd.GeoDataFrame(totals_df, geometry='geometry')
        key = f"{self.name}"
        self.result_dfs[key] = totals_df


class VehicleStopToStopPassengerCounts(EventHandlerTool):
    """
    Build Passenger Counts between stops for given mode in mode vehicles.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]
    invalid_modes = ['car']

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        self.groupby_person_attribute = groupby_person_attribute
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None
        self.veh_occupancy = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Check for car
        if self.mode in ['car', 'walk', 'bike']:
            raise ValueError(f"Passenger Counts Handlers not intended for use with mode type = {self.mode}")

        # Initialise class attributes
        self.attributes, found_attributes = self.extract_attributes()
        self.classes, self.class_indices = self.generate_elem_ids(found_attributes)
        self.logger.debug(f'available population {self.groupby_person_attribute} values = {self.classes}')

        # Get vehicle IDs and generate vehicle indices
        veh_ids_list = resources['transit_schedule'].mode_to_veh_map.get(self.mode)
        self.veh_ids, self.veh_ids_indices = self.generate_elem_ids(veh_ids_list)

        # Vehicle --> route map
        self.veh_route = resources['transit_schedule'].veh_to_route_map

        # Initialise element attributes
        self.elem_gdf = resources['transit_schedule'].stop_gdf
        # get stops used by this mode
        viable_stops = resources['transit_schedule'].mode_to_stops_map.get(self.mode)
        if viable_stops is None:
            self.logger.warning(
                f"""
                No viable stops found for mode:{self.mode} in TransitSchedule,
                this may be because the Schedule modes do not match the configured
                modes. Elara will continue with all stops found in schedule.
                """
                )
        else:
            self.logger.debug(f'Filtering stops for mode:{self.mode}.')
            self.elem_gdf = self.elem_gdf.loc[viable_stops, :]

        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_gdf)

        # Initialise results dictionary
        self.counts = dict()  # passenger counts
        self.veh_counts = dict()  # vehicle counts

        # Initialise agent status mapping
        self.veh_occupancy = dict()  # {veh_id : {attribute_class: COUNT}}
        self.veh_tracker = dict()  # {veh_id: last_stop}

    def process_event(self, elem):
        """
        Iteratively aggregate 'PersonEntersVehicle' and 'PersonLeavesVehicle'
        events to determine passenger volumes by stop interactions.
        :param elem: Event XML element

        The events of interest to this handler look like:

            <event time="300.0"
                type="PersonEntersVehicle"
                person="pt_veh_41173_bus_Bus"
                vehicle="veh_41173_bus"/>
            <event time="600.0"
                type="PersonLeavesVehicle"
                person="pt_veh_41173_bus_Bus"
                vehicle="veh_41173_bus"/>
            <event time="27000.0"
                type="VehicleArrivesAtFacility"
                vehicle="bus1"
                facility="home_stop_out"
                delay="Infinity"/>
        """
        event_type = elem.get("type")

        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if self.veh_occupancy.get(veh_id, None) is None:
                    self.veh_occupancy[veh_id] = {attribute_class: 1}
                elif not self.veh_occupancy[veh_id].get(attribute_class, None):
                    self.veh_occupancy[veh_id][attribute_class] = 1
                else:
                    self.veh_occupancy[veh_id][attribute_class] += 1

        elif event_type == "PersonLeavesVehicle":
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            # Filter out PT drivers from transit volume statistics
            if agent_id[:2] != "pt" and veh_mode == self.mode:
                attribute_class = self.attributes.get(agent_id, {}).get(self.groupby_person_attribute, None)

                if self.veh_occupancy[veh_id][attribute_class]:
                    self.veh_occupancy[veh_id][attribute_class] -= 1
                    if not self.veh_occupancy[veh_id]:
                        self.veh_occupancy.pop(veh_id, None)

        elif event_type == "VehicleArrivesAtFacility":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.mode:
                stop_id = elem.get('facility')
                prev_stop_id = self.veh_tracker.get(veh_id, None)
                self.veh_tracker[veh_id] = stop_id

                if prev_stop_id is not None:
                    time = float(elem.get("time"))
                    hour = floor(time / (86400.0 / self.config.time_periods)) % self.config.time_periods
                    occupancy_dict = self.veh_occupancy.get(veh_id, {})

                    for attribute_class, occupancy in occupancy_dict.items():
                        midx = (prev_stop_id, stop_id, veh_id, attribute_class, hour)
                        if self.counts.get(midx) is None:
                            self.counts[midx] = occupancy
                            self.veh_counts[midx] = 1
                        else:
                            # self.logger.warning(f'Vehicle {veh_id} arrives at stop {stop_id} more than once.')
                            self.counts[midx] += occupancy
                            self.veh_counts[midx] += 1

    def finalise(self):
        """
        Following event processing, the raw events table will contain passenger
        counts by od pair, attribute class and time slice. The only thing left to do is scale by the
        sample size and create dataframes.
        """
        # TODO this is a mess. requires some forcing to string hacks for None. The pd ops are forcing None to np.nan
        del self.veh_occupancy

        # Check if counts dictionary exists
        if self.counts:
            names = ['from_stop', 'to_stop', 'veh_id']
            if self.groupby_person_attribute:
                names = ['from_stop', 'to_stop', 'veh_id', str(self.groupby_person_attribute)]

            counts_df = pd.Series(self.counts)
            # include vehicle counts (in case a vehicle arrives at a stop more than once)
            counts_df = pd.concat([counts_df, pd.Series(self.veh_counts)], axis=1)
            counts_df.index.names = names + ['to_stop_arrival_hour']
            counts_df.columns = ['pax_counts', 'veh_counts']
            # move vehicle counts to the series index
            counts_df = counts_df.reset_index().set_index(names + ['to_stop_arrival_hour', 'veh_counts'])['pax_counts']

            # scale
            counts_df *= 1.0 / self.config.scale_factor

            del self.counts
            counts_df = counts_df.unstack(level='to_stop_arrival_hour').sort_index().fillna(0)

            # Join stop data and build geometry
            for n in ("from_stop", "to_stop"):
                counts_df = counts_df.reset_index().set_index(n)
                stop_info = self.elem_gdf.copy()
                stop_info.columns = [f"{n}_{c}" for c in stop_info.columns]
                counts_df = counts_df.join(
                        stop_info, how="left"
                    )

                counts_df.index.name = n

            counts_df = counts_df.reset_index().set_index(names+['veh_counts'])
            counts_df['route'] = counts_df.index.get_level_values('veh_id').map(self.veh_route)
            counts_df['total'] = counts_df.sum(1)

            counts_df['geometry'] = [LineString([o, d]) for o, d in zip(
                counts_df.from_stop_geometry, counts_df.to_stop_geometry)]
            counts_df.drop('from_stop_geometry', axis=1, inplace=True)
            counts_df.drop('to_stop_geometry', axis=1, inplace=True)
            counts_df = gpd.GeoDataFrame(counts_df, geometry='geometry')

            #################
            # temp: unit tests currently require all hours of the day as columns
            # TODO: planning to remove this requirement - then delete this code block
            for h in range(0, 24):
                if h not in counts_df.columns:
                    counts_df[h] = 0
            #################

            if self.groupby_person_attribute:
                key = f"{self.name}_{self.groupby_person_attribute}"
                self.result_dfs[key] = counts_df

            # # calc sum across all recorded attribute classes
            totals_df = counts_df.reset_index().groupby(['from_stop', 'to_stop', 'veh_id', 'route', 'veh_counts']).sum()

            # Join stop data and build geometry
            for n in ("from_stop", "to_stop"):
                totals_df = totals_df.reset_index().set_index(n)
                stop_info = self.elem_gdf.copy()
                stop_info.columns = [f"{n}_{c}" for c in stop_info.columns]
                totals_df = totals_df.join(
                        stop_info, how="left"
                    )
                totals_df.index.name = n

            totals_df['geometry'] = [
                LineString([o, d]) for o, d in zip(totals_df.from_stop_geometry, totals_df.to_stop_geometry)
            ]
            totals_df.drop('from_stop_geometry', axis=1, inplace=True)
            totals_df.drop('to_stop_geometry', axis=1, inplace=True)
            totals_df = gpd.GeoDataFrame(totals_df, geometry='geometry')

            key = f"{self.name}"
            self.result_dfs[key] = totals_df

        else:
            self.logger.warn('Vehicle counts dictionary is empty!!! No VehicleArrivesAtFacility events found!!!')


class VehicleDepartureLog(EventHandlerTool):
    """
    Extract vehicle depart times at stops.
    """

    requirements = ['events', 'transit_schedule']

    def __init__(self, config, mode="all", **kwargs):
        super().__init__(config, mode)
        self.vehicle_departure_log = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """

        super().build(resources, write_path=write_path)

        pt_csv_name = f"{self.name}.csv"

        self.vehicle_departure_log = self.start_chunk_writer(
            pt_csv_name, write_path=write_path
            )

    def process_event(self, elem) -> None:
        """
        :param elem: Event XML element
        """
        event_type = elem.get("type")

        if event_type == 'VehicleDepartsAtFacility':

            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)
            veh_route = self.vehicle_route(veh_id)
            stop_id = elem.get("facility")
            departure_time = int(float(elem.get("time")))
            delay = int(float(elem.get("delay")))

            if veh_mode == self.mode or self.mode == "all":  # None = all modes

                pt_departures = [

                    {
                        'veh_id': veh_id,
                        'veh_mode': veh_mode,
                        'veh_route': veh_route,
                        'stop_id': stop_id,
                        'departure_time': departure_time,
                        'delay': delay
                    }
                ]

                self.vehicle_departure_log.add(pt_departures)

        return None

    def finalise(self):
        self.vehicle_departure_log.finish()


class VehiclePassengerLog(EventHandlerTool):
    """
    Extract a log of passenger boardings and alightings to a PT vehicle.
    """

    requirements = ['events', 'transit_schedule']

    def __init__(self, config, mode="all", **kwargs):
        super().__init__(config, mode)
        self.vehicle_passenger_log = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        pt_csv_name = f"{self.name}.csv"
        self.veh_tracker = dict()  # {veh_id: last_stop}

        self.vehicle_passenger_log = self.start_chunk_writer(
            pt_csv_name, write_path=write_path
            )

    def process_event(self, elem) -> None:
        """
        :param elem: Event XML element
        """
        event_type = elem.get("type")

        # keep track of the last vehicle stop
        if event_type == 'VehicleArrivesAtFacility':
            veh_id = elem.get("vehicle")
            # veh_mode = self.vehicle_mode(veh_id)
            stop_id = elem.get("facility")
            self.veh_tracker[veh_id] = stop_id

        # add boardings/alightings to the log
        if event_type in ['PersonEntersVehicle', 'PersonLeavesVehicle']:
            agent_id = elem.get("person")
            veh_id = elem.get("vehicle")
            veh_route = self.vehicle_route(veh_id)
            event_time = elem.get("time")
            veh_mode = self.vehicle_mode(veh_id)
            stop_id = self.veh_tracker.get(veh_id, None)

            if veh_mode == self.mode or self.mode == "all":
                if agent_id[:2] != "pt":  # Filter out PT drivers from transit volume statistics

                    boardings = [
                        {
                            'agent_id': agent_id,
                            'event_type': event_type,
                            'veh_id': veh_id,
                            'stop_id': stop_id,
                            'time': event_time,
                            'veh_mode': veh_mode,
                            'veh_route': veh_route,
                        }
                    ]
                    self.vehicle_passenger_log.add(boardings)

        return None

    def finalise(self):
        self.vehicle_passenger_log.finish()


class VehicleLinkLog(EventHandlerTool):
    """
    Extract all vehicle link entry/link exit events
    """

    requirements = ['events', 'transit_schedule']

    def __init__(self, config, mode=None, **kwargs):
        super().__init__(config, mode)
        self.vehicle_link_log = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """

        super().build(resources, write_path=write_path)

        file_name = f"{self.name}.csv"

        self.vehicle_link_log = self.start_chunk_writer(
            file_name, write_path=write_path
        )

        # Only add to chunk writer when entry + exit complete
        self.event_staging = {}

    def process_event(self, elem) -> None:
        """
        Events are closed only when a vehicle enters then exits a link
        Events are staged on entry, then added to chunk writer when closed
        :param elem: Event XML element
        """
        event_type = elem.get("type")

        if event_type == "entered link":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)
            link_id = elem.get("link")
            entry_time = int(float(elem.get("time")))

            if veh_mode == self.mode or self.mode == "all":
                entry = {
                    "veh_id": veh_id,
                    "veh_mode": self.vehicle_mode(veh_id),
                    "link_id": link_id,
                    "entry_time": entry_time
                }

                self.event_staging[veh_id] = entry

        if event_type == "left link":
            veh_id = elem.get("vehicle")

            entry = self.event_staging.pop(veh_id, None)

            if entry is not None:
                entry["exit_time"] = int(float(elem.get("time")))
                self.vehicle_link_log.add([entry])

        if event_type == "vehicle leaves traffic":  # exit via leaves traffic event
            veh_id = elem.get("vehicle")
            # remove staged event. None = veh enters/leaves traffic on same link
            self.event_staging.pop(veh_id, None)

        return None

    def finalise(self):
        self.vehicle_link_log.finish()


class AgentTollsLog(EventHandlerTool):
    """
    Produces a raw log of tolling events by agent
    Additionally produces a 24-hr summary of tolls paid by each agent
    """

    requirements = ['events', 'attributes']

    def __init__(self, config, mode=None, groupby_person_attribute=None, **kwargs):
        super().__init__(config, mode)

        self.groupby_person_attribute = groupby_person_attribute
        self.valid_modes = ['all']

        # initialise results storage
        self.result_dfs = dict()
        self.agent_tolls_log = None
        self.toll_log_summary = dict()

        # keep track of pt drivers paying tolls
        self.tolled_pt = []

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """

        super().build(resources, write_path=write_path)

        self.agent_attributes, found_attributes = self.extract_attributes()

        file_name = f'{self.name}.csv'

        self.agent_tolls_log = self.start_chunk_writer(
            file_name, write_path=write_path
        )

    def process_event(self, elem) -> None:
        '''
        Logs tolling events using ChunkWriter.
        Additionally, add agents and toll amounts to dictionary
        Tolls paid are incremented over the 24hr simulation

        Events of interest look like:
        <event time="100" type="personMoney" person="agentID"
        amount="-1.0" purpose="toll"  />

        TODO - Mode filters (don't have veh id)
        '''

        event_type = elem.get("type")

        if event_type == "personMoney":
            if elem.get("purpose") == "toll":
                agent_id = elem.get("person")

                if agent_id[:2] == "pt":  # filter out pt drivers, and break
                    self.tolled_pt.append(agent_id)
                    return None

                toll_amount = float(elem.get("amount")) * -1
                time = float(elem.get("time"))
                attrib = None

                toll_event = [
                    {
                        'agent_id': agent_id,
                        'toll_amount': toll_amount,
                        'time': time
                    }
                ]

                if self.groupby_person_attribute is not None:
                    attrib = self.agent_attributes.attributes.get(agent_id, {}).get(self.groupby_person_attribute)
                    toll_event[0]['class'] = attrib

                # Add to ChunkWriter and update summary dictionaries
                self.agent_tolls_log.add(toll_event)

                existing_record = self.toll_log_summary.get(agent_id)

                if existing_record is not None:
                    existing_record['toll_total'] += toll_amount
                    existing_record['tolls_incurred'] += 1
                else:
                    self.toll_log_summary[agent_id] = {
                        'toll_total': toll_amount,
                        'tolls_incurred': 1
                    }

                    if attrib is not None:
                        self.toll_log_summary[agent_id]['class'] = attrib

        return None

    def finalise(self):

        self.agent_tolls_log.finish()

        # warning about pt tolling
        if self.tolled_pt:
            count_pt = len(self.tolled_pt)
            self.logger.warning(f"{count_pt} PT vehicles incurred tolls. These are excluded from logs")

        # build summaries
        df = pd.DataFrame.from_dict(self.toll_log_summary, orient="index")
        df.index.name = 'agent_id'

        key = f"{self.name}_summary"
        self.result_dfs[key] = df

        if self.groupby_person_attribute:

            if "class" in df.columns:

                key = f"{self.name}_summary_{self.groupby_person_attribute}"
                grouper = df.groupby('class')
                df_grouped = grouper.agg({'toll_total': ['sum', 'mean', 'count'], 'tolls_incurred': 'sum'})
                df_grouped.droplevel(0, axis=1)
                df_grouped.columns = [
                    'toll_total', 'avg_per_agent', 'tolled_agents', 'tolls_incurred'
                ]
                self.result_dfs[key] = df_grouped
            else:
                self.logger.warning(
                    "groupby person attribute failed for {self} - no 'class' column found in df"
                    )

        del self.toll_log_summary


class EventHandlerWorkStation(WorkStation):

    """
    Work Station for holding and building Event Handlers.
    """

    tools = {
        "link_vehicle_speeds": LinkVehicleSpeeds,
        "link_vehicle_counts": LinkVehicleCounts,
        "link_passenger_counts": LinkPassengerCounts,
        "route_passenger_counts": RoutePassengerCounts,
        "stop_passenger_counts": StopPassengerCounts,
        "stop_passenger_waiting": StopPassengerWaiting,
        "vehicle_passenger_graph": VehiclePassengerGraph,
        "stop_to_stop_passenger_counts": StopToStopPassengerCounts,
        "vehicle_stop_to_stop_passenger_counts": VehicleStopToStopPassengerCounts,
        "vehicle_departure_log": VehicleDepartureLog,
        "vehicle_passenger_log": VehiclePassengerLog,
        "vehicle_link_log": VehicleLinkLog,
        "agent_tolls_log": AgentTollsLog
    }

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def build(self, write_path=None) -> None:
        """
        Build all required handlers, then finalise and save results.
        :param write_path: Optional output path overwrite
        :return: None
        """

        if not self.resources:
            self.logger.warning(f'{str(self)} has no resources, build returning None.')
            return None

        # build tools
        super().build(write_path=write_path)

        # iterate through events
        events = self.supplier_resources['events']
        self.logger.info('***Commencing Event Iteration***')
        base = 1

        for i, event in enumerate(events.elems):

            if not (i+1) % base:
                self.logger.info(f'parsed {i + 1} events')
                base *= 2

            for handler in self.resources.values():
                handler.process_event(event)

        self.logger.info('*** Completed Event Iteration ***')

        # finalise
        # Generate event file outputs
        self.logger.debug(f'{str(self)} .resources = {self.resources}')

        for handler_name, handler in self.resources.items():
            self.logger.debug(f'Finalising {str(handler)}')
            handler.finalise()

            if self.config.contract:
                self.logger.debug(f'Contracting {str(handler)}')
                handler.contract_results()

            self.logger.debug(f'{len(handler.result_dfs)} result_dfs at {str(handler)}')

            if handler.result_dfs:
                output_path = handler.config.output_path
                self.logger.info(f'Writing results from {str(handler)} to {output_path}')

                for name, df in handler.result_dfs.items():
                    csv_name = "{}.csv".format(name)
                    geojson_name = "{}.geojson".format(name)

                    self.write_csv(df, csv_name, write_path=write_path)
                    if isinstance(df, gpd.GeoDataFrame):
                        self.write_geojson(df, geojson_name, write_path=write_path)

                    del df


def table_position(elem_indices, class_indices, periods, elem_id, attribute_class, time):
    """
    Calculate the result table coordinates from a given a element ID, attribute class and timestamp.
    :param elem_indices: Element index list
    :param class_indices: attribute index list
    :param periods: Number of time periods across the day
    :param elem_id: Element ID string
    :param attribute_class: Class ID string
    :param time: Timestamp of event
    :return: (x, y, z) tuple to index results table
    """
    x = elem_indices[elem_id]
    y = class_indices[attribute_class]
    z = floor(time / (86400.0 / periods)) % periods
    return x, y, z


def table_position_4d(origin_elem_indices, destination_elem_indices, class_indices,
                      periods, o_id, d_id, attribute_class, time):
    """
    Calculate the result table coordinates from a given a origin element ID,
    destination elmement ID, attribute class and timestamp.

    :param origin_elem_indices: Element index list
    :param destination_elem_indices: Element index list
    :param class_indices: attribute index list
    :param periods: Number of time periods across the day
    :param o_id: Element ID string
    :param d_id: Element ID string
    :param attribute_class: Class ID string
    :param time: Timestamp of event
    :return: (0, d, y, z) tuple to index results table
    """
    o = origin_elem_indices[o_id]
    d = destination_elem_indices[d_id]
    y = class_indices[attribute_class]
    z = floor(time / (86400.0 / periods)) % periods
    return o, d, y, z


def export_geojson(gdf, path):
    """
    Given a geodataframe, export geojson representation to specified path.
    :param gdf: Input geodataframe
    :param path: Output path
    """
    with open(path, "w") as file:
        file.write(gdf.to_json())

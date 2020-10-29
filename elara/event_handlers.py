from math import floor
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Union, Tuple, Optional
import logging
import os
import networkx as nx
from shapely.geometry import LineString


from elara.factory import WorkStation, Tool

logger = logging.getLogger(__name__)


class EventHandlerTool(Tool):
    """
    Base Tool class for Event Handling.
    """
    result_dfs = dict()
    options_enabled = True

    def __init__(self, config, option=None):
        self.logger = logging.getLogger(__name__)
        super().__init__(config, option)

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

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
        if not isinstance(elems_in, list):
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
        cols = [h for h in range(self.config.time_periods)]
        return df.loc[df[cols].sum(axis=1) > 0]

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


class AgentGraph(EventHandlerTool):
    """
    Extract Waiting times for agents.
    """

    requirements = [
        'events',
        'transit_vehicles',
        'attributes',
    ]

    def __init__(self, config, option=None) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

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

        for person, subpopulation in self.resources['attributes'].map.items():
            self.graph.add_node(person, subpop=subpopulation)

    def process_event(self, elem) -> None:
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
        :param elem: Event XML element
        """
        event_type = elem.get("type")

        if event_type == "PersonEntersVehicle":
            agent_id = elem.get("person")

            if agent_id not in self.resources['attributes'].map:
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
        name = "graph.pkl".format(self.option)
        path = os.path.join(self.config.output_path, name)
        nx.write_gpickle(self.graph, path)


class AgentWaitingTimes(EventHandlerTool):
    """
    Extract interaction graph for agents.
    """

    requirements = [
        'events',
        'transit_schedule',
        'attributes',
    ]

    def __init__(self, config, option=None) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

        self.agent_status = dict()
        self.veh_waiting_occupancy = dict()

        self.waiting_time_log = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        csv_name = "{}_agent_waiting_times_{}.csv".format(self.config.name, self.option)

        self.waiting_time_log = self.start_chunk_writer(csv_name, write_path=write_path)

    def process_event(self, elem) -> None:
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
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

                # get subpop
                subpop = self.resources['attributes'].map.get(agent_id)

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
                        'subpop': subpop
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
        self.waiting_time_log.finish()


class VolumeCounts(EventHandlerTool):
    """
    Extract Volume Counts for mode on given network.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]

    def __init__(self, config, option=None) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(
            self.resources['attributes'].classes)
        self.logger.debug(f'sub_populations = {self.classes}')

        # generate index and map for network link dimension
        self.elem_gdf = self.resources['network'].link_gdf
        
        links = resources['network'].mode_to_links_map.get(self.option)
        if links is None:
            self.logger.warning(
                f"""
                No viable links found for mode:{self.option} in Network, 
                this may be because the Network modes do not match the configured 
                modes. Elara will continue with all links found in network.
                """
                )
        else:
            self.logger.debug(f'Selecting links for mode:{self.option}.')
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
            if veh_mode == self.option:
                # look for attribute_class, if not found assume pt and use mode
                attribute_class = self.resources['attributes'].map.get(ident, 'not_applicable')
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
        if self.option != "car":
            scale_factor = 1.0

        # Scale final counts
        self.counts *= 1.0 / scale_factor

        names = ['elem', 'class', 'hour']
        indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]
        counts_df = counts_df.unstack(level='hour').sort_index()
        counts_df = counts_df.reset_index().set_index('elem')
        # Create volume counts output
        key = f"volume_counts_{self.option}_classes"
        self.result_dfs[key] = self.elem_gdf.join(
            counts_df, how="left"
        )

        # calc sum across all recorded attribute classes
        total_counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()

        key = f"volume_counts_{self.option}"
        self.result_dfs[key] = self.elem_gdf.join(
            totals_df, how="left"
        )


class PassengerCounts(EventHandlerTool):
    """
    Build Passenger Counts on links for given mode in mode vehicles.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]
    invalid_options = ['car']

    def __init__(self, config, option=None):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None
        self.total_counts = None
        self.veh_occupancy = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Check for car
        if self.option == 'car':
            raise ValueError("Passenger Counts Handlers not intended for use with mode type = car")

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(resources['attributes'].classes)
        self.logger.debug(f'sub_populations = {self.classes}')

        # Initialise element attributes
        self.elem_gdf = resources['network'].link_gdf

        links = resources['network'].mode_to_links_map.get(self.option)
        if links is None:
            self.logger.warning(
                f"""
                No viable links found for mode:{self.option} in Network, 
                this may be because the Network modes do not match the configured 
                modes. Elara will continue with all links found in network.
                """
                )
        else:
            self.logger.debug(f'Selecting links for mode:{self.option}.')
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
            if agent_id[:2] != "pt" and veh_mode == self.option:
                attribute_class = self.resources['attributes'].map[agent_id]

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
            if agent_id[:2] != "pt" and veh_mode == self.option:
                attribute_class = self.resources['attributes'].map[agent_id]

                if not self.veh_occupancy[veh_id][attribute_class]:
                    pass
                else:
                    self.veh_occupancy[veh_id][attribute_class] -= 1
                    if not self.veh_occupancy[veh_id]:
                        self.veh_occupancy.pop(veh_id, None)

        elif event_type == "left link" or event_type == "vehicle leaves traffic":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.option:
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

        names = ['elem', 'class', 'hour']
        indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]
        counts_df = counts_df.unstack(level='hour').sort_index()
        counts_df = counts_df.reset_index().set_index('elem')

        # Create volume counts output
        key = f"passenger_counts_{self.option}_classes"
        self.result_dfs[key] = self.elem_gdf.join(
            counts_df, how="left"
        )

        # calc sum across all recorded attribute classes
        total_counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()

        key = f"passenger_counts_{self.option}"
        self.result_dfs[key] = self.elem_gdf.join(
            totals_df, how="left"
        )


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
    invalid_options = ['car']

    def __init__(self, config, option=None):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None
        self.total_counts = None
        self.route_occupancy = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(resources['attributes'].classes)
        self.logger.debug(f'sub_populations = {self.classes}')

        # # Initialise element attributes
        # all_routes = set(resources['transit_schedule'].veh_to_route_map.values())
        # get routes used by this mode
        self.logger.debug(f'Selecting routes for mode:{self.option}.')
        routes = resources['transit_schedule'].mode_to_routes_map.get(self.option)
        if routes is None:
            self.logger.warning(
                f"""
                No viable routes found for mode:{self.option} in TransitSchedule, 
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

        self.total_counts = None

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
            if agent_id[:2] != "pt" and veh_mode == self.option:
                attribute_class = self.resources['attributes'].map[agent_id]

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
            if agent_id[:2] != "pt" and veh_mode == self.option:
                attribute_class = self.resources['attributes'].map[agent_id]

                if self.route_occupancy[veh_route][attribute_class]:
                    self.route_occupancy[veh_route][attribute_class] -= 1
                    if not self.route_occupancy[veh_route]:
                        self.route_occupancy.pop(veh_route, None)
        elif event_type == "left link" or event_type == "vehicle leaves traffic":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)
            veh_route = self.vehicle_route(veh_id)

            if veh_mode == self.option:
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

        # Scale final counts
        self.counts *= 1.0 / self.config.scale_factor

        names = ['elem', 'class', 'hour']
        indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]
        counts_df = counts_df.unstack(level='hour').sort_index()
        counts_df = counts_df.reset_index().set_index('elem')

        # Create volume counts output
        key = "route_passenger_counts_{}_classes".format(self.option)
        self.result_dfs[key] = counts_df

        # calc sum across all recorded attribute classes
        total_counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()
        key = "route_passenger_counts_{}".format(self.option)
        self.result_dfs[key] = totals_df


class StopInteractions(EventHandlerTool):
    """
    Alightings and Boardings handler for given mode.
    """

    requirements = [
        'events',
        'transit_schedule',
        'attributes',
    ]
    invalid_options = ['car']

    def __init__(self, config, option=None):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.boardings = None
        self.alightings = None
        self.agent_status = None
        self.total_counts = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Check for car
        if self.option == 'car':
            raise ValueError("Stop Interaction Handlers not intended for use with mode type = car")

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(resources['attributes'].classes)
        self.logger.debug(f'sub_populations = {self.classes}')

        # Initialise element attributes
        self.elem_gdf = resources['transit_schedule'].stop_gdf
        self.logger.debug(f'Filtering stops for mode:{self.option}.')
        elem_gdf = self.elem_gdf.loc[self.elem_gdf.loc[:, "mode"] == self.option]
        if len(elem_gdf) == 0:
            self.logger.warning(
                f"""
                No viable stops found for mode:{self.option} in TransitSchedule, 
                this may be because the Schedule modes do not match the configured 
                modes. Continuing with all stops.
                """
                )
        else:
            self.elem_gdf = elem_gdf

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

            if veh_mode == self.option:
                agent_id = elem.get("person")

                if self.agent_status.get(agent_id, None) is not None:
                    time = float(elem.get("time"))
                    attribute_class = self.resources['attributes'].map[agent_id]
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

            if veh_mode == self.option:
                agent_id = elem.get("person")

                if self.agent_status.get(agent_id, None) is not None:
                    time = float(elem.get("time"))
                    attribute_class = self.resources['attributes'].map[agent_id]
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

        # Scale final counts
        self.boardings *= 1.0 / self.config.scale_factor
        self.alightings *= 1.0 / self.config.scale_factor

        # Create passenger counts output
        for data, name in zip([self.boardings, self.alightings], ['stop_boardings', 'stop_alightings']):

            names = ['elem', 'class', 'hour']
            indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
            index = pd.MultiIndex.from_product(indexes, names=names)
            counts_df = pd.DataFrame(data.flatten(), index=index)[0]
            counts_df = counts_df.unstack(level='hour').sort_index()
            counts_df = counts_df.reset_index().set_index('elem')

            # Create volume counts output
            key = "{}_{}_classes".format(name, self.option)
            self.result_dfs[key] = self.elem_gdf.join(
                counts_df, how="left"
            )

            # calc sum across all recorded attribute classes
            total_counts = data.sum(1)

            totals_df = pd.DataFrame(
                data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
            ).sort_index()

            key = "{}_{}".format(name, self.option)
            self.result_dfs[key] = self.elem_gdf.join(
                totals_df, how="left"
            )


class VehicleInteractions(EventHandlerTool):
    """
    Vehicle Alightings and Boardings handler for given mode.

    Using events:

    <event time="300.0"
        type="PersonEntersVehicle"
        person="pt_veh_41173_bus_Bus"
        vehicle="veh_41173_bus"/>
    <event time="600.0"
        type="PersonLeavesVehicle"
        person="pt_veh_41173_bus_Bus"
        vehicle="veh_41173_bus"/>
    """

    requirements = [
        'events',
        'transit_schedule',
        'attributes',
    ]
    invalid_options = ['car']

    def __init__(self, config, option=None):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)
        self.classes = None
        self.class_indices = None
        self.elem_df = None
        self.elem_ids = None
        self.elem_indices = None
        self.boardings = None
        self.alightings = None
        self.total_counts = None

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
        if self.option == 'car':
            raise ValueError("Stop Interaction Handlers not intended for use with mode type = car")

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(resources['attributes'].classes)
        self.logger.debug(f'sub_populations = {self.classes}')

        # Initialise element attributes
        self.elem_df = resources['transit_schedule'].vehicles_df
        self.logger.debug(f'Filtering vehicles for mode:{self.option}.')
        elem_df = self.elem_df.loc[self.elem_df.loc[:, "mode"] == self.option]
        if len(elem_df) == 0:
            self.logger.warning(
                f"""
                No viable stops found for mode:{self.option} in TransitSchedule, 
                this may be because the Schedule modes do not match the configured 
                modes. Continuing with all stops.
                """
                )
        else:
            self.elem_df = elem_df

        self.elem_ids, self.elem_indices = self.generate_elem_ids(self.elem_df)

        # Initialise results tables
        self.boardings = np.zeros((len(self.elem_indices),
                                   len(self.class_indices),
                                   self.config.time_periods))
        self.alightings = np.zeros((len(self.elem_indices),
                                    len(self.class_indices),
                                    self.config.time_periods))

    def process_event(self, elem):
        """
        Iteratively aggregate 'waitingForPt', 'PersonEntersVehicle' and
        'PersonLeavesVehicle' events to determine vehicle boardings and alightings.
        :param elem: Event XML element
        """
        event_type = elem.get("type")

        if event_type == "PersonEntersVehicle":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.option:
                agent_id = elem.get("person")

                if agent_id[:2] != "pt": # check for pt driver
                    time = float(elem.get("time"))
                    attribute_class = self.resources['attributes'].map[agent_id]
                    x, y, z = table_position(
                        self.elem_indices,
                        self.class_indices,
                        self.config.time_periods,
                        veh_id,
                        attribute_class,
                        time
                    )
                    self.boardings[x, y, z] += 1

        elif event_type == "PersonLeavesVehicle":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.option:
                agent_id = elem.get("person")

                if agent_id[:2] != "pt": # check for pt driver
                    time = float(elem.get("time"))
                    attribute_class = self.resources['attributes'].map[agent_id]
                    x, y, z = table_position(
                        self.elem_indices,
                        self.class_indices,
                        self.config.time_periods,
                        veh_id,
                        attribute_class,
                        time
                    )
                    self.alightings[x, y, z] += 1

    def finalise(self):
        """
        Following event processing, the raw events table will contain boardings
        and alightings by link by time slice. The only thing left to do is scale
        by the sample size and create dataframes.
        """

        # Scale final counts
        self.boardings *= 1.0 / self.config.scale_factor
        self.alightings *= 1.0 / self.config.scale_factor

        # Create counts output
        for data, name in zip([self.boardings, self.alightings], ['veh_boardings', 'veh_alightings']):

            names = ['veh_id', 'class', 'hour']
            indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
            index = pd.MultiIndex.from_product(indexes, names=names)
            counts_df = pd.DataFrame(data.flatten(), index=index)[0]
            counts_df = counts_df.unstack(level='hour').sort_index()
            counts_df = counts_df.reset_index().set_index('veh_id')

            # Create output
            key = f"{name}_{self.option}_classes"
            counts_df = self.elem_df.join(
                counts_df, how="left"
            )
            self.result_dfs[key] = counts_df

            # calc sum across classes
            total_counts = data.sum(1)
            totals_df = pd.DataFrame(
                data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
            ).sort_index()

            key = f"{name}_{self.option}"
            self.result_dfs[key] = self.elem_df.join(
                totals_df, how="left"
            )


class PassengerStopToStopCounts(EventHandlerTool):
    """
    Build Passenger Counts between stops for given mode in mode vehicles.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'attributes',
    ]
    invalid_options = ['car']

    def __init__(self, config, option=None):
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)
        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None
        self.total_counts = None
        self.veh_occupancy = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources, write_path=write_path)

        # Check for car
        if self.option in ['car','walk','bike']:
            raise ValueError(f"Passenger Counts Handlers not intended for use with mode type = {self.option}")

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(resources['attributes'].classes)
        self.logger.debug(f'sub_populations = {self.classes}')

        # Initialise element attributes
        self.elem_gdf = resources['transit_schedule'].stop_gdf
        self.logger.debug(f'Filtering stops for mode:{self.option}.')
        elem_gdf = self.elem_gdf.loc[self.elem_gdf.loc[:, "mode"] == self.option]
        if len(elem_gdf) == 0:
            self.logger.warning(
                f"""
                No viable stops found for mode:{self.option} in TransitSchedule, 
                this may be because the Schedule modes do not match the configured 
                modes. Continuing with all stops.
                """
                )
        else:
            self.elem_gdf = elem_gdf

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
            if agent_id[:2] != "pt" and veh_mode == self.option:
                attribute_class = self.resources['attributes'].map[agent_id]

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
            if agent_id[:2] != "pt" and veh_mode == self.option:
                attribute_class = self.resources['attributes'].map[agent_id]

                if self.veh_occupancy[veh_id][attribute_class]:
                    self.veh_occupancy[veh_id][attribute_class] -= 1
                    if not self.veh_occupancy[veh_id]:
                        self.veh_occupancy.pop(veh_id, None)

        elif event_type == "VehicleArrivesAtFacility":
            veh_id = elem.get("vehicle")
            veh_mode = self.vehicle_mode(veh_id)

            if veh_mode == self.option:
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

        # Scale final counts
        self.counts *= 1.0 / self.config.scale_factor

        names = ['origin', 'destination', 'class', 'hour']
        indexes = [self.elem_ids, self.elem_ids, self.classes, range(self.config.time_periods)]
        index = pd.MultiIndex.from_product(indexes, names=names)
        counts_df = pd.DataFrame(self.counts.flatten(), index=index)[0]
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

        counts_df = counts_df.reset_index().set_index(['origin', 'destination', 'class'])

        counts_df['geometry'] = [LineString([o, d]) for o,d in zip(counts_df.origin_geometry, counts_df.destination_geometry)]
        counts_df.drop('origin_geometry', axis=1, inplace=True)
        counts_df.drop('destination_geometry', axis=1, inplace=True)
        counts_df = gpd.GeoDataFrame(counts_df, geometry='geometry')

        # Create volume counts output
        key = f"stop_to_stop_passenger_counts_{self.option}_classes"
        self.result_dfs[key] = counts_df

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

        totals_df['geometry'] = [LineString([o, d]) for o,d in zip(totals_df.origin_geometry, totals_df.destination_geometry)]
        totals_df.drop('origin_geometry', axis=1, inplace=True)
        totals_df.drop('destination_geometry', axis=1, inplace=True)
        totals_df = gpd.GeoDataFrame(totals_df, geometry='geometry')

        totals_df = gpd.GeoDataFrame(totals_df, geometry='geometry')
        key = f"stop_to_stop_passenger_counts_{self.option}"
        self.result_dfs[key] = totals_df


class EventHandlerWorkStation(WorkStation):

    """
    Work Station for holding and building Event Handlers.
    """

    tools = {
        "volume_counts": VolumeCounts,
        "passenger_counts": PassengerCounts,
        "route_passenger_counts": RoutePassengerCounts,
        "stop_interactions": StopInteractions,
        "vehicle_interactions": VehicleInteractions,
        "waiting_times": AgentWaitingTimes,
        "graph": AgentGraph,
        "passenger_stop_to_stop_loading": PassengerStopToStopCounts
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
            self.logger.warning(f'{self.__str__} has no resources, build returning None.')
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
        self.logger.debug(f'{self.__str__()} .resources = {self.resources}')

        for handler_name, handler in self.resources.items():
            self.logger.info(f'Finalising {handler.__str__()}')
            handler.finalise()

            if self.config.contract:
                self.logger.info(f'Contracting {handler.__str__()}')
                handler.contract_results()

            self.logger.debug(f'{len(handler.result_dfs)} result_dfs at {handler.__str__()}')

            if handler.result_dfs:
                self.logger.info(f'Writing results for {handler.__str__()}')

                for name, df in handler.result_dfs.items():
                    csv_name = f"{name}.csv"
                    geojson_name = f"{name}.geojson"

                    self.write_csv(df, csv_name, write_path=write_path)
                    if isinstance(df, gpd.GeoDataFrame):
                        self.write_geojson(df, geojson_name, write_path=write_path)


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


def table_position_4d(origin_elem_indices, destination_elem_indices, class_indices, periods, o_id, d_id, attribute_class, time):
    """
    Calculate the result table coordinates from a given a origin element ID, destination elmement ID, attribute class and timestamp.
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

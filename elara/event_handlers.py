from math import floor
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from typing import Union, Tuple


from elara.factory import WorkStation, Tool


class EventHandlerTool(Tool):
    """
    Base Tool class for Event Handling.
    """
    result_gdfs = dict()

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
        if vehicle_id in self.resources['transit_vehicles'].veh_id_veh_type_map.keys():
            return self.resources['transit_vehicles'].veh_type_mode_map[
                self.resources['transit_vehicles'].veh_id_veh_type_map[vehicle_id]
            ]
        else:
            return "car"

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
        self.result_gdfs = {
            k: self.remove_empty_rows(df) for (k, df) in self.result_gdfs.items()
        }


class AgentWaitingTimes(EventHandlerTool):
    """
    Extract Waiting times for agents by mode

    waiting for pt:

    <event time="25752.0" type="waitingForPt" agent="gerry" atStop="home_stop_out"
    destinationStop="work_stop_in"/>

    <event time="27001.0" type="PersonEntersVehicle" person="gerry" vehicle="bus1"/>

    <event time="27300.0" type="VehicleDepartsAtFacility" vehicle="bus1"
    facility="home_stop_out" delay="0.0"/>

    interchange:

    <event time="46589.0" type="PersonLeavesVehicle" person="census_1001" vehicle="veh_26916_rail"  />

    <event time="46589.0" type="arrival" person="census_1001" link="pt_IRELANDWHOLE3575" legMode="pt"  />
	<event time="46589.0" type="actstart" person="census_1001" link="pt_IRELANDWHOLE3575" actType="pt interaction"  />
	<event time="46589.0" type="actend" person="census_1001" link="pt_IRELANDWHOLE3575" actType="pt interaction"  />
	<event time="46589.0" type="departure" person="census_1001" link="pt_IRELANDWHOLE3575" legMode="transit_walk"  />
	<event time="46589.0" type="VehicleDepartsAtFacility" vehicle="veh_8487_tram" facility="IRELANDWHOLE10719.link:pt_IRELANDWHOLE10719" delay="0.0"  />
	<event time="46589.0" type="travelled" person="census_1001" distance="0.0"  />
	<event time="46589.0" type="arrival" person="census_1001" link="491835" legMode="transit_walk"  />
	<event time="46589.0" type="actstart" person="census_1001" link="491835" actType="pt interaction"  />
	<event time="46589.0" type="actend" person="census_1001" link="491835" actType="pt interaction"  />
	<event time="46589.0" type="departure" person="census_1001" link="491835" legMode="pt"  />
	<event time="46589.0" type="waitingForPt" agent="census_1001" atStop="IRELANDWHOLE3575.link:491835" destinationStop="IRELANDWHOLE3633.link:857029"  />

	<event time="47601.0" type="PersonEntersVehicle" person="census_1001" vehicle="veh_27216_bus"  />


	EXIT:
	<event time="27534.0" type="PersonLeavesVehicle" person="gerry" vehicle="bus1"  />
	<event time="27534.0" type="arrival" person="gerry" link="3-4" legMode="pt"  />
	<event time="27534.0" type="actstart" person="gerry" link="3-4" actType="pt interaction"  />
	<event time="27534.0" type="actend" person="gerry" link="3-4" actType="pt interaction"  />
	<event time="27534.0" type="departure" person="gerry" link="3-4" legMode="transit_walk"  />
	<event time="28318.0" type="travelled" person="gerry" distance="653.2419153728579"  />
	<event time="28318.0" type="arrival" person="gerry" link="3-4" legMode="transit_walk"  />
	<event time="28318.0" type="actstart" person="gerry" link="3-4" actType="work"  />

	1. get number of pt legs
	2. minus number of unique agents
	3. = number of start interactions and interchanges
	4. stream events for pt interactions
	5. keep dict of agents status {time at interaction}
	6. stream events for agents entering pt vehicles
	7. refer to dict to add wait to results
	8. if agent starts a non pt interaction activity then clear dict

	9. record type of interchange? (ie strart/inter)
	10. record modes? (bus-bus/train-bus etc)
	11. collect time and loc
    """

    requirements = [
        'events',
        # 'agent_logs',
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

        self.waiting_log = []

        # Initialise results storage
        self.results = dict()  # Result dataframes ready to export

    def __str__(self):
        return f'AgentWaitingTimes'

    def build(self, resources: dict) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :return: None
        """
        super().build(resources)

        # name = "{}_agents_leg_logs_{}".format(self.config.name, self.option)
        # legs_df = self.resources['agent_logs'].results[name]
        #
        # unique_pt_interactions = legs_df.iloc[legs_df.act == 'pt interaction']
        # num_pt_interactions = len(unique_pt_interactions)
        # num_unique_agents = unique_pt_interactions.agent_id.nunique()

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

            for agent_id in self.veh_waiting_occupancy[veh_ident]:

                if agent_id not in self.agent_status:  # ignore (ie driver)
                    continue

                start_time, previous_mode = self.agent_status[agent_id]

                waiting_time = time - start_time

                # first waiting
                vehicle_waiting_report.append(
                    {
                        'agent_id': agent_id,
                        'prev_mode': previous_mode,
                        'mode': veh_mode,
                        'stop': stop,
                        'duration': waiting_time,
                        'departure': time
                    }
                )

                # update agent status for 'previous mode'
                self.agent_status[agent_id] = [None, veh_mode]

            # clear veh_waiting_occupancy
            self.veh_waiting_occupancy.pop('veh_ident', None)

            self.waiting_log.extend(vehicle_waiting_report)

            return None

        if event_type == "actstart":

            if elem.get("type") == "pt interaction":  # ignore pt interactions
                return None

            agent_id = elem.get("person")
            self.agent_status.pop(agent_id, None)  # agent has finished transit - remove record


class VolumeCounts(EventHandlerTool):
    """
    Extract Volume Counts for mode on given network.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
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

        self.classes = None
        self.class_indices = None
        self.elem_gdf = None
        self.elem_ids = None
        self.elem_indices = None
        self.counts = None

        # Initialise results storage
        self.result_gdfs = dict()  # Result geodataframes ready to export

    def __str__(self):
        return f'VolumeCounts'

    def build(self, resources: dict) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :return: None
        """
        super().build(resources)

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(
            self.resources['attributes'].classes)

        # generate index and map for network link dimension
        self.elem_gdf = self.resources['network'].link_gdf
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
        key = "volume_counts_{}".format(self.option)
        self.result_gdfs[key] = self.elem_gdf.join(
            counts_df, how="left"
        )

        # calc sum across all recorded attribute classes
        total_counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()

        key = "volume_counts_{}_total".format(self.option)
        self.result_gdfs[key] = self.elem_gdf.join(
            totals_df, how="left"
        )


class PassengerCounts(EventHandlerTool):
    """
    Build Passenger Counts for given mode in mode vehicles.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'transit_vehicles',
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

        # Initialise results storage
        self.result_gdfs = dict()  # Result geodataframes ready to export

    def __str__(self):
        return f'PassengerCounts'

    def build(self, resources: dict) -> None:
        """
        Build Handler.
        :param resources: dict, supplier resources
        :return: None
        """
        super().build(resources)

        # Check for car
        if self.option == 'car':
            raise ValueError("Passenger Counts Handlers not intended for use with mode type = car")

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(resources['attributes'].classes)

        # Initialise element attributes
        self.elem_gdf = resources['network'].link_gdf
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
        key = "passenger_counts_{}".format(self.option)
        self.result_gdfs[key] = self.elem_gdf.join(
            counts_df, how="left"
        )

        # calc sum across all recorded attribute classes
        total_counts = self.counts.sum(1)

        totals_df = pd.DataFrame(
            data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
        ).sort_index()

        key = "passenger_counts_{}_total".format(self.option)
        self.result_gdfs[key] = self.elem_gdf.join(
            totals_df, how="left"
        )


class StopInteractions(EventHandlerTool):
    """
    Alightings and Boardings handler for given mode.
    """

    requirements = [
        'events',
        'network',
        'transit_schedule',
        'transit_vehicles',
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

        # Initialise results storage
        self.result_gdfs = dict()  # Result geodataframes ready to export

    def __str__(self):
        return f'StopInteractions'

    def build(self, resources: dict) -> None:
        """
        Build handler.
        :param resources: dict, supplier resources.
        :return: None
        """
        super().build(resources)

        # Check for car
        if self.option == 'car':
            raise ValueError("Stop Interaction Handlers not intended for use with mode type = car")

        # Initialise class attributes
        self.classes, self.class_indices = self.generate_elem_ids(resources['attributes'].classes)

        # Initialise element attributes
        self.elem_gdf = resources['transit_schedule'].stop_gdf
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
        for data, name in zip([self.boardings, self.alightings], ['boardings', 'alightings']):

            names = ['elem', 'class', 'hour']
            indexes = [self.elem_ids, self.classes, range(self.config.time_periods)]
            index = pd.MultiIndex.from_product(indexes, names=names)
            counts_df = pd.DataFrame(data.flatten(), index=index)[0]
            counts_df = counts_df.unstack(level='hour').sort_index()
            counts_df = counts_df.reset_index().set_index('elem')

            # Create volume counts output
            key = "{}_{}".format(name, self.option)
            self.result_gdfs[key] = self.elem_gdf.join(
                counts_df, how="left"
            )

            # calc sum across all recorded attribute classes
            total_counts = data.sum(1)

            totals_df = pd.DataFrame(
                data=total_counts, index=self.elem_ids, columns=range(0, self.config.time_periods)
            ).sort_index()

            key = "{}_{}_total".format(name, self.option)
            self.result_gdfs[key] = self.elem_gdf.join(
                totals_df, how="left"
            )


class EventHandlerWorkStation(WorkStation):

    """
    Work Station for holding and building Event Handlers.
    """

    tools = {
        "volume_counts": VolumeCounts,
        "passenger_counts": PassengerCounts,
        "stop_interactions": StopInteractions,
    }

    def __str__(self):
        return f'Events Handler WorkStation'

    def build(self, spinner=None) -> None:
        """
        Build all required handlers, then finalise and save results.
        :return: None
        """
        # build tools
        super().build(spinner)

        # iterate through events
        events = self.supplier_resources['events']
        for i, event in enumerate(events.elems):
            for event_handler in self.resources.values():
                event_handler.process_event(event)
            if not i % 10000 and spinner:
                spinner.text = f'{self} processed {i} events.'

        # finalise
        # Generate event file outputs
        for handler_name, event_handler in self.resources.items():
            if spinner:
                spinner.text = f'{self} finalising {handler_name}.'
            event_handler.finalise()
            if self.config.contract:
                event_handler.contract_results()

            for name, gdf in event_handler.result_gdfs.items():
                if spinner:
                    spinner.text = f'{self} writing {name} results to disk.'
                csv_name = "{}_{}.csv".format(self.config.name, name)
                geojson_name = "{}_{}.geojson".format(self.config.name, name)
                csv_path = os.path.join(self.config.output_path, csv_name)
                geojson_path = os.path.join(self.config.output_path, geojson_name)

                # File exports
                gdf.drop("geometry", axis=1).to_csv(csv_path)
                export_geojson(gdf, geojson_path)


def table_position(elem_indices, class_indices, periods, elem_id, attribute_class, time):
    """
    Calculate the result table coordinates from a given a element ID and timestamp.
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


def export_geojson(gdf, path):
    """
    Given a geodataframe, export geojson representation to specified path.
    :param gdf: Input geodataframe
    :param path: Output path
    """
    with open(path, "w") as file:
        file.write(gdf.to_json())

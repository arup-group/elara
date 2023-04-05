import geopandas as gdp
from lxml import etree
import pandas as pd
import pyproj
from shapely.geometry import Point, LineString
import gzip
from io import BytesIO
from math import floor
from typing import Optional
import logging
from collections import defaultdict

from elara.factory import WorkStation, Tool
from elara.helpers import decode_polyline_to_shapely_linestring

WGS_84 = pyproj.Proj("epsg:4326")


class InputTool(Tool):

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)
        self.logger = logging.getLogger(__name__)

    def set_and_change_crs(self, target: gdp.GeoDataFrame, set_crs=None, to_crs='epsg:4326'):
        """
        Set and change a target GeoDataFrame crs. Wrapper for geopandas .crs and .to_crs.
        If set_crs or to_crs are None then warning is logged and no action taken.
        :param target: geopandas GeoDataFrame
        :param set_crs: optional string crs
        :param to_crs: optional string crs
        :return: None
        """
        if not isinstance(target, gdp.GeoDataFrame):
            raise TypeError(f'Expected geopandas.GeoDataFrame not {type(target)}')

        if set_crs is None:
            self.logger.warning(f'No set_crs, re-projection disabled at {self.__str__()}')
            return None

        else:
            self.logger.debug(f'Setting target projection to {set_crs}')
            target.crs = set_crs

        if to_crs is None:
            self.logger.warning(f'No to_crs, re-projection disabled at {self.__str__()}')
            return None

        else:
            self.logger.debug(f'Re-projecting target geo dataframe of size {len(target.index)} to {to_crs}')

            target.to_crs(to_crs, inplace=True)


class Events(InputTool):
    requirements = ['events_path']
    elems = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Events object constructor.
        :param resources: GetPath resources from suppliers
        :param write_path: Optional output path overwrite
        """
        super().build(resources)

        path = resources['events_path'].path

        self.elems = get_elems(path, "event")



class Network(InputTool):
    requirements = ['network_path', 'crs']
    node_gdf = None
    link_gdf = None

    def build(
            self,
            resources: dict,
            write_path: Optional[str] = None
    ) -> None:
        """
        Network object constructor.
        :param resources: dict, resources from suppliers
        :param write_path: Optional output path overwrite
        """
        super().build(resources)

        path = resources['network_path'].path
        crs = resources['crs'].path

        # Extract element properties
        self.logger.debug(f'Loading nodes')
        nodes = [
            self.get_node_elem(elem) for elem in get_elems(path, "node")
        ]
        node_lookup = {node["id"]: node for node in nodes}

        self.logger.debug(f'Loading links')
        links = [
            self.get_link_elem(elem, node_lookup)
            for elem in get_elems(path, "link")
        ]

        self.logger.debug(f'Building network nodes geodataframe')
        node_df = pd.DataFrame(nodes)
        node_df.set_index("id", inplace=True)
        node_df.sort_index(inplace=True)
        self.node_gdf = gdp.GeoDataFrame(node_df, geometry="geometry").sort_index()

        self.logger.debug(f'Building network links geodataframe')
        link_df = pd.DataFrame(links)
        link_df.set_index("id", inplace=True)
        link_df.sort_index(inplace=True)
        self.link_gdf = gdp.GeoDataFrame(link_df, geometry="geometry").sort_index()

        self.logger.debug(f'Building mode to network links mapping')
        self.mode_to_links_map = defaultdict(set)
        for link_id, modes in zip(self.link_gdf.index, [ms.split(",") for ms in self.link_gdf.modes]):
            for mode in modes:
                self.mode_to_links_map[mode].add(link_id)

        # transform
        self.logger.debug(f'Re-projecting network nodes geodataframe')
        self.set_and_change_crs(self.node_gdf, set_crs=crs)
        self.logger.debug(f'Re-projecting network links geodataframe')
        self.set_and_change_crs(self.link_gdf, set_crs=crs)

    @staticmethod
    def transform_node_elem(elem, crs):
        """
        Convert raw node XML element into dictionary.
        :param elem: Node XML element
        :param crs: Original coordinate reference system code
        :return: Dictionary
        """
        x = float(elem.get("x"))
        y = float(elem.get("y"))

        geometry = generate_point(x, y, crs)

        return {"id": str(elem.get("id")), "geometry": geometry}

    @staticmethod
    def get_node_elem(elem):
        """
        Convert raw node XML element into dictionary.
        :param elem: Node XML element
        :return: Dictionary
        """
        x = float(elem.get("x"))
        y = float(elem.get("y"))

        geometry = Point(x, y)

        return {"id": str(elem.get("id")), "geometry": geometry}

    @staticmethod
    def get_link_elem(elem, node_lookup):
        """
        Convert raw link XML element into dictionary.
        :param elem: Link XML element
        :param node_lookup: node lookup dict
        :return: Equivalent dictionary with relevant fields
        """
        from_id = str(elem.get("from"))
        to_id = str(elem.get("to"))

        geometry = elem.find('.//attribute[@name="geometry"]')
        if geometry is not None:
            geometry = decode_polyline_to_shapely_linestring(geometry.text)
        else:
            from_point = node_lookup[from_id]["geometry"]
            to_point = node_lookup[to_id]["geometry"]
            geometry = LineString([from_point, to_point])

        return {
            "id": str(elem.get("id")),
            "from": from_id,
            "to": to_id,
            "length": float(elem.get("length")),
            "lanes": float(elem.get("permlanes")),
            "capacity": float(elem.get("capacity")),
            "modes": elem.get("modes"),
            "geometry": geometry,
        }


class OSMWays(InputTool):
    """
    Light weight network input with no geometries - just osm way attribute and length.
    """

    requirements = ['network_path']
    ways = None
    lengths = None
    classes = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        OSM highway attribute map.
        :param resources: dict, resources from suppliers
        :param write_path: Optional output path overwrite
        """
        super().build(resources)

        path = resources['network_path'].path

        # Extract element properties
        self.logger.debug(f'Extracting links for OSM Ways input')
        links = [
            self.get_link_attribute(elem)
            for elem in get_elems(path, "link")
        ]
        self.ways = {link['id']: link['way'] for link in links}
        self.lengths = {link['id']: link['length'] for link in links}
        self.classes = list(set(self.ways.values()))
        self.logger.debug(f'OSM classes = {self.classes}')

    @staticmethod
    def get_link_attribute(elem):
        """
        Convert raw link XML element into tuple of ident and named attribute.
        :param elem: Link XML element
        :return: tuple of ident and attribute
        """

        # for name in ['osm:way:highway', 'osm:way:railway', 'osm:way:network']:
        for name in ['osm:way:highway', 'osm:way:railway', 'osm:way:network']:
            attribute = elem.find('.//attribute[@name="{}"]'.format(name))
            if attribute is not None:
                attribute = attribute.text
                break

        return {
            'id': str(elem.get("id")),
            "length": float(elem.get("length")),
            "way": str(attribute)
        }


class TransitSchedule(InputTool):
    requirements = ['transit_schedule_path', 'crs']
    stop_gdf = None
    mode_map = None
    modes = None

    def build(
            self,
            resources: dict,
            write_path: Optional[str] = None
    ) -> None:
        """
        Transit schedule object constructor.
        :param resources: dict, resources from suppliers
        :param write_path: Optional output path overwrite
        """
        super().build(resources)

        path = resources['transit_schedule_path'].path
        crs = resources['crs'].path

        # Retrieve stop attributes
        self.logger.debug(f'Loading transit stops')
        stops = [
            self.get_node_elem(elem)
            for elem in get_elems(path, "stopFacility")
        ]

        # Generate empty geodataframes
        stop_df = pd.DataFrame(stops)
        stop_df.set_index("id", inplace=True)
        stop_df.sort_index(inplace=True)

        self.stop_gdf = gdp.GeoDataFrame(stop_df, geometry="geometry").sort_index()

        # transform
        self.logger.debug(f'Reprojecting stops')
        self.set_and_change_crs(self.stop_gdf, set_crs=crs)

        # TODO the following is nested to try and recover some efficiency.
        # However this "Schedule" class now has a lot of variables/mappings/etc.
        # It would probably be sensible to refactor this class into two or more
        # distinct classes such as "Schedules Routes" and "Schedules Stops" or similar.

        # Generate route -> mode, mode -> stops and vehicles -> routes maps
        self.logger.debug(f'Generating mappings.')

        self.route_to_mode_map = {}
        self.mode_to_stops_map = defaultdict(set)
        self.veh_to_route_map = {}
        self.mode_to_veh_map = defaultdict(set)

        for elem in get_elems(path, "transitRoute"):
            # route -> mode map
            route_id, mode = self.get_route_mode(elem)
            self.route_to_mode_map[route_id] = mode

            # mode -> stops map
            stops = self.get_route_stops(elem)
            self.mode_to_stops_map[mode].update(stops)

            route_id, route_vehicles = self.get_route_vehicles(elem)
            # mode -> vehs map
            self.mode_to_veh_map[mode].update(route_vehicles)

            for vehicle_id in route_vehicles:
                # vehicles -> routes map
                self.veh_to_route_map[vehicle_id] = route_id

        # mode -> veh map
        self.veh_to_mode_map = {v: mode for mode, vs in self.mode_to_veh_map.items() for v in vs}

        # mode -> routes map
        self.mode_to_routes_map = defaultdict(set)
        for route, mode in self.route_to_mode_map.items():
            self.mode_to_routes_map[mode].add(route)

        self.modes = list(set(self.route_to_mode_map.values()))

        self.logger.debug(f'Transit Schedule Route modes = {self.modes}')
        self.logger.debug(f'Transit Schedule Stop modes = {list(self.mode_to_stops_map)}')

    @staticmethod
    def get_node_elem(elem):
        """
        Convert raw node XML element into dictionary.
        :param elem: Node XML element
        :return: Dictionary
        """
        x = float(elem.get("x"))
        y = float(elem.get("y"))

        geometry = Point(x, y)

        return {
            "id": str(elem.get("id")),
            "link": str(elem.get("linkRefId")),
            "name": str(elem.get("name")),
            "geometry": geometry,
        }

    @staticmethod
    def transform_stop_elem(elem, crs):
        """
        Convert raw stop facility XML element into dictionary.
        :param elem: Stop facility XML element
        :param crs: Original coordinate reference system code
        :return: Dictionary
        """
        x = float(elem.get("x"))
        y = float(elem.get("y"))

        geometry = generate_point(x, y, crs)

        return {
            "id": str(elem.get("id")),
            "link": str(elem.get("linkRefId")),
            "name": str(elem.get("name")),
            "geometry": geometry,
        }

    @staticmethod
    def get_route_mode(elem):
        """
        Get mode for each transit route
        :param elem: TransitRoute XML element
        :return: (route id, mode) tuple
        """

        route_id = elem.get('id')
        mode = elem.xpath('transportMode')[0].text

        return route_id, mode

    @staticmethod
    def get_route_stops(elem):
        """
        Get stop id set for each transit route.
        :param elem: TransitRoute XML element
        :return: stop_ids set
        """
        return set(elem.xpath("routeProfile/stop/@refId"))

    @staticmethod
    def get_route_vehicles(elem):
        """
        Get all vehicle IDs for a given transit route
        :param elem: TransitRoute XML element
        :return: route id, a set of vehicle IDs, empty if no departures were found on the route
        """
        route_vehicles = []

        route_id = elem.get('id')
        for departure in elem.xpath('departures')[0]:
            route_vehicles.append(departure.get('vehicleRefId'))

        return route_id, set(route_vehicles)


class TransitVehicles(InputTool):
    requirements = ['transit_vehicles_path']
    veh_type_mode_map = None
    veh_type_capacity_map = None
    types = None
    veh_id_veh_type_map = None
    transit_vehicle_counts = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Transit vehicles object constructor.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """
        super().build(resources)

        path = resources['transit_vehicles_path'].path

        # Vehicle types to mode correspondence
        self.veh_type_mode_map = {
            "Rail": "train",
            "Suburban Railway": "suburban rail",
            "Underground Service": "subway",
            "Metro Service": "subway",
            "Bus": "bus",
            "Coach Service": "coach",
            "Tram": "tram",
            "Ferry": "ferry",
            "Gondola": "gondola"
        }
        self.logger.debug(f'veh type mode map = {self.veh_type_mode_map}')

        # Vehicle type to total capacity correspondence
        self.veh_type_capacity_map = dict(
            [
                self.transform_veh_type_elem(elem)
                for elem in get_elems(path, "vehicleType")
            ]
        )
        self.logger.debug(f'veh type capacity map = {self.veh_type_capacity_map}')

        # Vehicle ID to vehicle type correspondence
        self.logger.debug('Building veh id to veh type map')
        self.veh_id_veh_type_map = {
            elem.get("id"): elem.get("type") for elem in get_elems(path, "vehicle")
        }

        self.types, self.transit_vehicle_counts = count_values(self.veh_id_veh_type_map)
        self.logger.debug(f'veh types = {self.types}')

    @staticmethod
    def transform_veh_type_elem(elem):
        """
        Extract the vehicle type and total capacity from a vehicleType XML element.
        :param elem: vehicleType XML element
        :return: (vehicle type, capacity) tuple
        """
        strip_namespace(elem)  # TODO only needed for this input = janky
        ident = elem.xpath("@id")[0]
        if len(elem.xpath("capacity/seats/@persons")) > 0:
            # https://www.matsim.org/files/dtd/vehicleDefinitions_v1.0.xsd
            seated_capacity = float(elem.xpath("capacity/seats/@persons")[0])
            standing_capacity = float(elem.xpath("capacity/standingRoom/@persons")[0])
        else:
            # https://www.matsim.org/files/dtd/vehicleDefinitions_v2.0.xsd
            seated_capacity = float(elem.xpath("capacity/@seats")[0])
            standing_capacity = float(elem.xpath("capacity/@standingRoomInPersons")[0])
        return ident, seated_capacity + standing_capacity


class Subpopulations(InputTool):
    requirements = ['attributes_path']
    # final_attribute_map = None
    map = None
    classes = None
    attribute_count_map = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Population subpopulation attribute constructor.
        :param resources: dict, of supplier resources.
        :param write_path: Optional output path overwrite.
        """
        super().build(resources)

        path = resources['attributes_path'].path

        if self.config.version == 12:
            self.logger.debug("Loading attribute map from V12 plan")
            self.map = dict(
                [
                    self.get_person_attribute_from_plans(elem, 'subpopulation')
                    for elem in get_elems(path, "person")
                ]
            )

        else:
            self.logger.debug("Loading attribute map from V11 personAttributes")
            self.map = dict(
                [
                    self.get_attribute_text(elem, 'subpopulation')
                    for elem in get_elems(path, "object")
                ]
            )

        self.classes, self.attribute_count_map = count_values(self.map)
        self.classes.append('not_applicable')
        # self.idents = sorted(list(self.map))
        # self.attribute_fields = set([k for v in self.map.values() for k in v.keys()])
        # self.attributes_df = pd.DataFrame.from_dict(self.map, orient='index')

    def get_attribute_text(self, elem, tag):
        ident = elem.xpath("@id")[0]
        attribute = elem.find('./attribute[@name="{}"]'.format(tag)).text
        # attribute = self.final_attribute_map.get(attribute.text, attribute.text)
        return ident, attribute

    def get_person_attribute_from_plans(self, elem, tag):
        ident = elem.xpath("@id")[0]
        attribute = elem.find(f'./attributes/attribute[@name="{tag}"]')
        if attribute is not None:
            return ident, attribute.text
        # attribute = self.final_attribute_map.get(attribute.text, attribute.text)
        return ident, None


class Attributes(InputTool):
    requirements = ['attributes_path']
    attributes = {}
    attribute_count_map = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Population subpopulation attribute constructor.
        :param resources: dict, of supplier resources.
        :param write_path: Optional output path overwrite.
        """
        super().build(resources)

        path = resources['attributes_path'].path

        if self.config.version == 12:
            self.logger.debug("Loading attribute map from V12 plan")
            self.attributes = dict(
                [
                    self.get_attributes_from_plans(elem)
                    for elem in get_elems(path, "person")
                ]
            )

        else:
            self.logger.debug("Loading attribute map from V11 personAttributes")
            self.attributes = dict(
                [
                    self.get_attributes(elem)
                    for elem in get_elems(path, "object")
                ]
            )

        self.idents = sorted(list(self.attributes))
        self.attribute_names = set([k for v in self.values() for k in v.keys()])
        # self.attributes_df = pd.DataFrame.from_dict(self.attributes, orient='index')  # todo: is this needed? - make it lazy

    def get(self, key, default):
        return self.attributes.get(key, default)

    def __contains__(self, other):
        return other in self.attributes

    def __getattr__(self, key):
        return self.attributes[key]

    def items(self):
        for k, v in self.attributes.items():
            yield k, v

    def keys(self):
        for key in self.attributes.keys():
            yield key

    def values(self):
        for value in self.attributes.values():
            yield value

    def get_attributes(self, elem):
        ident = elem.xpath("@id")[0]
        attributes = {}
        for attr in elem.xpath('./attribute'):
            attributes[attr.get('name')] = attr.text
        return ident, attributes

    def get_attributes_from_plans(self, elem):
        ident = elem.xpath("@id")[0]
        attributes = {}
        for attr in elem.xpath('./attributes/attribute'):
            if attr is not None:
                attributes[attr.get('name')] = attr.text
        return ident, attributes

    def attribute_values(self, attribute_key):
        """
        Get set of all available attribute values for a given attribute key.
        """
        return set([attribute_map.get(attribute_key) for attribute_map in self.values()])

    def attribute_key_availability(self, attribute_key):
        """
        Get proportion of agents with given attribute key available.
        """
        availability = [attribute_key in attribute_map for attribute_map in self.values()]
        return sum(availability) / len(availability)


class Plans(InputTool):
    requirements = ['plans_path']
    plans = None
    persons = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Plans object constructor.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """
        super().build(resources)

        # call by list index so 'input_plans' can be used by InputPlans
        path = resources[self.requirements[0]].path

        self.plans = get_elems(path, "plan")
        # self.persons = get_elems(path, "person")
        self.persons = self.filter_out_persons_with_empty_plans(get_elems(path, "person"))  # skip empty-plan persons

    def filter_out_persons_with_empty_plans(self, iterator):
        """
        Skip persons with empty plans. When reading from the `plans_experienced.xml` file,
            these may be persons with no simulated legs.
        """
        for elem in iterator:
            if len(elem.find('./plan').getchildren()) > 0:
                yield elem


class InputPlans(Plans):
    """
    InputTool for iterating through plans used as inputs to a MATSim simulation.
    Used to calculate agent choice differences in a given simulation.
    """
    requirements = ['input_plans_path']
    plans = None
    persons = None


class OutputConfig(InputTool):
    requirements = ['output_config_path']

    modes = set()
    activities = set()
    sub_populations = set()

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Config object constructor.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        """
        super().build(resources)

        path = str(resources['output_config_path'].path)
        elems = etree.parse(path)

        for e in elems.xpath(
                '//module/parameterset/parameterset/param[@name="mode"]'
        ):
            self.modes.add(e.get('value'))

        for e in elems.xpath(
                '//module/parameterset/parameterset/param[@name="activityType"]'
        ):
            self.activities.add(e.get('value'))

        for e in elems.xpath(
                '//module/parameterset/param[@name="subpopulation"]'
        ):
            self.sub_populations.add(e.get('value'))

        self.modes = list(self.modes | set(["transit_walk", "pt"]))
        self.activities = list(self.activities)
        self.sub_populations = list(self.sub_populations)


class ModeHierarchy(InputTool):
    requirements = []

    hierarchy = [
        "ferry",
        "rail",
        "light rail",
        "suburban rail",
        "subway",
        "metro",
        "tram",
        "streetcar",
        "bus",
        "gondola",
        "car",
        "bike",
        "walk",
        "transit_walk",
        "access_walk",
        "egress_walk"
    ]

    def get(self, modes: list) -> str:
        if not isinstance(modes, list):
            raise TypeError(
                f"ModeHierarchy get method expects list, received: {modes}, type: {type(modes)}"
            )
        if not all(isinstance(mode, str) for mode in modes):
            raise TypeError(
                f"ModeHierarchy get method expects list of strings, received: {modes}, "
                f"type: {type(modes[0])}")
        for mode in modes:
            if str(mode).strip() not in self.hierarchy:
                self.logger.warning(f" {mode} not found in hierarchy, returning {mode} as main mode")
                return mode
        for h in self.hierarchy:
            for mode in modes:
                if h == mode:
                    return mode
        # # if mode not found in hierarchy then return middle mode
        # mode_index = floor(len(modes) / 2)
        # mode = modes[mode_index]
        # self.logger.warning(f"WARNING {modes} not in hierarchy, returning {mode}")
        # return mode


class ModeMap(InputTool):
    requirements = []

    modemap = {
        "ferry": "ferry",
        "rail": "rail",
        "tram": "tram",
        "bus": "bus",
        "car": "car",
        "bike": "bike",
        "walk": "walk",
        "transit_walk": "walk",
        "access_walk": "walk",
        "egress_walk": "walk"
    }

    def __getitem__(self, key: str) -> str:
        if key in self.modemap:
            return self.modemap[key]

        raise KeyError(f"key:'{key}' not found in ModeMap")


class RoadPricing(InputTool):
    requirements = ['road_pricing_path']
    links = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Network road pricing input file.
        :param resources: dict, of supplier resources.
        :param write_path: Optional output path overwrite.
        """
        super().build(resources)

        path = resources['road_pricing_path'].path

        self.logger.debug(f"Loading road pricing from {path}")
        self.links = dict(
            [
                self.get_costs(elem)
                for elem in get_elems(path, "link")
            ]
        )
        self.tollnames = dict(
            [
                self.get_tollnames(elem)
                for elem in get_elems(path, "link")
            ]
        )

    def get_costs(self, elem):
        ident = elem.xpath("@id")[0]
        costs = [dict(cost.items()) for cost in elem.xpath('./cost')]
        costs = sorted(costs, key=lambda k: k['start_time'])
        return ident, costs

    def get_tollnames(self, elem):
        ident = elem.xpath("@id")[0]
        if len(elem.xpath('./tollname/@name')):  # if the toll link has a tag called tollname
            tollname = elem.xpath('./tollname/@name')[0]
        else:
            tollname = 'missing'
        return ident, tollname


class Vehicles(InputTool):
    requirements = ['vehicles_path']
    vehicles = None
    vehicle_types = None
    veh_to_mode_map = None

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Vehicles output file. Defines agents' vehicle specs.
        :param resources: dict, of supplier resources.
        :param write_path: Optional output path overwrite.
        """
        super().build(resources)

        path = resources['vehicles_path'].path

        self.logger.debug(f"Loading vehicles from {path}")
        self.vehicles = dict(
            [
                self.get_vehicles(elem)
                for elem in get_elems(path, "vehicle")
            ]
        )
        self.vehicle_types = dict(
            [
                self.get_vehicle_types(elem)
                for elem in get_elems(path, "vehicleType")
            ]
        )
        self.veh_to_mode_map = {
            k: self.vehicle_types[v]['networkMode']['networkMode']
            for k, v in self.vehicles.items()
        }

    def get_vehicles(self, elem):
        return elem.xpath("@id")[0], str(elem.xpath("@type")[0])

    def get_vehicle_types(self, elem):
        ident = elem.xpath("@id")[0]
        vehicle_specs = {}
        for spec in elem.xpath('./*'):
            vehicle_specs[spec.tag.replace('{http://www.matsim.org/files/dtd}', '')] = dict(spec.attrib)
        return ident, vehicle_specs


class InputsWorkStation(WorkStation):
    tools = {
        "events": Events,
        'network': Network,
        'osm_ways': OSMWays,
        'transit_schedule': TransitSchedule,
        'transit_vehicles': TransitVehicles,
        'subpopulations': Subpopulations,
        'attributes': Attributes,
        'plans': Plans,
        'input_plans': InputPlans,
        'output_config': OutputConfig,
        'mode_map': ModeMap,
        'road_pricing': RoadPricing,
        'vehicles': Vehicles,
        # 'mode_hierarchy': ModeHierarchy,
    }

    def __init__(self, config):
        super().__init__(config=config)
        self.logger = logging.getLogger(__name__)


def get_elems(path, tag):
    """
    Wrapper for unzipping and dealing with xml namespaces
    :param path: xml path string
    :param tag: The tag type to extract , e.g. 'link'
    :return: Generator of elements
    """
    target = try_unzip(path)
    tag = get_tag(target, tag)
    target = try_unzip(path)  # need to repeat :(
    return parse_elems(target, tag)


def parse_elems(target, tag):
    """
    Traverse the given XML tree, retrieving the elements of the specified tag.
    :param target: Target xml, either BytesIO object or string path
    :param tag: The tag type to extract , e.g. 'link'
    :return: Generator of elements
    """
    doc = etree.iterparse(target, tag=tag)
    for _, element in doc:
        yield element
        element.clear()
        del element.getparent()[0]
    del doc


def try_unzip(path):
    """
    Attempts to unzip xml at given path, if fails, returns path
    :param path: xml path string
    :return: either BytesIO object or string path
    """
    try:
        with gzip.open(path) as unzipped:
            xml = unzipped.read()
            target = BytesIO(xml)
            return target
    except OSError:
        return path


def get_tag(target, tag):
    """
    Check for namespace declaration. If they exists return tag string
    with namespace [''] ie {namespaces['']}tag. If no namespaces declared
    return original tag
    TODO Not working with iterparse, generated elem also have ns which is dealt with later
    """
    nsmap = {}
    doc = etree.iterparse(target, events=('end', 'start-ns',))
    count = 0
    for event, element in doc:
        count += 1
        if event == 'start-ns':
            nsmap[element[0]] = element[1]
        if count == 10:  # assume namespace declared at top so can break early
            del doc
            break
    if not nsmap:
        return tag
    else:
        tag = '{' + nsmap[''] + '}' + tag
        return tag


def strip_namespace(elem):
    """
    Strips namespaces from given xml element
    :param elem: xml element
    :return: xml element
    """
    if elem.tag.startswith("{"):
        elem.tag = elem.tag.split('}', 1)[1]  # strip namespace
    for k in elem.attrib.keys():
        if k.startswith("{"):
            k2 = k.split('}', 1)[1]
            elem.attrib[k2] = elem.attrib[k]
            del elem.attrib[k]
    for child in elem:
        strip_namespace(child)


def generate_point(x, y, crs):
    """
    Given x, y coordinates and a CRS, return a Point geometry object reprojected
    into WGS 84.
    :param x: X coordinate
    :param y: Y coordinate
    :param crs: Initial coordinate reference system EPSG code
    :return: Point object
    """
    proj = pyproj.Proj(crs)
    lon, lat = pyproj.transform(proj, WGS_84, x, y)
    return Point(lon, lat)


def count_values(dictionary):
    """
    Build a dictionary of value counts in a given dictionary.
    :param dictionary: input dictionary object
    :return: dictionary object
    """
    counter = {}
    for value in dictionary.values():
        if counter.get(value):
            counter[value] += 1
        else:
            counter[value] = 1
    return list(counter.keys()), counter

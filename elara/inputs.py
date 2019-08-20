import geopandas as gdp
from lxml import etree
import pandas as pd
import pyproj
from shapely.geometry import Point, LineString
from shapely.ops import transform
import gzip
from io import BytesIO
from math import floor

from elara.factory import WorkStation, Tool

WGS_84 = pyproj.Proj(init="epsg:4326")


class Events(Tool):

    requirements = ['events_path']
    elems = None

    def build(self, resources: dict) -> None:
        """
        Events object constructor.
        :param resources: GetPath resources from suppliers
        """
        super().build(resources)

        path = resources['events_path'].path

        self.elems = get_elems(path, "event")


class Network(Tool):

    requirements = ['network_path', 'crs']
    node_gdf = None
    link_gdf = None

    def build(self, resources: dict) -> None:
        """
        Network object constructor.
        :param resources: dict, resources from suppliers
        """
        super().build(resources)

        path = resources['network_path'].path
        crs = resources['crs'].path

        # Extract element properties
        nodes = [
            self.get_node_elem(elem) for elem in get_elems(path, "node")
        ]
        node_lookup = {node["id"]: node for node in nodes}
        links = [
            self.get_link_elem(elem, node_lookup)
            for elem in get_elems(path, "link")
        ]
        # Generate empty geodataframes
        node_df = pd.DataFrame(nodes)
        node_df.set_index("id", inplace=True)
        node_df.sort_index(inplace=True)
        link_df = pd.DataFrame(links)
        link_df.set_index("id", inplace=True)
        link_df.sort_index(inplace=True)

        self.node_gdf = gdp.GeoDataFrame(node_df, geometry="geometry").sort_index()
        self.node_gdf.crs = {'init': crs}
        self.link_gdf = gdp.GeoDataFrame(link_df, geometry="geometry").sort_index()
        self.link_gdf.crs = {'init': crs}

        self.node_gdf.to_crs(epsg=4326, inplace=True)
        self.link_gdf.to_crs(epsg=4326, inplace=True)

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
        :param crs: Original coordinate reference system code
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
        :return: Equivalent dictionary with relevant fields
        """
        from_id = str(elem.get("from"))
        to_id = str(elem.get("to"))

        from_point = node_lookup[from_id]["geometry"]
        to_point = node_lookup[to_id]["geometry"]

        geometry = LineString([from_point, to_point])

        return {
            "id": str(elem.get("id")),
            "from": from_id,
            "to": to_id,
            "length": float(elem.get("length")),
            "lanes": float(elem.get("permlanes")),
            "capacity": float(elem.get("permlanes")) * float(elem.get("capacity")),
            "geometry": geometry,
        }


class TransitSchedule(Tool):
    requirements = ['transit_schedule_path', 'crs']
    stop_gdf = None
    mode_map = None
    modes = None

    def build(self, resources:dict) -> None:
        """
        Transit schedule object constructor.
        :param resources: dict, resources from suppliers
        """
        super().build(resources)

        path = resources['transit_schedule_path'].path
        crs = resources['crs'].path

        # Retrieve stop attributes
        stops = [
            self.transform_stop_elem(elem, crs)
            for elem in get_elems(path, "stopFacility")
        ]

        # Generate empty geodataframes
        stop_df = pd.DataFrame(stops)
        stop_df.set_index("id", inplace=True)
        stop_df.sort_index(inplace=True)

        self.stop_gdf = gdp.GeoDataFrame(stop_df, geometry="geometry").sort_index()

        # Generate routes to modes map
        self.mode_map = dict(
            [
                self.get_route_mode(elem) for elem in get_elems(path, "transitRoute")
            ]
        )

        self.modes = list(set(self.mode_map.values()))

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
            "stop_area": str(elem.get("stopAreaId")),
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


class TransitVehicles(Tool):
    requirements = ['transit_vehicles_path']
    veh_type_mode_map = None
    veh_type_capacity_map = None
    types = None
    veh_id_veh_type_map = None
    transit_vehicle_counts = None

    def build(self, resources):
        """
        Transit vehicles object constructor.
        :param path: Path to MATSim transit vehicles XML file (.xml)
        """
        super().build(resources)

        path = resources['transit_vehicles_path'].path

        # Vehicle types to mode correspondence
        self.veh_type_mode_map = {
            "Rail": "train",
            "Suburban Railway": "train",
            "Bus": "bus",
            "Coach Service": "bus",
            "Tram": "tram",
            "Ferry": "ferry"
        }

        # Vehicle type to total capacity correspondence
        self.veh_type_capacity_map = dict(
            [
                self.transform_veh_type_elem(elem)
                for elem in get_elems(path, "vehicleType")
            ]
        )

        # Vehicle ID to vehicle type correspondence
        self.veh_id_veh_type_map = {
            elem.get("id"): elem.get("type") for elem in get_elems(path, "vehicle")
        }

        self.types, self.transit_vehicle_counts = count_values(self.veh_id_veh_type_map)

    @staticmethod
    def transform_veh_type_elem(elem):
        """
        Extract the vehicle type and total capacity from a vehicleType XML element.
        :param elem: vehicleType XML element
        :return: (vehicle type, capacity) tuple
        """
        strip_namespace(elem)  # TODO only needed for this input = janky
        id = elem.xpath("@id")[0]
        seated_capacity = float(elem.xpath("capacity/seats/@persons")[0])
        standing_capacity = float(elem.xpath("capacity/standingRoom/@persons")[0])
        return id, seated_capacity + standing_capacity


class Attributes(Tool):
    requirements = ['attributes_path']
    final_attribute_map = None
    map = None
    classes = None
    attribute_count_map = None

    def build(self, resources):
        """
        Population subpopulation attributes constructor.
        :param path: Path to MATSim transit vehicles XML file (.xml or .xml.gz)
        """
        super().build(resources)

        path = resources['attributes_path'].path

        # Attribute label mapping
        # TODO move model specific setup elsewhere
        self.final_attribute_map = {
            "inc7p": "inc7p",
            "inc56": "inc56",
            "inc34": "inc34",
            "inc12": "inc12",
            "inc7p_nocar": "inc7p",
            "inc56_nocar": "inc56",
            "inc34_nocar": "inc34",
            "inc12_nocar": "inc12",
            "freight": "freight",
        }

        self.map = dict(
            [
                self.get_attribute_text(elem, 'subpopulation')
                for elem in get_elems(path, "object")
            ]
        )

        self.classes, self.attribute_count_map = count_values(self.map)
        self.classes.append('not_applicable')

    def get_attribute_text(self, elem, tag):
        ident = elem.xpath("@id")[0]
        attribute = elem.find('.//attribute[@name="{}"]'.format(tag))
        attribute = self.final_attribute_map.get(attribute.text, attribute.text)
        return ident, attribute


class Plans(Tool):
    requirements = ['plans_path']
    elems = None
    transit_schedule = None
    modes = None
    activities = None
    agents = None

    def build(self, resources):
        """
        Plans object constructor.
        :param path: Path to MATSim events XML file (.xml)
        :param transit_schedule: TransitSchedule object
        """
        super().build(resources)

        path = resources['plans_path'].path

        self.elems = get_elems(path, "plan")
        self.transit_schedule = TransitSchedule(self.config)
        self.transit_schedule.build(resources)

        self.modes, self.activities = self.get_classes()

        # re-init elements
        self.elems = get_elems(path, "plan")
        self.agents = get_elems(path, "person")

    def get_classes(self):
        """
        Extract sets of used activities and modes from chosen population plans.
        # TODO
        confirm this is how we want to handle plans (ie with handler (iters) and with pre
        calculated inputs...
        Note that we use this method to get all categories before flattening results
        later using a hander, will provide some minor speed up, but is a duplication
        so slows overall. But is more in keeping with overall flow. For example it
        provides early logging of activities and modes used.
        :param elem: Node xml element.
        :return: (List, List)
        """
        modes = set()
        activities = set()

        for plan in self.elems:
            if plan.get('selected') == 'yes':
                for stage in plan:
                    if stage.tag == 'activity':
                        activities.add(stage.get('type'))
                    elif stage.tag == 'leg':
                        mode = stage.get('mode')
                        if mode == 'pt':
                            route = stage.xpath('route')[0].text.split('===')[-2]
                            mode = self.transit_schedule.mode_map.get(route)
                        modes.add(mode)
        return list(modes), list(activities)


class ModeHierarchy(Tool):

    requirements = []

    hierarchy = [
        "ferry",
        "rail",
        "tram",
        "bus",
        "car",
        "bike",
        "walk",
        "transit_walk",
        "access_walk",
        "egress_walk"
    ]

    def get(self, modes: list) -> str:
        if not isinstance(modes, list):
            raise TypeError("ModeHierarchy get method expects list")
        if not all(isinstance(mode, str) for mode in modes):
            raise TypeError("ModeHierarchy get method expects list of strings")
        for mode in modes:
            if mode not in self.hierarchy:
                print(f"WARNING {mode} not found in hierarchy")
        for h in self.hierarchy:
            for mode in modes:
                if h == mode:
                    return mode
        # if mode not found in hierarchy then return middle mode
        mode_index = floor(len(modes) / 2)
        mode = modes[mode_index]
        print(f"WARNING {modes} not in hierarchy, returning {mode}")
        return mode


class ModeMap(Tool):

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


class InputsWorkStation(WorkStation):
    tools = {
        'events': Events,
        'network': Network,
        'transit_schedule': TransitSchedule,
        'transit_vehicles': TransitVehicles,
        'attributes': Attributes,
        'plans': Plans,
        'mode_map': ModeMap,
        'mode_hierarchy': ModeHierarchy,
    }

    def __str__(self):
        return f'Inputs WorkStation'


def get_elems(path, tag):
    """
    Wrapper for unzipping and dealing with xml namespaces
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
    proj = pyproj.Proj(init=crs)
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

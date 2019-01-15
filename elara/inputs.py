import geopandas as gdp
from lxml import etree
import pandas as pd
import pyproj
from shapely.geometry import Point, LineString

WGS_84 = pyproj.Proj(init="epsg:4326")


class Events:
    def __init__(self, path):
        """
        Events object constructor.
        :param path: Path to MATSim events XML file (.xml)
        """
        self.event_elems = get_elems(path, "event")


class Network:
    def __init__(self, path, crs):
        """
        Network object constructor.
        :param path: Path to MATSim network XML file (.xml)
        """

        # Extract element properties
        nodes = [
            self.transform_node_elem(elem, crs) for elem in get_elems(path, "node")
        ]
        node_lookup = {node["id"]: node for node in nodes}
        links = [
            self.transform_link_elem(elem, node_lookup)
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
        self.link_gdf = gdp.GeoDataFrame(link_df, geometry="geometry").sort_index()

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
    def transform_link_elem(elem, node_lookup):
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


class TransitVehicles:
    def __init__(self, path):
        print("hello")
        # Vehicle types to mode correspondence
        self.veh_type_mode_map = {
            "Rail": "train",
            "Suburban Railway": "train",
            "Bus": "bus",
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

    @staticmethod
    def transform_veh_type_elem(elem):
        """
        Extract the vehicle type and total capacity from a vehicleType XML element.
        :param elem: vehicleType XML element
        :return: (vehicle type, capacity) tuple
        """
        id = elem.xpath("@id")[0]
        seatedCapacity = float(elem.xpath("capacity/seats/@persons")[0])
        standingCapacity = float(elem.xpath("capacity/standingRoom/@persons")[0])
        return (id, seatedCapacity + standingCapacity)


def get_elems(path, tag):
    """
    Traverse the given XML tree, retrieving the elements of the specified tag.
    :param tag: The tag type to extract , e.g. 'link'
    :return: Generator of elements
    """
    doc = etree.iterparse(path, tag=tag)
    for _, element in doc:
        yield element
        element.clear()
        del element.getparent()[0]
    del doc


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

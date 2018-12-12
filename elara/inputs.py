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
        self.nodes = [
            self.transform_node_elem(elem, crs) for elem in get_elems(path, "node")
        ]
        node_lookup = {node["id"]: node for node in self.nodes}
        self.links = [
            self.transform_link_elem(elem, node_lookup)
            for elem in get_elems(path, "link")
        ]

        self.node_ids = [node["id"] for node in self.nodes]
        self.link_ids = [link["id"] for link in self.links]

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

from lxml import etree


class Events:
    def __init__(self, path):
        """
        Events object constructor.
        :param path: Path to MATSim events XML file (.xml)
        """
        self.event_elems = get_elems(path, "event")


class Network:
    def __init__(self, path):
        """
        Network object constructor.
        :param path: Path to MATSim network XML file (.xml)
        """
        self.nodes = list(map(self.transform_node_elem, get_elems(path, "node")))
        self.links = list(map(self.transform_link_elem, get_elems(path, "link")))

        self.node_ids = [node["id"] for node in self.nodes]
        self.link_ids = [link["id"] for link in self.links]

    @staticmethod
    def transform_node_elem(elem):
        """
        Convert raw node XML element into dictionary.
        :param elem: Node XML element
        :return: Equivalent dictionary with relevant fields
        """
        return {
            "id": str(elem.get("id")),
            "x": float(elem.get("x")),
            "y": float(elem.get("y")),
        }

    @staticmethod
    def transform_link_elem(elem):
        """
        Convert raw link XML element into dictionary.
        :param elem: Link XML element
        :return: Equivalent dictionary with relevant fields
        """
        return {
            "id": str(elem.get("id")),
            "from": str(elem.get("from")),
            "to": str(elem.get("to")),
            "length": float(elem.get("length")),
            "lanes": float(elem.get("permlanes")),
            "capacity": float(elem.get("permlanes")) * float(elem.get("capacity")),
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

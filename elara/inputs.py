from lxml import etree


class Events:
    def __init__(self, path):
        """
        Events object constructor.
        :param path: Path to MATSim events XML file (.xml)
        """
        self.path = path
        self.events = self.get_events()

    def get_events(self):
        """
        Traverse the events XML tree, retrieving the event elements.
        :return: Generator of event elements
        """
        doc = etree.iterparse(self.path, tag="event")
        for _, element in doc:
            yield element
            element.clear()
            del element.getparent()[0]
        del doc


class Network:
    def __init__(self, path):
        """
        Network object constructor.
        :param path: Path to MATSim network XML file (.xml)
        """
        self.path = path
        self.nodes = {
            elem.get("id"): self.transform_node_elem(elem)
            for elem in self.get_elems("node")
        }
        self.links = {
            elem.get("id"): self.transform_link_elem(elem)
            for elem in self.get_elems("link")
        }

    def get_elems(self, tag):
        """
        Traverse the XML tree, retrieving the elements of the specified
        tag.
        :param tag: The tag type to extract , e.g. 'link'
        :return: Generator of elements
        """
        doc = etree.iterparse(self.path, tag=tag)
        for _, element in doc:
            yield element
            element.clear()
            del element.getparent()[0]
        del doc

    @staticmethod
    def transform_node_elem(elem):
        """
        Convert raw node XML element into dictionary.
        :param elem: Node XML element
        :return: Equivalent dictionary with relevant fields
        """
        return {"x": elem.get("x"), "y": elem.get("y")}

    @staticmethod
    def transform_link_elem(elem):
        """
        Convert raw link XML element into dictionary.
        :param elem: Link XML element
        :return: Equivalent dictionary with relevant fields
        """
        return {
            "from": elem.get("from"),
            "to": elem.get("to"),
            "length": float(elem.get("length")),
            "lanes": float(elem.get("permlanes")),
            "capacity": float(elem.get("permlanes")) * float(elem.get("capacity")),
        }

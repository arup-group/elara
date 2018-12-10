from pathlib import Path

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
        self.links = list(self.get_ids("link"))
        self.nodes = list(self.get_ids("node"))

    def get_ids(self, tag):
        """
        Traverse the network XML tree, retrieving the 'id' field of elements
        with the specified tag.
        :param tag: The tag type to extract the 'id' field from, e.g. 'link'
        :return: Generator of 'id' values
        """
        doc = etree.iterparse(self.path, tag=tag)
        for _, element in doc:
            yield element.get("id")
            element.clear()
            del element.getparent()[0]
        del doc

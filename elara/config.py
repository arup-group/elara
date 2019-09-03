import os.path
import toml
from elara.factory import WorkStation, Tool


class Config:

    def __init__(self, path):
        """
        Config object constructor.
        :param path: Path to scenario configuration TOML file
        """
        self.parsed_toml = toml.load(path, _dict=dict)

        # Scenario settings
        self.name = self.parsed_toml["scenario"]["name"]
        self.time_periods = self.valid_time_periods(
            self.parsed_toml["scenario"]["time_periods"]
        )
        self.scale_factor = self.valid_scale_factor(
            self.parsed_toml["scenario"]["scale_factor"]
        )
        self.verbose = self.parsed_toml["scenario"].get("verbose", False)

        # Factory requirements
        self.event_handlers = self.parsed_toml.get("event_handlers", {})
        self.plan_handlers = self.parsed_toml.get("plan_handlers", {})
        self.post_processors = self.parsed_toml.get("post_processors", {})
        self.benchmarks = self.parsed_toml.get("benchmarks", {})

        # Output settings
        self.output_path = self.parsed_toml["outputs"]["path"]
        self.contract = self.parsed_toml["outputs"].get("contract", False)

    @property
    def dummy_path(self):
        return self.parsed_toml["scenario"]["name"]

    @property
    def crs(self):
        return self.parsed_toml["scenario"]["crs"]

    @property
    def events_path(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["events"], "events"
        )

    @property
    def plans_path(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["plans"], "plans"
        )

    @property
    def network_path(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["network"], "network"
        )

    @property
    def attributes_path(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["attributes"], "attributes"
        )

    @property
    def transit_schedule_path(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["transit_schedule"], "transit_schedule"
        )

    @property
    def transit_vehicles_path(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["transit_vehicles"], "transit_vehicles"
        )

    @property
    def output_config_path(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["output_config_path"], "output_config"
        )

    @staticmethod
    def valid_time_periods(inp):
        """
        Raise exception if specified number of time periods is outside an acceptable
        range (15 minute slice minimum), otherwise return time periods.
        :param inp: Number of time periods
        :return: Pass through number of time periods if valid
        """
        if inp <= 0 or inp > 96:
            raise Exception(
                "Specified time periods ({}) not in valid range".format(inp)
            )
        return int(inp)

    @staticmethod
    def valid_scale_factor(inp):
        """
        Raise exception if specified scale factor is outside an acceptable range, i.e.
        beyond (0, ], otherwise return scale factor.
        :param inp: Scale factor
        :return: Pass through scale factor if valid
        """
        if inp <= 0 or inp > 1:
            raise Exception(
                "Specified scale factor ({}) not in valid range".format(inp)
            )
        return float(inp)

    @staticmethod
    def valid_path(path, field_name):
        """
        Raise exception if specified path does not exist, otherwise return path.
        :param path: Path to check
        :param field_name: Field name to use in exception if path does not exist
        :return: Pass through path if it exists
        """
        if not os.path.exists(path):
            raise Exception("Specified path for {} does not exist".format(field_name))
        return path


class GetCRS(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.crs


class GetEventsPath(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.events_path


class GetPlansPath(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.plans_path


class GetNetworkPath(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.network_path


class GetAttributesPath(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.attributes_path


class GetTransitSchedulePath(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.transit_schedule_path


class GetTransitVehiclesPath(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.transit_vehicles_path


class GetOutputConfigPath(Tool):
    path = None

    def build(self, resource: dict):
        super().build(resource)
        self.path = self.config.output_config_path


class PathFinderWorkStation(WorkStation):
    tools = {
        'crs': GetCRS,
        'events_path': GetEventsPath,
        'plans_path': GetPlansPath,
        'network_path': GetNetworkPath,
        'attributes_path': GetAttributesPath,
        'transit_schedule_path': GetTransitSchedulePath,
        'transit_vehicles_path': GetTransitVehiclesPath,
        'output_config_path': GetOutputConfigPath,
    }

    def __str__(self):
        return f'PathFinder WorkStation'


class RequirementsWorkStation(WorkStation):

    tools = None

    def gather_manager_requirements(self):
        reqs = {}
        reqs.update(self.config.event_handlers)
        reqs.update(self.config.plan_handlers)
        reqs.update(self.config.post_processors)
        reqs.update(self.config.benchmarks)
        return reqs

    def __str__(self):
        return f'Requirements WorkStation'


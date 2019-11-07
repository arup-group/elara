import os.path
import toml
from elara.factory import WorkStation, Tool
import logging

logger = logging.getLogger(__name__)


class Config:

    def __init__(self, path):
        """
        Config object constructor.
        :param path: Path to scenario configuration TOML file
        """
        self.logger = logging.getLogger(__name__)

        self.logger.debug(f'Loading config from {path}')
        self.parsed_toml = toml.load(path, _dict=dict)

        # Scenario settings
        self.logger.debug(f'Loading scenario settings from config')

        self.name = self.parsed_toml["scenario"]["name"]
        self.logger.debug(f'Scenario name = {self.name}')

        self.time_periods = self.valid_time_periods(
            self.parsed_toml["scenario"]["time_periods"]
        )
        self.logger.debug(f'Scenario time periods = {self.time_periods}')

        self.scale_factor = self.valid_scale_factor(
            self.parsed_toml["scenario"]["scale_factor"]
        )
        self.logger.debug(f'Scale factor = {self.scale_factor}')

        self.logging = self.valid_verbosity(
            self.parsed_toml["scenario"].get("verbose", False)
        )

        env_level = os.environ.get('ELARA_LOGLEVEL', False)
        if env_level:
            self.logging = self.valid_verbosity(env_level)
            self.logger.warning(f'***Config logging level overwritten by env to {env_level}***')
        self.logger.debug(f'Verbosity/logging = {self.logging}')

        # Factory requirements
        self.logger.debug(f'Loading factory build requirements from config')

        self.event_handlers = self.parsed_toml.get("event_handlers", {})
        self.logger.debug(f'Required Event Handlers = {self.event_handlers}')

        self.plan_handlers = self.parsed_toml.get("plan_handlers", {})
        self.logger.debug(f'Required Plan Handlers = {self.plan_handlers}')

        self.post_processors = self.parsed_toml.get("post_processors", {})
        self.logger.debug(f'Required Post Processors = {self.post_processors}')

        self.benchmarks = self.parsed_toml.get("benchmarks", {})
        self.logger.debug(f'Required Benchmarks = {self.benchmarks}')

        # Output settings
        self.logger.debug(f'Loading output settings from config')

        self.output_path = self.parsed_toml["outputs"]["path"]
        self.logger.debug(f'Output Path = {self.output_path}')

        self.contract = self.parsed_toml["outputs"].get("contract", False)
        self.logger.debug(f'Contract = {self.contract}')

        if not os.path.exists(self.output_path):
            self.logger.info(f'Creating output path: {self.output_path}')
            os.mkdir(self.output_path)
            
        benchmarks_path = os.path.join(self.output_path, "benchmarks")
        if not os.path.exists(benchmarks_path):
            self.logger.info(f'Creating output path: {benchmarks_path}')
            os.mkdir(benchmarks_path)

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

    @staticmethod
    def valid_verbosity(inp):
        """
        Raise exception if specified verbosity does not exist, otherwise return logging level.
        :param inp: Proposed logging level
        :return: logging level
        """
        options = {
            'true': logging.DEBUG,
            'false': logging.INFO,
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'warn': logging.WARN,
            'warning': logging.WARNING
        }
        if not str(inp).lower() in options:
            raise Exception(f"Config verbosity/logging level must be one of: {options.keys()}")
        return options[str(inp).lower()]


class PathTool(Tool):

    def __init__(self, config, option=None):
        super().__init__(config, option)
        self.logger = logging.getLogger(__name__)


class GetCRS(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
        super().build(resource)
        self.path = self.config.crs


class GetEventsPath(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
        super().build(resource)
        self.path = self.config.events_path


class GetPlansPath(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
        super().build(resource)
        self.path = self.config.plans_path


class GetNetworkPath(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
        super().build(resource)
        self.path = self.config.network_path


class GetAttributesPath(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
        super().build(resource)
        self.path = self.config.attributes_path


class GetTransitSchedulePath(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
        super().build(resource)
        self.path = self.config.transit_schedule_path


class GetTransitVehiclesPath(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
        super().build(resource)
        self.path = self.config.transit_vehicles_path


class GetOutputConfigPath(PathTool):
    path = None

    def build(self, resource: dict, write_path=None):
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

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)


class RequirementsWorkStation(WorkStation):

    tools = None

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def gather_manager_requirements(self):
        reqs = {}
        reqs.update(self.config.event_handlers)
        reqs.update(self.config.plan_handlers)
        reqs.update(self.config.post_processors)
        reqs.update(self.config.benchmarks)
        return reqs

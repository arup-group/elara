import ast
import os.path
import toml
from elara.factory import WorkStation, Tool
import logging
from elara import ConfigError

logger = logging.getLogger(__name__)


class Config:

    default_settings = {
        "scenario":
            {
                "name": 'default',
                "time_periods": 24,
                "scale_factor": .1,
                "crs": "EPSG:27700",
                "verbose": False,
            },
        "inputs":
            {
                "events": "./output_events.xml.gz",
                "network": "./output_network.xml.gz",
                "transit_schedule": "./output_transitSchedule.xml.gz",
                "transit_vehicles": "./output_transitVehicles.xml.gz",
                "attributes": "./output_personAttributes.xml.gz",
                "plans": "./output_plans.xml.gz",
                "output_config_path": "./output_config.xml",
            },
        "outputs":
            {
                "path": './elara_outputs',
                "contract": True,
            },
    }

    def __init__(self, path=None, override=None):
        """
        Config object constructor.
        :param path: Path to scenario configuration TOML file
        """
        self.logger = logging.getLogger(__name__)
        self.name = None
        self.time_periods = None
        self.scale_factor = None
        self.logging = None
        self.event_handlers = None
        self.plan_handlers = None
        self.post_processors = None
        self.benchmarks = None
        self.output_path = None
        self.contract = None

        if path:
            self.logger.debug(f' Loading config from {path}')
            self.settings = toml.load(path, _dict=dict)
        elif override:
            self.logger.debug(f' Loading config from dict override')
            self.settings = override
        else:
            self.logger.debug(f' Loading default config')
            self.settings = self.default_settings

        self.load_required_settings()

        if not os.path.exists(self.output_path):
            self.logger.info(f'Creating output path: {self.output_path}')
            os.mkdir(self.output_path)

        benchmarks_path = os.path.join(self.output_path, "benchmarks")
        if not os.path.exists(benchmarks_path):
            self.logger.info(f'Creating output path: {benchmarks_path}')
            os.mkdir(benchmarks_path)

        self.logger.debug(f'Scenario name = {self.name}')
        self.logger.debug(f'Output Path = {self.output_path}')
        self.logger.debug(f'Scenario time periods = {self.time_periods}')
        self.logger.debug(f'Scale factor = {self.scale_factor}')
        self.logger.debug(f'Verbosity/logging = {self.logging}')
        self.logger.debug(f'Required Event Handlers = {self.event_handlers}')
        self.logger.debug(f'Required Plan Handlers = {self.plan_handlers}')
        self.logger.debug(f'Required Post Processors = {self.post_processors}')
        self.logger.debug(f'Required Benchmarks = {self.benchmarks}')
        self.logger.debug(f'Contract = {self.contract}')

    def load_required_settings(self):

        # Scenario settings
        self.logger.debug(f'Loading and validating required settings from config')

        self.name = self.settings["scenario"]["name"]
        self.time_periods = self.valid_time_periods(
            self.settings["scenario"]["time_periods"]
        )
        self.scale_factor = self.valid_scale_factor(
            self.settings["scenario"]["scale_factor"]
        )
        self.logging = self.valid_verbosity(
            self.settings["scenario"].get("verbose", False)
        )

        # Factory requirements
        self.logger.debug(f'Loading factory build requirements from config')
        self.event_handlers = self.settings.get("event_handlers", {})
        self.plan_handlers = self.settings.get("plan_handlers", {})
        self.post_processors = self.settings.get("post_processors", {})
        self.benchmarks = self.settings.get("benchmarks", {})

        # Output settings
        self.logger.debug(f'Loading output settings from config')
        self.output_path = self.settings["outputs"]["path"]
        self.contract = self.settings["outputs"].get("contract", False)

    """
    Property methods used for config dependant requirements.
    For example crs is only required if spatial processing required.
    """

    @property
    def dummy_path(self):
        return self.settings["scenario"]["name"]

    @property
    def crs(self):
        return self.valid_crs(
            self.settings["scenario"]["crs"]
        )

    @property
    def events_path(self):
        return self.valid_path(
            self.settings["inputs"]["events"], "events"
        )

    @property
    def plans_path(self):
        return self.valid_path(
            self.settings["inputs"]["plans"], "plans"
        )

    @property
    def network_path(self):
        return self.valid_path(
            self.settings["inputs"]["network"], "network"
        )

    @property
    def attributes_path(self):
        return self.valid_path(
            self.settings["inputs"]["attributes"], "attributes"
        )

    @property
    def transit_schedule_path(self):
        return self.valid_path(
            self.settings["inputs"]["transit_schedule"], "transit_schedule"
        )

    @property
    def transit_vehicles_path(self):
        return self.valid_path(
            self.settings["inputs"]["transit_vehicles"], "transit_vehicles"
        )

    @property
    def output_config_path(self):
        return self.valid_path(
            self.settings["inputs"]["output_config_path"], "output_config"
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
            raise ConfigError(
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
            raise ConfigError(
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
            raise ConfigError("Specified path {} for {} does not exist".format(path, field_name))
        return path

    def valid_verbosity(self, inp):
        """
        Raise exception if specified verbosity does not exist, otherwise return logging level.
        Allows overwrite by env variable.
        :param inp: Proposed logging level
        :return: logging level
        """
        env_level = os.environ.get('ELARA_LOGLEVEL', None)
        if env_level is not None:
            self.logger.warning(f'*** Config logging level overwritten by env to {env_level} ***')
            inp = env_level

        options = {
            'true': logging.DEBUG,
            'false': logging.INFO,
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'warn': logging.WARN,
            'warning': logging.WARNING
        }
        if not str(inp).lower() in options:
            raise ConfigError(f"Config verbosity/logging level must be one of: {options.keys()}")
        return options[str(inp).lower()]

    def valid_crs(self, inp):
        """
        Raise exception if specified verbosity does not exist, otherwise return logging level.
        :param inp: Proposed crs
        :return: logging level
        """
        if inp == 'None':
            self.logger.warning(f'Re-projection disabled at configuration')
            return None

        if isinstance(inp, int):
            inp = f"EPSG:{inp}"
            self.logger.warning(f'Configured CRS inferred as {inp}.')

        if not isinstance(inp, str):
            raise ConfigError('Configured CRS should be string format, for example: "EPSG:27700"')
        return inp

    def override(self, path_override):
        """
        :param path_override: override the config input paths
        """
        # Construct a dictionary from the path_overrides str
        for path in self.settings['inputs']:
            file_name = self.settings['inputs'][path].split('/')[-1]
            self.settings['inputs'][path] = "{}/{}".format(path_override, file_name)

        self.settings['outputs']['path'] = path_override
        self.output_path = path_override


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

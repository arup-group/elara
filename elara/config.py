import os.path
import toml
from elara import ConfigInputError


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
        self.crs = self.parsed_toml["scenario"]["crs"]
        self.verbose = self.parsed_toml["scenario"]["verbose"]

        # Handler objects
        self.event_handlers = self.parsed_toml["event_handlers"]
        self.plan_handlers = self.parsed_toml["plan_handlers"]

        # Output settings
        self.output_path = self.parsed_toml["outputs"]["path"]
        self.contract = self.parsed_toml["outputs"]["contract"]
        self.post_processing = self.parsed_toml["outputs"]["post_processing"]

        # Benchmark settings
        self.benchmarks = self.parsed_toml["benchmarking"]["benchmarks"]

        self.events_path = None

    def validate_required_paths(self, requirements):
        for requirement in requirements:
            if not hasattr(self, requirement):
                raise ConfigInputError(f'Missing input path: {requirement} found in requirements')

    @property
    def events(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["events"], "events"
        )

    @property
    def plans(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["plans"], "plans"
        )

    @property
    def network(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["network"], "network"
        )

    @property
    def attributes(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["attributes"], "attributes"
        )

    @property
    def transit_schedule(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["transit_schedule"], "transit_schedule"
        )

    @property
    def transit_vehicles(self):
        return self.valid_path(
            self.parsed_toml["inputs"]["transit_vehicles"], "transit_vehicles"
        )

    @property
    def mode_hierarchy(self):
        return 'place_holder'

    @property
    def mode_map(self):
        return 'place_holder'

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

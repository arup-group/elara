import os.path

import toml


class Config:
    def __init__(self, path):
        """
        Config object constructor.
        :param path: Path to scenario configuration TOML file
        """
        parsed_toml = toml.load(path, _dict=dict)

        # Scenario settings
        self.name = parsed_toml["scenario"]["name"]
        self.time_periods = self.valid_time_periods(
            parsed_toml["scenario"]["time_periods"]
        )
        self.scale_factor = self.valid_scale_factor(
            parsed_toml["scenario"]["scale_factor"]
        )
        self.crs = parsed_toml["scenario"]["crs"]

        # Input settings
        self.events_path = self.valid_path(parsed_toml["inputs"]["events"], "events")
        self.network_path = self.valid_path(parsed_toml["inputs"]["network"], "network")
        self.transit_schedule_path = self.valid_path(
            parsed_toml["inputs"]["transit_schedule"], "transit_schedule"
        )
        self.transit_vehicles_path = self.valid_path(
            parsed_toml["inputs"]["transit_vehicles"], "transit_vehicles"
        )

        # Handler objects
        self.handlers = parsed_toml["handlers"]

        # Output settings
        self.output_path = parsed_toml["outputs"]["path"]
        self.contract = parsed_toml["outputs"]["contract"]
        self.post_processing = parsed_toml["outputs"]["post_processing"]

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

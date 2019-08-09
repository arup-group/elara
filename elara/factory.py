from elara.config import Config
from elara.inputs import *
from elara.handlers import *
from elara.postprocessing import *
from elara.benchmarking import *
from elara import ConfigError
from elara import ConfigPostProcessorError, PostProcessorPrerequisiteError


class Manager:
    tools = {}
    resources = {}

    def __init__(self, factory_supervisor, managers, suppliers):

        self.factory_supervisor = factory_supervisor
        self.managers = managers
        self.suppliers = self.suppliers

    def validate(self):
        if not self.managers:
            raise ConfigError('Unknown manager for {type(self).}')
        for manager in self.managers:
            for requirement in list(manager.requirements()):
                resource = self.tools.get(requirement, None)
                if not resource:
                    raise ConfigError(
                        f'Unknown resource: {resource} required.'
                    )

    def requirements(self):
        if not self.manager:
            raise ConfigError('Unknown manager for {type(self).}')
        required = set()
        for requirement in self.manager.requirements():
            resource = self.tools[requirement]
            required.update(resource.requirements)
        return list(required)

    def build(self):
        if not self.manager:
            raise ConfigError('Unknown manager for {type(self).}')
        # build in order of dict, ie prioritise
        requirements = self.manager.requirements()
        for tool_name, tool in self.tools.items():
            if tool_name in requirements:
                self.resources[tool_name] = self.tools[tool_name](self.supplier)


class ConfigSupervisor(Manager):

    def _validate(self):
        config = self.factory_supervisor.config
        for resource in self.manager.requirements():
            if not hasattr(config, resource):
                raise ConfigError(
                    f'Unknown resource path: {resource} required.'
                )
            path = getattr(config, resource)

    def requirements(self):
        return None

    def build(self):
        print(f'--> Building {type(self)} requirements -->')
        for todo in self.manager.requirements():
            self.resources[todo] = self.factory_supervisor.config.__getattribute__(
                todo
            )


class InputSupervisor(Manager):
    supplier_class = ConfigSupervisor

    MAP = {
        'events': Events,
        'network': Network,
        'transit_schedule': TransitSchedule,
        'transit_vehicles': TransitVehicles,
        'attributes': Attributes,
        'plans': Plans,
        'mode_map': ModeMap,
        'mode_hierarchy': ModeHierarchy,
    }

    priorities = [
        'events',
        'plans',
        'network',
        'attributes',
        'transit_schedule',
        'transit_vehicles',
        'mode_hierarchy',
        'mode_map'
    ]

    def build(self):
        print(f'--> Building {type(self)}')
        for todo in (self._prioritise(self.manager.requirements())):
            self.resources[todo] = self.MAP[todo](self.supplier.resources)

    def print(self):
        print('--- Input Summary ---')
        for required_input in self.resources:
            print(required_input)


class HandlerSupervisor(Manager):
    supplier_class = InputSupervisor

    MAP = {
        "volume_counts": VolumeCounts,
        "passenger_counts": PassengerCounts,
        "stop_interactions": StopInteractions,
        "activities": Activities,
        "legs": Legs,
        "mode_share": ModeShare,
    }

    def build(self):
        for handler_name, selections in self.manager.requirements().items():
            for selection in selections:
                self.resources[handler_name + '-' + selection] = self.MAP[handler_name](
                            selection=selection,
                            resources=self.supplier.resources,
                            time_periods=self.factory_supervisor.config.time_periods,
                            scale_factor=self.factory_supervisor.config.scale_factor,
                        )


class OutputSupervisor(Manager):
    supplier_class = HandlerSupervisor
    tools = {}
    resources = {}

    def __init__(self, config_path):
        super().__init__(self, manager=None)

        # initiate config
        self.config = Config(config_path)

    def _validate(self):
        for

    def requirements(self):
        return {**self.config.event_handlers, **self.config.plan_handlers}


    @staticmethod
    def user_check_config():
        if not input('Continue? (y/n) --').lower() in ('y', 'yes', 'ok'):
            quit()


class PostProcessorSupervisor(Manager):
    pass


class Supervisor(Manager):

    post_process_super = PostProcessorSupervisor()

    chain




# class PostProcessManager(Manager):
#
#     reportees = None
#
#     def __init__(self, config):
#
#         # Dictionary used to map configuration string to post-processor type
#         self.POST_PROCESSOR_MAP = {"vkt": VKT}
#
#         self.config = config
#         self._validate_config()
#
#         self.post_processors = []
#
#     def prepare(self, input_manager, handler_manager):
#         for post_processor_name in self.config.post_processing:
#             post_processor = self.POST_PROCESSOR_MAP[post_processor_name](
#                     self.config,
#                     input_manager.resources['network'],
#                     input_manager.resources['transit_schedule'],
#                     input_manager.resources['transit_vehicles']
#                 )
#             post_processor.check_handler_prerequisite(handler_manager)
#             self.post_processors.append(post_processor)
#
#     def _validate_config(self):
#         for post_processor_name in self.config.post_processing:
#             post_processor_class = self.POST_PROCESSOR_MAP.get(post_processor_name, None)
#             if not post_processor_class:
#                 raise ConfigPostProcessorError(
#                     f'Unknown post=processor name: {post_processor_name} found in config')
#
#
# class BenchmarkManager(Manager):
#
#     reportees = None
#
#     def __init__(self, config):
#
#         self.BENCHMARK_MAP = {"london_inner_cordon_car": LondonInnerCordonCar,
#                          "dublin_canal_cordon_car": DublinCanalCordonCar,
#                          "ireland_commuter_modeshare": IrelandCommuterStats,
#                          "test_town_cordon": TestTownHourlyCordon,
#                          "test_town_peak_cordon": TestTownPeakIn,
#                          "test_town_modeshare": TestTownCommuterStats}
#
#         self.BENCHMARK_WEIGHTS = {"london_inner_cordon_car": 1,
#                              "dublin_canal_cordon_car": 1,
#                              "ireland_commuter_modeshare": 1,
#                              "test_town_cordon": 1,
#                              "test_town_peak_cordon": 1,
#                              "test_town_modeshare": 1}
#
#         self.config = config
#         self._validate_config()
#
#         self.benchmarks = None
#
#     def prepare(self, input_manager, handler_manager):
#         self.benchmarks = Benchmarks(self.config)
#
#     def _validate_config(self):
#         for benchmark_name in self.config.benchmarks:
#             benchmark_class = self.BENCHMARK_MAP.get(benchmark_name, None)
#             if not benchmark_class:
#                 raise ConfigBenchmarkError(
#                     f'Unknown benchmark name: {benchmark_name} found in config')

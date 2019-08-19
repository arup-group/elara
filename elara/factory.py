class WorkStation:
    """
    Base Class for holding dictionary of Tool objects.
    """
    depth = 0
    tools = {}

    def __init__(self, config):
        self.resources = {}
        self.requirements = {}
        self.config = config
        self.managers = None
        self.suppliers = None
        self.supplier_resources = {}

    def connect(self, managers: list, suppliers: list) -> None:
        """
        Connect workstations to their respective managers and suppliers to form a DAG.
        Note that arguments should be provided as lists.
        :param managers: list of managers
        :param suppliers: list of suppliers
        :return: None
        """
        self.managers = managers
        self.suppliers = suppliers

    def engage(self) -> None:

        all_requirements = []

        # get manager requirements
        manager_requirements = self.gather_manager_requirements()
        if not self.tools:
            self.requirements = manager_requirements

        # init required tools
        # build new requirements from tools
        # loop tool first looking for matches so that tool order is preserved
        else:
            for tool_name, tool in self.tools.items():
                for manager_requirement, options in manager_requirements.items():
                    if manager_requirement == tool_name:
                        if options:
                            for option in options:
                                # init
                                key = str(tool_name) + ':' + str(option)
                                self.resources[key] = tool(self.config, option)
                                tool_requirements = self.resources[key].get_requirements()
                                all_requirements.append(tool_requirements)
                        else:
                            # init
                            key = str(tool_name)
                            self.resources[key] = tool(self.config)
                            tool_requirements = self.resources[key].get_requirements()
                            all_requirements.append(tool_requirements)

            self.requirements = combine_reqs(all_requirements)

    def validate_suppliers(self) -> None:

        # gather supplier tools
        supplier_tools = {}
        if self.suppliers:
            for supplier in self.suppliers:
                if not supplier.tools:
                    continue
                supplier_tools.update(supplier.tools)

        # check for missing requirements
        missing = set(self.requirements) - set(supplier_tools)
        if missing:
            raise ValueError(
                f'Missing requirements: {missing} from suppliers: {self.suppliers}.'
            )

    def gather_manager_requirements(self):
        reqs = []
        if self.managers:
            for manager in self.managers:
                reqs.append(manager.requirements)
        return combine_reqs(reqs)

    def build(self):
        """
        Gather resources from suppliers for current workstation and build() all resources in
        order of .resources map.
        :return: None
        """
        # gather resources
        if self.suppliers:
            for supplier in self.suppliers:
                self.supplier_resources.update(supplier.resources)

        if self.resources:
            for tool_name, tool in self.resources.items():
                tool.build(self.supplier_resources)

    def load_all_tools(self, option=None):
        for name, tool in self.tools.items():
            self.resources[name] = tool(self.config, option)


# Define tools to be used by Sub Processes
class Tool:
    """
    Base tool class.
    """
    requirements = None
    valid_options = None  # todo this might be more useful as INVALID OPTIONS or optional both
    resources = None
    options_enabled = False

    def __init__(self, config, option=None):
        """
        Initiate a tool instance with optional option (ie: 'bus'). Raise UserWarning if option is
        not in .valid_options.
        :param option: optional option, typically assumed to be str
        """
        self.config = config
        if self.valid_options:
            if option not in self.valid_options:
                raise UserWarning(f'Unsupported option: {option} at tool: {self}')
        self.option = option

    def get_requirements(self) -> dict:
        """
        Default return requirements of tool for given .option.
        Returns None if .option is None.
        :return: dict of requirements
        """
        if not self.requirements:
            return None

        if self.options_enabled:
            requirements = {req: [self.option] for req in self.requirements}
        else:
            requirements = {req: None for req in self.requirements}

        # if self.config.verbose:
        #     print(f"> Req for tool: {self} Option: {self.option} = {requirements}")
        return requirements

    def build(self, resource: dict) -> None:
        """
        Default build self.
        :param resource:
        :return: None
        """
        for requirement in convert_to_unique_keys(self.get_requirements()):
            if requirement not in list(resource):
                raise ValueError(f'Missing requirement: {requirement}')

        self.resources = resource


def operate_workstation_graph(start_node: WorkStation, verbose=False) -> list:
    """
    Main function for validating graph requirements, then initiating and building minimum resources.

    Stage1: Traverse graph from starting workstation to suppliers with depth-first search,
    marking workstations with possible longest path.

    Stage 2: Traverse graph with breadth-first search, prioritising shallowest nodes, sequentially
    initiating required workstation .tools as .resources and building .requirements.

    Stage 3: Traverse graph along same path but backward, gathering resources from suppliers at
    each workstation and building all own resources.

    Note that circular dependencies are not supported.

    Note that the function should be given the workstation with the initial/final requirements
    for the factory.

    :param start_node: starting workstation
    :param verbose: bool, verbose behaviour
    :return: list, sequence of visits for stages 2 (initiation and validation) and 3 (building)
    """

    # stage 1:
    build_graph_depth(start_node)

    # stage 2:
    visited = []
    queue = []
    start_node.engage()
    # start_node._engage_suppliers()
    queue.append(start_node)
    visited.append(start_node)

    while queue:
        current = queue.pop(0)
        if verbose:
            print(f'> Engaging: {current}')
        current.engage()

        if current.suppliers:
            if verbose:
                print(f'> Validating suppliers for: {current}')
            current.validate_suppliers()

            for supplier in order_by_distance(current.suppliers):
                if supplier not in visited:
                    queue.append(supplier)
                    visited.append(supplier)

    # stage 3:
    sequence = visited
    return_queue = visited[::-1]
    visited = []
    while return_queue:
        current = return_queue.pop(0)
        if verbose:
            print(f'> Building: {current}')
        current.build()
        visited.append(current)
    # return full sequence for testing
    return sequence + visited


def build_graph_depth(node: WorkStation, visited=None, depth=0) -> list:
    """
    Function to recursive depth-first traverse graph of suppliers, recording workstation depth in
    graph.
    :param node: starting workstation
    :param visited: list, visited workstations
    :param depth: current depth
    :return: list, visited workstations in order
    """
    if not visited:
        visited = []
    visited.append(node)

    # Recur for all the nodes supplying this node
    if node.suppliers:
        depth = depth + 1
        for supplier in node.suppliers:
            if supplier.depth < depth:
                supplier.depth = depth
            build_graph_depth(supplier, visited, depth)

    return visited


def order_by_distance(candidates: list) -> list:
    """
    Returns candidate list ordered by .depth.
    :param candidates: list, of workstations
    :return: list, of workstations
    """
    return sorted(candidates, key=lambda x: x.depth, reverse=False)


def combine_reqs(reqs: list) -> dict:
    """
    Helper function for combining lists of requirements (dicts of lists) into a single
    requirements dict:

    [{req1:[a,b], req2:[b]}, {req1:[a], req2:[a], req3:None}] -> {req1:[a,b], req2:[a,b], req3:None}

    Note that no requirements are returned as an empty dict.

    Note that no options are returned as None ie {requirment: None}
    :param reqs: list of dicts of lists
    :return: dict, of requirements
    """
    if not reqs:
        return {}
    tool_set = set()
    for req in reqs:
        if req:
            tool_set.update(list(req))
    combined_reqs = {}
    for tool in tool_set:
        if tool_set:
            options = set()
            for req in reqs:
                if req and req.get(tool):
                    options.update(req[tool])
            if options:
                combined_reqs[tool] = list(options)
            else:
                combined_reqs[tool] = None
    return combined_reqs


def convert_to_unique_keys(d: dict) -> list:
    """
    Helper function to convert a requirements dictionary into a list of unique keys:

    {req1:[a,b], req2:[a], req3:None} -> ['req1:a', 'req1:b', 'req2:a', 'req3'}

    Note that if option is None, key will be returned as requirement key only.

    :param d: dict, of requirements
    :return: list
    """
    keys = []
    if not d:
        return []
    for name, options in d.items():
        if not options:
            keys.append(name)
            continue
        for option in options:
            keys.append(f'{name}:{option}')
    return keys


def list_equals(l1, l2):
    """
    Helper function to check for equality between two lists of options.
    :param l1: list
    :param l2: list
    :return: bool
    """
    if l1 is None:
        if l2 is None:
            return True
        return False
    if not len(l1) == len(l2):
        return False
    if not sorted(l1) == sorted(l2):
        return False
    return True


def equals(d1, d2):
    """
    Helper function to check for equality between two dictionaries of requirements.
    :param d1: dict
    :param d2: dict
    :return: bool
    """
    if not list_equals(list(d1), list(d2)):
        return False
    for d1k, d1v, in d1.items():
        if not list_equals(d1v, d2[d1k]):
            return False
    return True

###########

#
#
#
# class PPPipe(Pipe):
#     req = ['volume_counts']
#
#
# class Factory:
#
#     def __init__(self, config):
#         if config.post_processors:
#             self.post_processors = PostProcessor
#
#     def _build(self):
#         pass
#
#     def _build_requirements(self):
#         req = []
#         if self.config.plan_handlers:
#             req.append(['plan_handlers'])
#         if self.config.event_handlers:
#             req.append(['event_handlers'])
#         if self.config.post_processors:
#             req.append(['post_processors'])
#         # benchmarkers = self.config.benchmarks
#         return req
#
#
# class PlanManager(Manager):
#
#     _tools = {
#         'plan_handlers': Pipe,
#         'event_handlers': Pipe,
#         'post_processors': PPPipe,
#     }
#
#     def _build(self):
#         print(f'STREAM {self}')
#
#     def _build_requirements(self):
#         req = super()._build_requirements()
#         req.update(self.config.plan_handlers)
#
# #
# # class BProcess(Manager):
# #     _tools = {
# #         'a': Tool1,
# #     }
# #
# #
# # class CProcess(Manager):
# #     _tools = {
# #         'b': Tool3,
# #     }
# #
# #
# # class DProcess(Manager):
# #     _tools = {
# #         'a': Tool1,
# #         'b': Tool3,
# #         'e': Tool2,
# #         'c': Tool5,
# #     }
# #
# #
# # class FinalProcess(Manager):
# #     _tools = {
# #         'a': 'A',
# #         'b': 'B',
# #         'c': 'C',
# #         'd': 'D',
# #         'e': 'E',
# #         'f': 'F',
# #     }
# #
# #     def _build(self):
# #         print('building final')
# #         self._resources = self._tools
#
#
# class ConfigSupervisor(Manager):
#
#     @property
#     def tools(self):
#         return list(self._tools)
#
#     def _build_requirements(self):
#         return None
#
#     def _build(self):
#         print(f'building: {self}')
#         for todo in self._gather_manager_requirements():
#             self._resources[todo] = self.config.__getattribute__(
#                 todo
#             )
#
#
# class InputSupervisor(Manager):
#     supplier_class = ConfigSupervisor
#
#     MAP = {
#         'events': Events,
#         'network': Network,
#         'transit_schedule': TransitSchedule,
#         'transit_vehicles': TransitVehicles,
#         'attributes': Attributes,
#         'plans': Plans,
#         'mode_map': ModeMap,
#         'mode_hierarchy': ModeHierarchy,
#     }
#
#     priorities = [
#         'events',
#         'plans',
#         'network',
#         'attributes',
#         'transit_schedule',
#         'transit_vehicles',
#         'mode_hierarchy',
#         'mode_map'
#     ]
#
#     def build(self):
#         print(f'--> Building {type(self)}')
#         for todo in (self._prioritise(self.manager.requirements())):
#             self._resources[todo] = self.MAP[todo](self.supplier.resources)
#
#     def print(self):
#         print('--- Input Summary ---')
#         for required_input in self._resources:
#             print(required_input)
#
#
# class HandlerSupervisor(Manager):
#     supplier_class = InputSupervisor
#
#     MAP = {
#         "volume_counts": VolumeCounts,
#         "passenger_counts": PassengerCounts,
#         "stop_interactions": StopInteractions,
#         "activities": Activities,
#         "legs": Legs,
#         "mode_share": ModeShare,
#     }
#
#     def build(self):
#         for handler_name, selections in self.manager.requirements().items():
#             for selection in selections:
#                 self._resources[handler_name + '-' + selection] = self.MAP[handler_name](
#                             selection=selection,
#                             resources=self.supplier.resources,
#                             time_periods=self.factory_supervisor.config.time_periods,
#                             scale_factor=self.factory_supervisor.config.scale_factor,
#                         )
#
#
# class OutputSupervisor(Manager):
#     supplier_class = HandlerSupervisor
#     tools = {}
#     _resources = {}
#
#     def __init__(self, config_path):
#         super().__init__(self, manager=None)
#
#         # initiate config
#         self.config = Config(config_path)
#
#     # def _validate(self):
#     #     for
#
#     def requirements(self):
#         return {**self.config.event_handlers, **self.config.plan_handlers}
#
#
#     @staticmethod
#     def user_check_config():
#         if not input('Continue? (y/n) --').lower() in ('y', 'yes', 'ok'):
#             quit()




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

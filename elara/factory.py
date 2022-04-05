from typing import Dict, List, Union, Optional
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import os
import json
import logging
from matplotlib.figure import Figure

from elara.helpers import camel_to_snake


class Tool:
    """
    Base Tool class, defines basic behaviour for all tools at each Workstation.
    Values:
        .requirements: list, of requirements that that must be provided by suppliers.
        .options_enabled: bool, for if requirements should carry through manager options.
        .valid_modes/.invalid_modes: Optional lists for option validation at init.
        .resources: dict, of supplier resources collected with .build() method.

    Methods:
        .__init__(): instantiate and validate option.
        .get_requirements(): return tool requirements.
        .build(): check if requirements are met then build.
    """
    requirements = []
    options_enabled = False

    mode = "all"
    groupby_person_attribute = None
    destination_activity_filters = None
    kwargs = None

    valid_modes = None
    invalid_modes = None

    resources = {}

    def __init__(
            self, config,
            mode: Union[None, str] = 'all',
            groupby_person_attribute: Union[None, str] = None,
            compression: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Initiate a tool instance with optional option (ie: 'bus').
        :param mode: optional mode, typically assumed to be str
        :param groupby_person_attribute: optional key for person attribute, str
        :param compression: compression used for output (csv) files. Can be one of: 'infer', 'gzip', 'zip', 'bz2', 'zstd', None
        """
        self.config = config
        self.mode = self._validate_mode(mode)
        self.groupby_person_attribute = groupby_person_attribute
        self.compression = self._validate_compression_method(compression)
        self.kwargs = kwargs

    def __str__(self):
        if self.mode:
            return self.__class__.__name__.split(".")[-1] + self.mode.title()
        return self.__class__.__name__.split(".")[-1]

    @property
    def name(self):
        class_name = self.__class__.__name__.split('.')[-1]
        suffix = ""
        if self.mode:
            suffix += f"_{self.mode}"
        if self.kwargs:  # add other options to ensure unique name
            for value in self.kwargs.values():
                if isinstance(value, str) and os.path.isfile(value):  # if arg is a path then get file name minus extension
                    # value = os.path.basename(value).split(".")[0]
                    continue
                if not isinstance(value, list):
                    value = [value]
                for v in value:
                    suffix += f"_{v}"
        return f"{camel_to_snake(class_name)}{suffix}"

    def get_requirements(self) -> Union[None, Dict[str, list]]:
        """
        Default return requirements of tool for given .option.
        Returns None if .option is None.
        :return: dict of requirements
        """
        self.logger.debug(f'getting requirements for {self.__str__()}')

        if not self.requirements:
            return None

        if self.options_enabled:
            requirements = {}
            for req in self.requirements:
                requirements[req] = {
                    'modes': [self.mode],
                    'groupby_person_attributes': [self.groupby_person_attribute],
                }
                requirements[req].update(self.kwargs)
                requirements[req].pop("name", None)
                requirements[req].pop("benchmark_data_path", None)
        else:
            requirements = {req: {"modes": ["all"]} for req in self.requirements}
            # requirements = {req: None for req in self.requirements}
        return requirements

    def build(
            self,
            resource: Dict[str, list],
            write_path: Optional[str] = None
    ) -> None:
        """
        Default build self.
        :param resource: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        self.logger.debug(f'Building Tool {self.__str__()}')
        self.logger.debug(f'Resources handed to {self.__str__()} = {resource}')
        self.resources = resource

    def _validate_mode(self, mode: str) -> str:
        """
        Validate option based on .valid_modes and .invalid_mode if not None.
        Raises UserWarning if option is not in .valid_modes or in .invalid_modes.
        :param option: str
        :return: str
        """
        if self.valid_modes:
            if mode not in self.valid_modes:
                raise UserWarning(f'Unsupported mode option: {mode} at tool: {self}')
        if self.invalid_modes:
            if mode in self.invalid_modes:
                raise UserWarning(f'Invalid mode option: {mode} at tool: {self}')
        return mode

    def _validate_compression_method(self, compression: str) -> str:
        """
        Validate output file compression method.
        :param compression: compression method used by the pandas.to_csv method.
        :return: str
        """
        valid_compression_methods = [None, 'bz2', 'gzip', 'xz', 'zip']
        if compression not in valid_compression_methods:
            raise ValueError(f'Unsupported compression method: {compression} at tool: {self}')
        return compression

    def start_csv_chunk_writer(self, csv_name: str, write_path=None, compression=None):
        """
        Return a simple csv ChunkWriter, default to config path if write_path (used for testing)
        not given.
        """
        if write_path:
            path = os.path.join(write_path, csv_name)
        else:
            path = os.path.join(self.config.output_path, csv_name)

        if compression is not None:
            path = path_compressed(path, compression)

        return CSVChunkWriter(path, compression)

    def start_arrow_chunk_writer(self, file_name: str, write_path=None):
        """
        Return a simple arrow ChunkWriter, default to config path if write_path (used for testing)
        not given.
        """
        if write_path:
            path = os.path.join(write_path, file_name)
        else:
            path = os.path.join(self.config.output_path, file_name)

        return ArrowChunkWriter(path)

    def write_csv(
            self,
            write_object: Union[pd.DataFrame, gpd.GeoDataFrame],
            csv_name: str,
            compression = None,
            write_path=None
    ):
        """
        Simple write to csv, default to config path if write_path (used for testing) not given.
        """
        if compression:
            csv_name = path_compressed(csv_name, compression)

        if write_path:
            csv_path = os.path.join(write_path, csv_name)
            self.logger.warning(f'path overwritten to {csv_path}')
        else:
            csv_path = os.path.join(self.config.output_path, csv_name)
            self.logger.debug(f'writing to {csv_path}')

        # File exports
        if isinstance(write_object, gpd.GeoDataFrame):
            write_object.drop("geometry", axis=1).to_csv(csv_path, header=True)

        elif isinstance(write_object, (pd.DataFrame, pd.Series)):
            write_object.to_csv(csv_path, header=True)

        else:
            raise TypeError(f"don't know how to write object of type {type(write_object)} to csv")

    def write_geojson(
            self,
            write_object: gpd.GeoDataFrame,
            name: str,
            write_path=None
    ):
        """
        Simple write to geojson, default to config path if write_path (used for testing) not given.
        """

        if write_path:
            path = os.path.join(write_path, name)
            self.logger.warning(f'path overwritten to {path}')
        else:
            path = os.path.join(self.config.output_path, name)
            self.logger.debug(f'writing to {path}')

        # File exports
        if isinstance(write_object, gpd.GeoDataFrame):
            with open(path, "w") as file:
                file.write(write_object.to_json())

        else:
            raise TypeError(
                f"don't know how to write object of type {type(write_object)} to geojson"
            )

    def write_json(
            self,
            write_object: dict,
            name: str,
            write_path=None
    ):
        """
        Simple write to json, default to config path if write_path (used for testing) not given.
        """

        if write_path:
            path = os.path.join(write_path, name)
            self.logger.warning(f'path overwritten to {path}')
        else:
            path = os.path.join(self.config.output_path, name)
            self.logger.debug(f'writing to {path}')

        # File exports
        if isinstance(write_object, dict):
            with open(path, 'w') as outfile:
                json.dump(write_object, outfile)

        else:
            raise TypeError(
                f"don't know how to write object of type {type(write_object)} to json"
            )

    def write_png(
        self,
        write_object: Figure,
        name: str,
        write_path=None
    ):
        """
        Simple write to imaage (png), default to config path if write_path (used for testing) not given.
        """

        if write_path:
            path = os.path.join(write_path, name)
            self.logger.warning(f'path overwritten to {path}')
        else:
            path = os.path.join(self.config.output_path, name)
            self.logger.debug(f'writing to {path}')

        # File exports
        if isinstance(write_object, Figure):
            write_object.savefig(path)

        else:
            raise TypeError(
                f"don't know how to write object of type {type(write_object)} to png"
            )


class WorkStation:
    """
    Base Class for WorkStations.

    Values:
        .depth: int, depth of workstation in longest path search, used for ordering graph operation.
        .tools: dict, of available tools.
    """
    depth = 0
    tools = {}

    def __init__(self, config) -> None:
        """
        Instantiate WorkStation.
        :param config: Config object
        """
        self.config = config

        self.resources = {}
        self.requirements = {}
        self.managers = None
        self.suppliers = None
        self.supplier_resources = {}
        self.logger = logging.getLogger(__name__)

    # def __str__(self):
    #     return f'{self.__class__.__name__}'

    def __str__(self):
        return self.__class__.__name__.split(".")[-1]

    @property
    def name(self):
        class_name = self.__class__.__name__.split('.')[-1]
        return camel_to_snake(class_name)

    def connect(
            self,
            managers: Union[None, list],
            suppliers: Union[None, list]
    ) -> None:
        """
        Connect workstations to their respective managers and suppliers to form a DAG.
        Note that arguments should be provided as lists.
        :param managers: list of managers
        :param suppliers: list of suppliers
        :return: None
        """
        self.managers = managers
        self.suppliers = suppliers

    def display_string(self):
        managers, suppliers, tools = ["-None-"], ["-None-"], ["-None-"]
        if self.managers:
            managers = [str(m) for m in list(self.managers)]
        if self.suppliers:
            suppliers = [str(s) for s in list(self.suppliers)]
        if self.tools:
            tools = list(self.tools)

        self.logger.info(f' *** Created {self}')
        for manager in managers:
            self.logger.debug(f'   * Manager: {manager}')
        for supplier in suppliers:
            self.logger.debug(f'   * Supplier: {supplier}')
        for tool in tools:
            self.logger.debug(f'   * Tooling: {tool}')

    def engage(self) -> None:
        """
        Engage workstation, initiating required tools and getting their requirements.
        Note that initiated tools are mapped in .resources.
        Note that requirements are mapped as .requirements.
        Note that tool order is preserves as defined in Workstation.tools.
        :return: None
        """

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

                    if len(manager_requirement.split("--")) > 1:
                        manager_requirement = manager_requirement.split("--")[0]

                    if manager_requirement == tool_name:
                        if options:
                            # split options between modes and optional arguments
                            modes = options.get("modes")
                            groupby_person_attributes = options.get("groupby_person_attributes")

                            if len(groupby_person_attributes) > 1 and None in groupby_person_attributes:
                                # then can remove None as it exits as a default in all cases
                                groupby_person_attributes.remove(None)

                            optional_args = {
                                key : options[key] for key in options if key not in ["modes", "groupby_person_attributes"]
                                }

                            optional_arg_values_string = ":".join([str(o) for o in (optional_args.values())])

                            for mode in modes:
                                for groupby_person_attribute in groupby_person_attributes:
                                    # build unique key for tool initiated with option
                                    if mode is None and groupby_person_attribute is None:
                                        key = tool_name
                                    else:
                                        key = f"{tool_name}:{mode}:{groupby_person_attribute}:{optional_arg_values_string}"

                                    self.resources[key] = tool(
                                        config=self.config,
                                        mode=mode,
                                        groupby_person_attribute=groupby_person_attribute,
                                        **optional_args
                                        )

                                    tool_requirements = self.resources[key].get_requirements()
                                    all_requirements.append(tool_requirements)
                        else:
                            # init
                            key = str(tool_name)
                            self.resources[key] = tool(self.config)
                            tool_requirements = self.resources[key].get_requirements()
                            all_requirements.append(tool_requirements)

            self.requirements = complex_combine_reqs(all_requirements)

            # Clean out unsupported options for tools that don't support options
            # todo: this sucks...
            # better catch option enabled tools at init but requires looking forward at suppliers
            #  before they are engaged or validated...
            if self.requirements:
                for req, options in self.requirements.items():
                    if self.suppliers and options:
                        for supplier in self.suppliers:
                            if supplier.tools:
                                for name, tool in supplier.tools.items():
                                    if req == name and not tool.options_enabled:
                                        self.requirements[name] = []

    def validate_suppliers(self) -> None:
        """
        Collects available tools from supplier workstations. Raises ValueError if suppliers have
        missing tools.
        :return: None
        """

        # gather supplier tools
        supplier_tools = {}
        if self.suppliers:
            for supplier in self.suppliers:
                if not supplier.tools:
                    continue
                supplier_tools.update(supplier.tools)

        # clean requirments
        clean_requirements = [r.split("--")[0] for r in self.requirements]

        # check for missing requirements
        missing = set(clean_requirements) - set(supplier_tools)
        missing_names = [str(m) for m in missing]
        if missing:
            supplier_resources_string = " & ".join([f"{s} (has available tools: {list(s.tools)})" for s in self.suppliers])
            raise ValueError(
                f'{self} workstation cannot find some requirements: {missing_names} from suppliers: {supplier_resources_string}.'
            )

    def gather_manager_requirements(self) -> Dict[str, List[str]]:
        """
        Gather manager requirements.
        :return: dict of manager reqs, eg {a: [1,2], b:[1]}
        """
        self.logger.debug("Gathering manager requirements ")
        reqs = []

        if self.managers:
            for manager in self.managers:
                reqs.append(manager.requirements)

        return complex_combine_reqs(reqs)

    def build(self, write_path=None):
        """
        Gather resources from suppliers for current workstation and build() all resources in
        order of .resources map.
        :param write_path: Optional output path overwrite
        :return: None
        """
        self.logger.debug(f'Building Workstation {self.__str__()}')

        # gather resources
        if self.suppliers:
            for supplier in self.suppliers:
                self.supplier_resources.update(supplier.resources)

        if self.resources:
            for tool_name, tool in self.resources.items():

                tool.build(self.supplier_resources, write_path)

    def load_all_tools(self, mode=None, groupby_person_attribute=None) -> None:
        """
        Method used for testing.
        Load all available tools into resources with given option.
        :param : mode, default None, must be valid for tools
        :param : groupby_person_attribute, default None, must be valid for tools
        :return: None
        """
        self.logger.info(f"Loading all tools for {self.__str__()}")
        for name, tool in self.tools.items():
            if tool.invalid_modes and mode in tool.invalid_modes:
                continue
            if mode is None and tool.valid_modes is not None:
                mode = tool.valid_modes[0]
            self.resources[name] = tool(self.config, mode=mode, groupby_person_attribute=groupby_person_attribute)

    def write_csv(
            self,
            write_object: Union[pd.DataFrame, gpd.GeoDataFrame],
            csv_name: str,
            write_path=None,
            compression = None,
    ):
        """
        Simple write to csv, default to config path if write_path (used for testing) not given.
        """

        if compression:
            csv_name = path_compressed(csv_name, compression)

        if write_path:
            csv_path = os.path.join(write_path, csv_name)
            self.logger.warning(f'path overwritten to {write_path}')
        else:
            csv_path = os.path.join(self.config.output_path, csv_name)
            self.logger.debug(f'writing to {csv_path}')

        # File exports
        if isinstance(write_object, gpd.GeoDataFrame):
            write_object.drop("geometry", axis=1).to_csv(csv_path, header=True)

        elif isinstance(write_object, (pd.DataFrame, pd.Series)):
            write_object.to_csv(csv_path, header=True)

        else:
            raise TypeError(f"don't know how to write object of type {type(write_object)} to csv")

    def write_geojson(
            self,
            write_object: gpd.GeoDataFrame,
            name: str,
            write_path=None
    ):
        """
        Simple write to geojson, default to config path if write_path (used for testing) not given.
        """

        if write_path:
            path = os.path.join(write_path, name)
            self.logger.warning(f'path overwritten to {write_path}')
        else:
            path = os.path.join(self.config.output_path, name)
            self.logger.debug(f'writing to {path}')
        # File exports
        if isinstance(write_object, gpd.GeoDataFrame):
            with open(path, "w") as file:
                file.write(write_object.to_json())

        else:
            raise TypeError(
                f"don't know how to write object of type {type(write_object)} to geojson"
            )

    def write_json(
            self,
            write_object: dict,
            name: str,
            write_path=None
    ):
        """
        Simple write to json, default to config path if write_path (used for testing) not given.
        """

        if write_path:
            path = os.path.join(write_path, name)
            self.logger.warning(f'path overwritten to {write_path}')
        else:
            path = os.path.join(self.config.output_path, name)
            self.logger.debug(f'writing to {path}')

        # File exports
        if isinstance(write_object, dict):
            with open(path, 'w') as outfile:
                json.dump(write_object, outfile)

        else:
            raise TypeError(
                f"don't know how to write object of type {type(write_object)} to json"
            )


class CSVChunkWriter:
    """
    Extend a list of lines (dicts) that are saved to drive once they reach a certain length.
    """

    def __init__(self, path, compression = None, chunksize=1000) -> None:
        self.path = path
        self.compression = compression
        self.chunksize = chunksize

        self.chunk = []
        self.idx = 0

        self.logger = logging.getLogger(__name__)
        self.logger.debug(f'Chunkwriter initiated for {path}, with size {chunksize} lines')

    def add(self, lines: list) -> None:
        """
        Add a list of lines (dicts) to the chunk.
        If chunk exceeds chunksize, then write to disk.
        :param lines: list of dicts
        :return: None
        """
        self.chunk.extend(lines)
        if len(self.chunk) > self.chunksize:
            self.write()

    def write(self) -> None:
        """
        Convert chunk to dataframe and write to disk.
        :return: None
        """
        chunk_df = pd.DataFrame(self.chunk, index=range(self.idx, self.idx + len(self.chunk)))
        if not self.idx:
            chunk_df.to_csv(self.path, compression=self.compression)
            self.idx += len(self.chunk)
        else:
            chunk_df.to_csv(self.path, header=None, mode="a", compression=self.compression)
            self.idx += len(self.chunk)
        del chunk_df
        self.chunk = []

    def finish(self) -> None:
        self.write()
        self.logger.info(f'Chunkwriter finished for {self.path}')

    def __len__(self):
        return self.idx + len(self.chunk)

class ArrowChunkWriter:
    """
    Extend a list of lines (dicts) that are saved to drive once they reach a certain length.
    """

    def __init__(self, path, chunksize=1000) -> None:
        self.path = path
        self.chunksize = chunksize
        self.writer = None

        self.chunk = []
        self.idx = 0

        self.logger = logging.getLogger(__name__)
        self.logger.debug(f'Chunkwriter initiated for {path}, with size {chunksize} lines')


    def add(self, lines: list) -> None:
        """
        Add a list of lines (dicts) to the chunk.
        If chunk exceeds chunksize, then write to disk.
        :param lines: list of dicts
        :return: None
        """
        self.chunk.extend(lines)
        if len(self.chunk) >= self.chunksize:
            self.write()

    def write(self) -> None:
        """
        Convert chunk to dataframe and write to arrow.
        """
        table = pa.Table.from_pandas(pd.DataFrame(self.chunk))  # TODO loose the pandas intermediate
        if not self.idx:
            self.writer = pa.ipc.RecordBatchStreamWriter(self.path, table.schema)
            self.writer.write(table)
            self.idx += len(self.chunk)
        else:
            self.writer.write(table)
            self.idx += len(self.chunk)
        del table
        self.chunk = []

    def finish(self) -> None:
        self.write()
        self.logger.info(f'Chunkwriter finished for {self.path}')
        self.writer.close()

    def __len__(self):
        return self.idx + len(self.chunk)


def build(start_node: WorkStation, write_path=None) -> list:
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
    :param write_path: Optional output path overwrite
    :return: list, sequence of visits for stages 2 (initiation and validation) and 3 (building)
    """
    logger = logging.getLogger(__name__)

    # stage 1:
    logger.debug(f'Starting DAG')

    if is_cyclic(start_node):
        raise UserWarning(f"Cyclic dependency found at {is_cyclic(start_node)}")
    if is_broken(start_node):
        raise UserWarning(f"Broken dependency found at {is_broken(start_node)}")

    build_graph_depth(start_node)

    display_graph(start_node)

    logger.debug(f'DAG prepared')

    # stage 2:
    logger.debug(f'Initiating DAG')

    visited = list()
    queue = list()
    queue.append(start_node)
    visited.append(start_node)

    while queue:
        current = queue.pop(0)
        current.engage()

        if current.suppliers:
            current.validate_suppliers()

            for supplier in order_by_distance(current.suppliers):
                if supplier not in visited:
                    queue.append(supplier)
                    visited.append(supplier)

    logger.info(f'All Workstations Initiated and Validated')

    # stage 3:
    logger.info(f'Initiating Build')
    sequence = visited
    return_queue = visited[::-1]
    visited = []
    while return_queue:
        current = return_queue.pop(0)
        current.build(write_path=write_path)
        visited.append(current)

    # return full sequence for testing
    return sequence + visited


def is_cyclic(start):
    """
    Return WorkStation if the directed graph starting at WorkStation has a cycle.
    :param start: starting WorkStation
    :return: WorkStation
    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        if vertex.suppliers:
            for supplier in vertex.suppliers:
                if supplier in path or visit(supplier):
                    return supplier
        path.remove(vertex)

    return visit(start)


def is_broken(start):
    """
    Return WorkStation if directed graph starting at WorkStation has broken connection,
    ie a supplier who does not have the correct manager in .managers.
    :param start: starting WorkStation
    :return: WorkStation
    """

    visited = set()

    def broken_link(manager, supplier):
        if not supplier.managers:
            return True
        if manager not in supplier.managers:
            return True

    def visit(vertex):
        visited.add(vertex)
        if vertex.suppliers:

            for supplier in vertex.suppliers:
                if supplier in visited:
                    continue
                if broken_link(vertex, supplier) or visit(supplier):
                    return supplier

    return visit(start)


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


def display_graph(node: WorkStation) -> None:
    """
    Function to depth first traverse graph from start vertex, displaying vertex connections
    :param node: starting Workstation
    :return: None
    """

    visited = set()

    def visit(vertex):
        vertex.display_string()
        visited.add(vertex)
        if vertex.suppliers:
            for supplier in vertex.suppliers:
                if supplier not in visited:
                    visit(supplier)

    visit(node)


def order_by_distance(candidates: list) -> list:
    """
    Returns candidate list ordered by .depth.
    :param candidates: list, of workstations
    :return: list, of workstations
    """
    return sorted(candidates, key=lambda x: x.depth, reverse=False)


def complex_combine_reqs(reqs: List[dict]) -> Dict[str, list]:

    if not reqs:
        return {}

    tool_set = set()
    for req in reqs:
        if req:
            tool_set |= set(req)

    combined_reqs = {}
    for tool in tool_set:
        combined_reqs[tool] = {}
        # collect unique mode dependencies
        modes = set()
        groupby_person_attributes = set()
        kwargs = {}
        for req in reqs:
            if req and req.get(tool):
                combined_reqs[tool] = req.get(tool)

            if req is None:
                req = {}

            tool_reqs = req.get(tool, {})
            if not tool_reqs:
                tool_reqs = {}  # TODO bit hacky, better to get handlers without requirments to return {}
            if tool_reqs.get("modes"):
                modes |= set(tool_reqs.get("modes"))
            if tool_reqs.get("groupby_person_attributes"):
                groupby_person_attributes |= set(tool_reqs.get("groupby_person_attributes"))
            remaining_args = {k:v for k,v in tool_reqs.items() if k not in ["modes","groupby_person_attributes"]}
            kwargs.update(remaining_args)

        if not modes:
            modes = {None}
        if not groupby_person_attributes:
            groupby_person_attributes = {None}
        combined_reqs[tool]['modes'] = modes
        combined_reqs[tool]['groupby_person_attributes'] = groupby_person_attributes
        combined_reqs[tool].update(kwargs)

    return combined_reqs


def combine_reqs(reqs: List[dict]) -> Dict[str, list]:
    """
    Helper function for combining lists of requirements (dicts of lists) into a single
    requirements dict:

    [{req1:[a], req2:[b]}, {req1:[a], req2:[a], req3:None}] -> {req1:[a,b], req2:[a,b], req3:None}

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
        combined_reqs[tool] = {}
        if tool_set:
            # collect unique mode dependencies
            modes = set()
            for req in reqs:
                if req and req.get(tool):
                    modes.update(req[tool]['modes'])

            if modes:
                for req in reqs:
                    # keep all arguments of current (in the loop) tool, but only pass "modes" argument to dependencies
                    if req and req.get(tool):
                        combined_reqs[tool] = req.get(tool)
                combined_reqs[tool]['modes'] = modes
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

    if l2 is None:
        if l1 is None:
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

def path_compressed(path:str, compression:str)->str:
    """
    Add an appropriate suffix to a compressed file path.
    :param path: Csv output filepath
    :param compression: Compression type
    """
    if compression == 'gzip':
        path = f'{path}.gz'
    else:
        path = f'{path}.{compression}'

    return path
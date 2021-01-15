import sys
import os
import pytest
import logging


sys.path.append(os.path.abspath('../elara'))
from elara.factory import WorkStation, Tool, equals, build, build_graph_depth

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


class Config:
    verbose = True

    def __init__(self):
        pass

    def get_requirements(self):
        return {'volume_counts': ['car'], 'vkt': ['bus'], 'mode_share': ['all']}


class ExampleTool(Tool):
    options_enbled = False
    logger = logging.getLogger(__name__)
    pass


def test_tool_naming():
    tool = ExampleTool(Config())
    assert (str(tool)) == "ExampleTool"
    assert tool.name == "example_tool"
    tool = ExampleTool(config=Config(), option="car")
    assert (str(tool)) == "ExampleToolCar"
    assert tool.name == "example_tool_car"


# Tools
class VKT(ExampleTool):
    options_enabled = True
    requirements = ['volume_counts']
    valid_options = ['car', 'bus']


class VolumeCounts(ExampleTool):
    options_enabled = True
    requirements = ['network', 'events']
    valid_options = ['car', 'bus']

    def get_requirements(self):
        return {req: None for req in self.requirements}


class ModeShare(ExampleTool):
    options_enabled = True
    requirements = ['network', 'plans']
    valid_options = ['all']

    def get_requirements(self):
        return {req: None for req in self.requirements}


class Network(ExampleTool):
    requirements = ['network_path']


class Events(ExampleTool):
    requirements = ['events_path']


class Plans(ExampleTool):
    requirements = ['plans_path']


class GetPath(ExampleTool):
    requirements = None


# Work stations

class StartProcess(WorkStation):
    tools = None
    logger = logging.getLogger(__name__)

    def gather_manager_requirements(self):

        return self.config.get_requirements()


class PostProcess(WorkStation):
    tools = {
        'vkt': VKT,
    }
    logger = logging.getLogger(__name__)


class EventHandlerProcess(WorkStation):
    tools = {
        'volume_counts': VolumeCounts,
    }
    logger = logging.getLogger(__name__)

class PlanHandlerProcess(WorkStation):
    tools = {
        'mode_share': ModeShare,
    }
    logger = logging.getLogger(__name__)


class InputProcess(WorkStation):
    tools = {
        'events': Events,
        'plans': Plans,
        'network': Network
    }
    logger = logging.getLogger(__name__)


class PathProcess(WorkStation):
    tools = {
        'network_path': GetPath,
        'events_path': GetPath,
        'plans_path': GetPath,
    }
    logger = logging.getLogger(__name__)


config = Config()


@pytest.fixture
def start():
    return StartProcess(config)


@pytest.fixture
def post_process():
    return PostProcess(config)


@pytest.fixture
def event_handler_process():
    return EventHandlerProcess(config)


@pytest.fixture
def plan_handler_process():
    return PlanHandlerProcess(config)


@pytest.fixture
def inputs_process():
    return InputProcess(config)


@pytest.fixture
def config_paths():
    return PathProcess(config)


def test_pipeline_connection(
        start,
        post_process,
        event_handler_process,
        plan_handler_process,
        inputs_process,
        config_paths
):
    start.connect(None, [post_process, event_handler_process, plan_handler_process])
    post_process.connect([start], [event_handler_process, plan_handler_process])
    event_handler_process.connect([start, post_process], [inputs_process])
    plan_handler_process.connect([start, post_process], [inputs_process])
    inputs_process.connect([event_handler_process, plan_handler_process], [config_paths])
    config_paths.connect([inputs_process], None)
    assert event_handler_process.managers == [start, post_process]


def test_requirements(
        start,
        post_process,
        event_handler_process,
        plan_handler_process,
        inputs_process,
        config_paths
):
    start.connect(None, [post_process, event_handler_process, plan_handler_process])
    post_process.connect([start], [event_handler_process, plan_handler_process])
    event_handler_process.connect([start, post_process], [inputs_process])
    plan_handler_process.connect([start, post_process], [inputs_process])
    inputs_process.connect([event_handler_process, plan_handler_process], [config_paths])
    config_paths.connect([inputs_process], None)

    build(start)

    assert equals(
        start.gather_manager_requirements(),
        {'volume_counts': ['car'], 'vkt': ['bus'], 'mode_share': ['all']}
    )
    assert equals(
        post_process.gather_manager_requirements(),
        {'volume_counts': ['car'], 'vkt': ['bus'], 'mode_share': ['all']}
    )
    assert equals(
        event_handler_process.gather_manager_requirements(),
        {'vkt': ['bus'], 'volume_counts': ['car', 'bus'], 'mode_share': ['all']}
    )
    assert equals(
        plan_handler_process.gather_manager_requirements(),
        {'vkt': ['bus'], 'volume_counts': ['car', 'bus'], 'mode_share': ['all']}
    )
    assert equals(
        inputs_process.gather_manager_requirements(),
        {'events': None, 'network': None, 'plans': None}
    )
    assert equals(
        config_paths.gather_manager_requirements(),
        {'events_path': None, 'network_path': None, 'plans_path': None}
    )


def test_engage_start_suppliers(
        start,
        post_process,
        event_handler_process,
        plan_handler_process,
        inputs_process,
        config_paths
):
    start.connect(None, [post_process, event_handler_process, plan_handler_process])
    post_process.connect([start], [event_handler_process, plan_handler_process])
    event_handler_process.connect([start, post_process], [inputs_process])
    plan_handler_process.connect([start, post_process], [inputs_process])
    inputs_process.connect([event_handler_process, plan_handler_process], [config_paths])
    config_paths.connect([inputs_process], None)

    start.engage()
    assert set(start.resources) == set()
    post_process.engage()
    assert set(post_process.resources) == {'vkt:bus'}
    event_handler_process.engage()
    assert set(event_handler_process.resources) == {'volume_counts:car', 'volume_counts:bus'}
    plan_handler_process.engage()
    assert set(plan_handler_process.resources) == {'mode_share:all'}
    inputs_process.engage()
    assert set(inputs_process.resources) == {'events', 'network', 'plans'}


def test_dfs_distances(
        start,
        post_process,
        event_handler_process,
        plan_handler_process,
        inputs_process,
        config_paths
):
    start.connect(None, [post_process, event_handler_process, plan_handler_process])
    post_process.connect([start], [event_handler_process, plan_handler_process])
    event_handler_process.connect([start, post_process], [inputs_process])
    plan_handler_process.connect([start, post_process], [inputs_process])
    inputs_process.connect([event_handler_process, plan_handler_process], [config_paths])
    config_paths.connect([inputs_process], None)

    build_graph_depth(start)

    assert start.depth == 0
    assert post_process.depth == 1
    assert event_handler_process.depth == 2
    assert plan_handler_process.depth == 2
    assert inputs_process.depth == 3
    assert config_paths.depth == 4


def test_bfs(
        start,
        post_process,
        event_handler_process,
        plan_handler_process,
        inputs_process,
        config_paths
):
    start.connect(None, [post_process, event_handler_process, plan_handler_process])
    post_process.connect([start], [event_handler_process, plan_handler_process])
    event_handler_process.connect([start, post_process], [inputs_process])
    plan_handler_process.connect([start, post_process], [inputs_process])
    inputs_process.connect([event_handler_process, plan_handler_process], [config_paths])
    config_paths.connect([inputs_process], None)

    sequence = build(start)
    assert sequence == [start, post_process, event_handler_process,
                        plan_handler_process, inputs_process, config_paths,
                        config_paths, inputs_process, plan_handler_process,
                        event_handler_process, post_process, start]


def test_engage_supply_chain(
        start,
        post_process,
        event_handler_process,
        plan_handler_process,
        inputs_process,
        config_paths
):
    start.connect(None, [post_process, event_handler_process, plan_handler_process])
    post_process.connect([start], [event_handler_process, plan_handler_process])
    event_handler_process.connect([start, post_process], [inputs_process])
    plan_handler_process.connect([start, post_process], [inputs_process])
    inputs_process.connect([event_handler_process, plan_handler_process], [config_paths])
    config_paths.connect([inputs_process], None)

    build(start)

    assert set(start.resources) == set()
    assert set(post_process.resources) == {'vkt:bus'}
    assert set(event_handler_process.resources) == {'volume_counts:car', 'volume_counts:bus'}
    assert set(plan_handler_process.resources) == {'mode_share:all'}
    assert set(inputs_process.resources) == {'events', 'network', 'plans'}
    assert set(config_paths.resources) == {'events_path', 'network_path', 'plans_path'}

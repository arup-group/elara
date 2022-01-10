import sys
import os
from attr import attributes
import pytest
import logging


sys.path.append(os.path.abspath('../elara'))
from elara.factory import WorkStation, Tool, complex_combine_reqs, equals, build, build_graph_depth, combine_reqs, complex_combine_reqs

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


@pytest.mark.parametrize(
    "A,B,expected",
    [
        ({}, {}, True),
        ({'req1': {'modes': {'a'}}}, {'req1': {'modes': {'a'}}}, True),
        ({'req1': {'modes': {'a', 'b'}}}, {'req1': {'modes': {'b', 'a'}}}, True),
        ({'req1': {'modes': {'a', 'b'}}}, {'req1': {'modes': {'a'}}}, False),
        ({'req1': {'modes': {'a', 'b'}}}, {'req2': {'modes': {'b', 'a'}}},False),
    ]
)
def test_equals(A, B, expected):
    assert (A == B) == expected


@pytest.mark.parametrize(
    "test,expected",
    [
        ([], {}),
        ([{'req1': {'modes': ['a']}}, {'req1': {'modes': ['b']}}], {"req1": {"modes": {'a', 'b'}}}),
        ([{'req1': {'modes': ['a']}}, {'req1': {'modes': ['a']}}], {"req1": {"modes": {'a'}}}),
        ([{'req1': {'modes': ['a']}}, {'req2': {'modes': ['a']}}], {"req1": {"modes": {'a'}}, "req2": {"modes": {'a'}}}),
        ([{'req1': {'modes': [None]}}, {'req1': {'modes': [None]}}], {"req1": {"modes": {None}}}),
        ([{'req1': {'modes': [None]}}, {'req1': {'modes': ['b']}}], {"req1": {"modes": {None,'b'}}}),
        ([{'req1': {'modes': []}}, {'req1': {'modes': []}}], {"req1": None}),
    ]
)
def test_combine_reqs(test, expected):
    assert combine_reqs(test) == expected


@pytest.mark.parametrize(
    "test,expected",
    [
        ([], {}),
        ([{'req1': {'modes': ['a']}}, {'req1': {'modes': ['b']}}], {"req1": {"modes": {'a', 'b'}, 'groupby_person_attributes': {None}}}),
        ([{'req1': {'modes': ['a'], 'groupby_person_attributes': ['A']}}, {'req1': {'modes': ['b'], 'groupby_person_attributes': ['B']}}], {"req1": {"modes": {'a', 'b'}, 'groupby_person_attributes': {'A', 'B'}}}),
        ([{'req1': {'modes': ['a'], 'groupby_person_attributes': ['A']}}, {'req1': {'modes': ['a'], 'groupby_person_attributes': ['A']}}], {"req1": {"modes": {'a'}, 'groupby_person_attributes': {'A'}}}),
        ([{'req1': {'modes': ['a'], 'groupby_person_attributes': ['A']}}, {'req2': {'modes': ['a'], 'groupby_person_attributes': ['A']}}], {"req1": {"modes": {'a'}, 'groupby_person_attributes': {'A'}}, "req2": {"modes": {'a'}, 'groupby_person_attributes': {'A'}}}),
        ([{'req1': {'modes': [None], 'groupby_person_attributes': [None]}}, {'req1': {'modes': [None], 'groupby_person_attributes':[None]}}], {"req1": {"modes": {None}, 'groupby_person_attributes': {None}}}),
        ([{'req1': {'modes': [None], 'groupby_person_attributes': ['A']}}, {'req1': {'modes': ['b'], 'groupby_person_attributes': ['A']}}], {"req1": {"modes": {None,'b'}, 'groupby_person_attributes': {'A'}}}),
        ([{'req1': {'modes': ['a'], 'groupby_person_attributes': [None]}}, {'req1': {'modes': ['b'], 'groupby_person_attributes': ['B']}}], {"req1": {"modes": {'a','b'}, 'groupby_person_attributes': {None, 'B'}}}),
        ([{'req1': {'modes': [], 'groupby_person_attributes': []}}, {'req1': {'modes': [], 'groupby_person_attributes': []}}], {"req1": {"modes": {None}, 'groupby_person_attributes': {None}}}),
        ([{'req1': None}, {'req1': None}], {"req1": {"modes": {None}, 'groupby_person_attributes': {None}}}),
    ]
)
def test_complex_combine_reqs(test, expected):
    assert complex_combine_reqs(test) == expected


class Config():
    verbose = True

    def __init__(self):
        pass

    def get_requirements(self):
        return {'volume_counts': {'modes':{'car'}}, 'vkt': {'modes':{'bus'}}, 'mode_share': {'modes':{'all'}, 'groupby_person_attributes': {'region'}}}


class ExampleTool(Tool):
    options_enabled = False
    logger = logging.getLogger(__name__)
    pass


def test_tool_naming():
    tool = ExampleTool(Config())
    assert (str(tool)) == "ExampleToolAll"
    assert tool.name == "example_tool_all"
    tool = ExampleTool(config=Config(), mode="car")
    assert (str(tool)) == "ExampleToolCar"
    assert tool.name == "example_tool_car"
    # TODO test naming with other args


# Tools
class VKT(ExampleTool):
    options_enabled = True
    requirements = ['volume_counts']
    valid_modes = ['car', 'bus']


class VolumeCounts(ExampleTool):
    options_enabled = True
    requirements = ['network', 'events']
    valid_modes = ['car', 'bus']

    def get_requirements(self):
        return {req: None for req in self.requirements}


class ModeShare(ExampleTool):
    options_enabled = True
    requirements = ['network', 'plans']
    valid_modes = ['all']

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
        {'volume_counts': {'modes':{'car'}}, 'vkt': {'modes':{'bus'}}, 'mode_share': {'modes':{'all'}, 'groupby_person_attributes': {'region'}}}
    )
    assert equals(
        post_process.gather_manager_requirements(),
        {'volume_counts': {'modes':{'car'}, 'groupby_person_attributes': {None}}, 'vkt': {'modes':{'bus'}, 'groupby_person_attributes': {None}}, 'mode_share': {'modes':{'all'}, 'groupby_person_attributes': {'region'}}}
    )
    assert equals(
        event_handler_process.gather_manager_requirements(),
        {'vkt': {'modes':{'bus'}, 'groupby_person_attributes': {None}}, 'volume_counts': {'modes':{'car', 'bus'}, 'groupby_person_attributes': {None}}, 'mode_share': {'modes': {'all'}, 'groupby_person_attributes': {'region'}}}
    )
    assert equals(
        plan_handler_process.gather_manager_requirements(),
        {'vkt': {'modes':{'bus'}, 'groupby_person_attributes': {None}}, 'volume_counts': {'modes': {'car', 'bus'}, 'groupby_person_attributes': {None}}, 'mode_share': {'modes': {'all'}, 'groupby_person_attributes': {'region'}}}
    )
    assert equals(
        inputs_process.gather_manager_requirements(),
        {'events': {'groupby_person_attributes': {None}, 'modes': {None}}, 'network': {'groupby_person_attributes': {None}, 'modes': {None}}, 'plans': {'groupby_person_attributes': {None}, 'modes': {None}}}
    )
    assert equals(  
        config_paths.gather_manager_requirements(),
        {'events_path': {'groupby_person_attributes': {None}, 'modes': {None}}, 'network_path': {'groupby_person_attributes': {None}, 'modes': {None}}, 'plans_path': {'groupby_person_attributes': {None}, 'modes': {None}}}
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
    assert set(post_process.resources) == {'vkt:bus:None:'}
    event_handler_process.engage()
    assert set(event_handler_process.resources) == {'volume_counts:car:None:', 'volume_counts:bus:None:'}
    plan_handler_process.engage()
    assert set(plan_handler_process.resources) == {'mode_share:all:region:'}
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
    assert set(post_process.resources) == {'vkt:bus:None:'}
    assert set(event_handler_process.resources) == {'volume_counts:car:None:', 'volume_counts:bus:None:'}
    assert set(plan_handler_process.resources) == {'mode_share:all:region:'}
    assert set(inputs_process.resources) == {'events', 'network', 'plans'}
    assert set(config_paths.resources) == {'events_path', 'network_path', 'plans_path'}

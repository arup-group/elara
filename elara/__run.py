import elara.main as m

from elara.helpers import PathPath, NaturalOrderGroup
from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.input_plan_handlers import InputPlanHandlerWorkstation
from elara.event_handlers import EventHandlerWorkStation
from elara.postprocessing import PostProcessWorkStation
from elara.benchmarking import BenchmarkWorkStation
from elara import factory

config_path = '/Users/andrew.kay/Documents/_repos/elara/example_configs/benchmarks_io.toml'
config = Config(path=config_path)

config_requirements = RequirementsWorkStation(config)
postprocessing = PostProcessWorkStation(config)
benchmarks = BenchmarkWorkStation(config)
event_handlers = EventHandlerWorkStation(config)
plan_handlers = PlanHandlerWorkStation(config)
input_plan_handlers = InputPlanHandlerWorkstation(config)
input_workstation = InputsWorkStation(config)
paths = PathFinderWorkStation(config)

m.main(config)
# plans_workstation = PlanHandlerWorkStation(config)
# plans_workstation.connect(managers=[config_requirements, benchmarks, postprocessing], suppliers=[input])
# plans_workstation.build()

# input_plans_workstation = InputPlanHandlerWorkstation(config)
# input_plans_workstation.build()

"""
Template Tools for WorkStations:
1) TemplateGetConfig
2) TemplateInput
3) TemplateEventHandler
4) TemplatePlanHandler
5) TemplatePostProcessor
6) TemplateBenchmark
"""

from typing import Optional

from elara.factory import Tool
from elara.event_handlers import EventHandlerTool
from elara.plan_handlers import PlanHandlerTool
from elara.postprocessing import PostProcessor


class TemplateGetConfig(Tool):
    """
    ****************************************************************************
    Template for getting value from CONFIG.
    ****************************************************************************
    """

    path = None

    def build(self, resource: dict, write_path: Optional[str] = None):
        # base tool build
        super().build(resource)
        # retrieve item from config
        self.path = self.config.dummy_path


class TemplateInput(Tool):
    """
    ****************************************************************************
    Template for INPUT data resource.
    ****************************************************************************
    """

    requirements = ['get_from_config_1', 'get_from_config_2']
    data_1 = None
    data_2 = None

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Input object constructor.
        :param resources: dict, resources from suppliers
        :param write_path: Optional output path overwrite
        """

        # build base tool
        super().build(resources)

        # build data
        self.data_1 = 'eg'
        self.data_2 = 'eg'


class TemplateEventHandler(EventHandlerTool):
    """
    ****************************************************************************
    Template for EVENT HANDLER.
    ****************************************************************************
    """

    requirements = [
        'events',
        '1',
        '2',
        '3',
    ]

    def __init__(self, config, mode=None) -> None:
        """
        Initiate class, creates results placeholders.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config, mode)

        self.data1 = None
        self.data2 = None

        # Initialise results storage
        self.result_dfs = dict()  # Result geodataframes ready to export

    def __str__(self):
        return f'<INSERT NAME> mode: {self.mode}'

    def build(self, resources: dict, write_path: Optional[str] = None) -> None:
        """
        Build handler from resources.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """
        super().build(resources)

        # build data

    def process_event(self, elem) -> None:
        """
        Iteratively aggregate 'vehicle enters traffic' and 'vehicle exits traffic'
        events to determine link volume counts.
        :param elem: Event XML element
        """

        # Process event from stream

    def finalise(self) -> None:
        """
        Following event processing, the raw events table will contain counts by link
        by time slice. The only thing left to do is scale by the sample size and
        create dataframes.
        """

        # Populate results_gdfs
        # Save to disk


class TemplatePlanHandler(PlanHandlerTool):
    """
    ****************************************************************************
    Template for PLAN HANDLER
    ****************************************************************************
    """

    requirements = [
        'plans',
        'transit_schedule',
        'attributes',
        'mode_map',
        'mode_hierarchy'
    ]

    # ....AS PER EVENT HANDLER


class TemplatePostProcessor(PostProcessor):
    """
    ****************************************************************************
    Template for PLAN HANDLER
    ****************************************************************************
    """

    requirements = ['typically a handler']
    valid_modes = ['eg', 'car', 'bus', 'train', 'subway', 'ferry']

    def __init__(self, config, mode) -> None:
        """
        Optional pre processing at init (for validation/early fail)
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config, mode)

        # prepare post processor

    def build(self, resources: dict, write_path: Optional[str] = None):
        """
        Build post processed output:
        1. read output from required handler.
        2. and/or consider requirement resources.
        3. Post process
        4. save to disk
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: None
        """


class TemplateBenchmark(Tool):
    """
    ****************************************************************************
    Template for PLAN HANDLER
    Note that there are various types of benchmarks already built and it is likely top be easier
    to use an existing class such as Cordon.
    ****************************************************************************
    """

    data1 = None
    data2 = None
    data3 = None

    param1 = {'in': 1, 'out': 2}
    year = 2016
    param2 = None
    param3 = None

    def __init__(self, config, mode) -> None:
        """
        Cordon parent object used for cordon benchmarks. Initiated with CordonCount
        objects as required.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config, mode)

        # prepare benchmark

    def build(self, resources: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        Collects scoring from CordonCount objects.
        :param resources: dict, supplier resources
        :param write_path: Optional output path overwrite
        :return: Dictionary of scores {'in': float, 'out': float}
        """

        # Build paths and load appropriate data
        # Process benchmarks
        # process scores
        scores = {'in': 1, 'out': 0.5}

        return scores

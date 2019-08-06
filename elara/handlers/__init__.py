from elara.handlers.agent_plan_handlers import *
from elara.handlers.network_event_handlers import *
from elara import HandlerConfigError


class HandlerManager:

    # Dictionary used to map configuration string to handler type
    HANDLER_MAP = {
        "volume_counts": VolumeCounts,
        "passenger_counts": PassengerCounts,
        "stop_interactions": StopInteractions,
        "activities": Activities,
        "legs": Legs,
        "mode_share": ModeShare,
    }

    def __init__(self, config):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        for handler_name, mode_list in {
            **self.config.event_handlers,
            **self.config.plan_handlers
        }.items():
            handler_class = self.HANDLER_MAP.get(handler_name, None)
            if not handler_class:
                raise HandlerConfigError(f'Unknown handler name: {handler_name} found in config')

    @property
    def requirements(self):
        required_feeds = set()
        required_resources = set()
        for handler_name, mode_list in {
            **self.config.event_handlers,
            **self.config.plan_handlers
        }.items():
            handler_class = self.HANDLER_MAP[handler_name]
            required_feeds.add(handler_class.subscription)
            required_resources.update(handler_class.requires)
        return list(required_feeds), list(required_resources)

    def print_requirements(self):
        print('--- Config Summary ---')
        if self.config.event_handlers:
            print('Event Handlers: {}'.format(self.config.event_handlers))
        if self.config.plan_handlers:
            print('Plan Handlers: {}'.format(self.config.plan_handlers))
        if self.config.post_processing:
            print('Post Processors: {}'.format(self.config.post_processing))
        if self.config.benchmarks:
            print('Benchmarks: {}'.format(self.config.benchmarks))


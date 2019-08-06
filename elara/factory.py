from elara.config import Config
from elara.handlers import HandlerManager
from elara.inputs import InputManager


class Supervisor:

    input_priority = [
        'events',
        'plans',
        'network',
        'attributes',
        'transit_schedule',
        'transit_vehicles',
        'mode_hierarchy',
        'mode_map'
    ]

    def __init__(self, config_path):

        self.config = Config(config_path)
        self.handler_manager = HandlerManager(self.config)
        self.input_manager = InputManager(self.config)

        self.feed_list = []
        self.resource_list = []

        self.live_feeds = {}
        self.live_resources = {}

        self.active_event_handlers = []
        self.active_plan_handlers = []

    def validate_config(self):

        self.handler_manager.print_requirements()
        feeds, resources = self.handler_manager.requirements

        print('--- Input Summary ---')
        for required_input in feeds + resources:
            print(required_input)

        self.config.validate_required_paths(feeds + resources)
        self.input_manager.validate_requirements(feeds + resources)

        self.feed_list = feeds
        self.resource_list = resources

    def prepare_feeds(self):
        for feed in self._prioritise(self.feed_list):
            self.live_feeds[feed] = getattr(self.input_manager, feed)

    def prepare_resources(self):
        for resource in self._prioritise(self.resource_list):
            self.live_resources[resource] = getattr(self.input_manager, resource)

    def prepare_handlers(self):
        for handler_name, mode_list in self.config.event_handlers.items():
            for mode in mode_list:
                self.active_event_handlers.append(
                    self.handler_manager.HANDLER_MAP[handler_name](
                        self.live_resources['network'],
                        self.live_resources['transit_schedule'],
                        self.live_resources['transit_vehicles'],
                        self.live_resources['attributes'],
                        mode,
                        self.config.time_periods,
                        self.config.scale_factor,
                    )
                )

    def _prioritise(self, unsorted):
        for p in self.input_priority:
            if p in unsorted:
                yield p

    @staticmethod
    def user_check_config():
        if not input('Continue? (y/n) --').lower() in ('y', 'yes', 'ok'):
            quit()


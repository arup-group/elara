from elara.handlers.plan_handlers import *
from elara.handlers.event_handlers import *


HANDLER_MAP = {
    "volume_counts": VolumeCounts,
    "passenger_counts": PassengerCounts,
    "stop_interactions": StopInteractions,
    "activities": Activities,
    "legs": Legs,
    "mode_share": ModeShare,
}

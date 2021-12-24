from elara.plan_handlers import PlanHandlerWorkStation, TripLogs


class InputTripLogs(TripLogs):
    requirements = ['input_plans', 'transit_schedule', 'attributes']
    valid_modes = ['all']


class InputPlanHandlerWorkstation(PlanHandlerWorkStation):
    """
    Work Station class for collecting and building Input Plans Handlers.
    Same invocation as PlanHandlerWorkstation but with unique toll definitions.
    """

    # dict key to hand to supplier.resources
    # allows handler to be subclassed by overriding
    plans_resource = 'input_plans'

    tools = {
        "input_trip_logs": InputTripLogs
    }
from pandas.core import groupby
import pandas as pd
import os
import numpy as np
from typing import Optional
import logging
import json
from matplotlib import pyplot as plt

from elara.factory import WorkStation, Tool
from elara import get_benchmark_data
from elara.helpers import try_sort_on_numeric_index


class BenchmarkTool(Tool):

    options_enabled = True
    weight = 1
    benchmark_data_path = None
    plot_types = []

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):

        # override default plot type if supplied, remove from kwargs as these don't affect suppliers
        if "plot_types" in kwargs:
            self.plot_types = kwargs.pop("plot_types", [])

        # override default path if supplied, remove from kwargs as these don't affect suppliers
        proposed_bm_path = kwargs.pop("benchmark_data_path", None)
        if not self.benchmark_data_path:
            self.benchmark_data_path = proposed_bm_path

        super().__init__(config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initiating: {str(self)} with name: {self.name}")
        self.logger.debug(f"groupby_person_attribute={groupby_person_attribute}, benchmark_data_path={self.benchmark_data_path}, kwargs={kwargs}")

    def __str__(self):
        return f'{self.__class__.__name__}'


class CsvComparison(BenchmarkTool):

    output_value_fields = ['trips_benchmark', 'trip_simulation']

    def __init__(self, config, mode, groupby_person_attribute=None, **kwargs) -> None:
        """
        Initiate class, checks benchmark data format.
        :param config: Config object
        :param mode: str, mode
        :param attribute: str, atribute key defaults to None
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        if self.unsafe_load:
            self.logger.debug(f"Data path is {self.benchmark_data_path}")
            return None

        self.logger.debug(f"Loading BM data from {self.benchmark_data_path}")
        self.logger.debug(f"Using indices '{self.index_fields}'")
        if self.benchmark_data_path is None:
            return None  # todo this is required for the input plan comparison tools
        if not os.path.exists(self.benchmark_data_path):
            raise UserWarning(f"Unable to find benchmark {self.benchmark_data_path}.")
        benchmarks_df = pd.read_csv(self.benchmark_data_path, index_col=self.index_fields)
        if self.value_field not in benchmarks_df.columns:
            raise UserWarning(f"Incorrectly formatted benchmarks, expected {self.value_field} column.")


    def build(self, resources: dict, write_path: Optional[str] = None) -> dict:
        """
        Compare two csv files (benchmark vs simulation), calculate and plot their differences
        """
        super().build(resources, write_path)

        # Read benchmark and simulation csv files
        self.logger.debug(f"Loading BM data from {self.benchmark_data_path}")
        self.logger.debug(f"Using indices '{self.index_fields}'")
        benchmarks_df = pd.read_csv(self.benchmark_data_path, index_col=self.index_fields)

        simulation_path = os.path.join(self.config.output_path, self.simulation_name)
        self.logger.debug(f"Loading Simulation data from {simulation_path}")
        simulation_df = pd.read_csv(simulation_path, index_col=self.index_fields)

        # compare
        bm_df = pd.concat([benchmarks_df[self.value_field], simulation_df[self.value_field]], axis = 1)
        bm_df.columns = self.output_value_fields
        bm_df.dropna(0, inplace=True)

        self.plot_comparisons(bm_df)

        bm_df['difference'] = bm_df[self.output_value_fields[0]] - bm_df[self.output_value_fields[1]]
        bm_df['abs_difference'] = bm_df.difference.abs()

        # write results
        csv_name = f'{self.name}.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_df, csv_path, write_path=write_path)

        # evaluation metrics
        scores = {
            'mse':np.mean(bm_df['difference'] ** 2), # mean squared error
            'mae': np.mean(bm_df.abs_difference)
            }

        return scores

    def plot_comparisons(self, df):
        for kind in self.plot_types:
            figure = self.plot(df, kind=kind)
            if figure is not None:
                figure.savefig(os.path.join(self.config.output_path,'benchmarks', f'{self.name}_{kind}.png'))

    def plot(self, df:pd.DataFrame, kind:str) -> plt.figure:
        """
        Comparison plot, either bar, line or histograms supported.
        """
        if kind == "hist":
            return df.plot.hist(figsize=(6,4)).get_figure()
        if kind == "bar":
            return self.barline(df, kind="bar")
        if kind == "line":
            return self.barline(df, kind="line")

        self.logger.warning(f"Unknown plot type '{kind}', returning 'None'.")
        return None

    def barline(self, df: pd.DataFrame, kind: str):
        """
        Plot a bar or line figure.
        Can handle multi indices of size 2 using subplots.
        Will attempt a sensible sort of x axis.

        :param pd.DataFrame df: data to plot
        :param str kind: ['bar','line','hist']
        :return plt.Figure: figure
        """
        if isinstance(df.index, pd.MultiIndex):
            if not len(df.index.levels) == 2:
                self.logger.warning(f"{self} cannot handle multi index > 2, returning None.")
                return None
            groups = [(m, g) for m, g in df.groupby(df.index.get_level_values(-1))]
            n = len(groups)
            if n == 1:
                try_sort_on_numeric_index(df)
                fig = df.plot(figsize=(6,4), kind=kind, rot=90).get_figure()
                fig.tight_layout()
                return fig

            fig, axs = plt.subplots(n, figsize=(6, (2.5*n)+2), sharex=True)
            for (m, data), ax in zip(groups, axs):
                data.index = data.index.get_level_values(0)
                try_sort_on_numeric_index(data)
                data.plot(ax=ax, title=m, kind=kind, rot=90)
            fig.tight_layout()
            return fig
        else:
            try_sort_on_numeric_index(df)
            fig = df.plot(figsize=(6,4), kind=kind, rot=90).get_figure()
            fig.tight_layout()
            return fig


class TripDurationsComparison(CsvComparison):

    """
    Compares observed trip durations against thise in trip_logs. Expects bm data with format:

    agent,seq,mode,duration_s
    chris,1,car,454.0
    chris,2,car,463.0
    nick,1,car,4.0
    nick,2,car,454.0

    'agent' is agent_id.
    'seq' is the agent trip sequence number (the current convention starts countimg from 1).
    By setting a kwarg "mode_consistency = true", mode is used to ensure only trips with
    matching mode are compared. Default is false.
    The value field 'duration_s' is the trip duartion in seconds.

    """

    plot_types = ["hist"]

    def __init__(
        self,
        config,
        mode,
        groupby_person_attribute=None,
        **kwargs
        ):
        self.requirements = ['trip_logs']
        self.valid_modes = ["all"]
        self.simulation_name = "trip_logs_all_trips.csv"
        self.value_field = "duration_s"

        # check for mode_consistent option and remove (not a required option for managers)
        if kwargs.pop("mode_consistent", False) is True:
            self.index_fields = ["agent", "seq", "mode"]
        else:
            self.index_fields = ["agent", "seq"]

        self.weight = 1

        super().__init__(
            config,
            mode=mode,
            groupby_person_attribute=groupby_person_attribute,
            **kwargs
            )


class LinkVehicleSpeedsComparison(CsvComparison):

    """
    Compares observed speeds against vehicle_link_speeds. Expects input bm data with format:

    id,8
    1-5,10.0
    5-1,10.0

    or

    id,8,17
    1-5,10.0,9.5
    5-1,10.0,9.8

    or

    id,class,8,17
    1-5,freight,10.0,9.5
    1-5,hhs,10.0,9.8
    5-1,freight,10.0,9.8

    Where 'id' is link_id and value columns represent time_slices.
    Class can be used with the groupby_person_attribute option.
    """

    plot_types = ["hist"]

    def __init__(
        self,
        config,
        mode,
        groupby_person_attribute=None,
        **kwargs
        ):
        self.requirements = ['link_vehicle_speeds']
        self.invalid_modes = ['all']

        # get required time-slice from kwargs
        time_slice = kwargs.pop("time_slice", None)  # this is the required column field, typically hour of day
        if time_slice is None:
            raise ValueError(f"Not found 'time_slice' in {self} kwargs: {kwargs}'")
        self.value_field = str(time_slice)

        self.index_fields = ['link_id']
        self.weight = 1

        super().__init__(
            config,
            mode=mode,
            groupby_person_attribute=groupby_person_attribute,
            **kwargs
            )
        self.groupby_person_attribute = groupby_person_attribute
        self.simulation_name = f"link_vehicle_speeds_{mode}_average"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += ".csv"


class TripModeSharesComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(
        self,
        config,
        mode,
        groupby_person_attribute=None,
        **kwargs
        ):
        self.requirements = ['trip_modes']
        self.valid_modes = ['all']
        self.value_field = 'share'
        self.index_fields = ['mode']
        self.weight = 1

        super().__init__(
            config,
            mode=mode,
            groupby_person_attribute=groupby_person_attribute,
            **kwargs
            )
        self.groupby_person_attribute = groupby_person_attribute
        self.simulation_name = f"trip_modes_all"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_shares.csv"


class TripModeCountsComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(
        self,
        config,
        mode,
        groupby_person_attribute=None,
        **kwargs
        ):
        self.requirements = ['trip_modes']
        self.valid_modes = ['all']
        self.value_field = 'count'
        self.index_fields = ['mode']
        self.weight = 1

        super().__init__(
            config,
            mode=mode,
            groupby_person_attribute=groupby_person_attribute,
            **kwargs
            )
        self.groupby_person_attribute = groupby_person_attribute
        self.simulation_name = f"trip_modes_all"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_counts.csv"


class TripActivityModeSharesComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(self, config, mode, **kwargs):
        self.requirements = ['trip_activity_modes']
        self.valid_modes = ['all']
        self.index_fields = ['mode']
        self.value_field = 'share'
        self.weight = 1

        super().__init__(config, mode=mode, **kwargs)
        destination_activities = kwargs.get("destination_activity_filters", [])
        groupby_person_attribute = kwargs.get("groupby_person_attribute")
        self.simulation_name = f"trip_activity_modes_all"
        for act in destination_activities:
            self.simulation_name += f"_{act}"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_shares.csv"


class TripActivityModeCountsComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(self, config, mode, **kwargs):
        self.requirements = ['trip_activity_modes']
        self.valid_modes = ['all']
        self.index_fields = ['mode']
        self.value_field = 'count'
        self.weight = 1

        super().__init__(config, mode=mode, **kwargs)
        destination_activities = kwargs.get("destination_activity_filters", [])
        groupby_person_attribute = kwargs.get("groupby_person_attribute")
        self.simulation_name = f"trip_activity_modes_all"
        for act in destination_activities:
            self.simulation_name += f"_{act}"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_counts.csv"


class PlanModeSharesComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(
        self,
        config,
        mode,
        groupby_person_attribute=None,
        **kwargs
        ):
        self.requirements = ['plan_modes']
        self.valid_modes = ['all']
        self.value_field = 'share'
        self.index_fields = ['mode']
        self.weight = 1

        super().__init__(
            config,
            mode=mode,
            groupby_person_attribute=groupby_person_attribute,
            **kwargs
            )
        self.groupby_person_attribute = groupby_person_attribute
        self.simulation_name = f"plan_modes_all"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_shares.csv"


class PlanModeCountsComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(
        self,
        config,
        mode,
        groupby_person_attribute=None,
        **kwargs
        ):
        self.requirements = ['plan_modes']
        self.valid_modes = ['all']
        self.value_field = 'count'
        self.index_fields = ['mode']
        self.weight = 1

        super().__init__(
            config,
            mode=mode,
            groupby_person_attribute=groupby_person_attribute,
            **kwargs
            )
        self.groupby_person_attribute = groupby_person_attribute
        self.simulation_name = f"plan_modes_all"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_counts.csv"


class PlanActivityModeSharesComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(self, config, mode, **kwargs):
        self.requirements = ['plan_activity_modes']
        self.valid_modes = ['all']
        self.index_fields = ['mode']
        self.value_field = 'share'
        self.weight = 1

        super().__init__(config, mode=mode, **kwargs)
        destination_activities = kwargs.get("destination_activity_filters", [])
        groupby_person_attribute = kwargs.get("groupby_person_attribute")
        self.simulation_name = f"plan_activity_modes_all"
        for act in destination_activities:
            self.simulation_name += f"_{act}"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_shares.csv"


class PlanActivityModeCountsComparison(CsvComparison):

    plot_types = ["bar"]

    def __init__(self, config, mode, **kwargs):
        self.requirements = ['plan_activity_modes']
        self.valid_modes = ['all']
        self.index_fields = ['mode']
        self.value_field = 'count'
        self.weight = 1

        super().__init__(config, mode=mode, **kwargs)
        destination_activities = kwargs.get("destination_activity_filters", [])
        groupby_person_attribute = kwargs.get("groupby_person_attribute")
        self.simulation_name = f"plan_activity_modes_all"
        for act in destination_activities:
            self.simulation_name += f"_{act}"
        if groupby_person_attribute is not None:
            self.logger.debug(f"Found 'groupby_person_attribute': {groupby_person_attribute}")
            self.simulation_name += f"_{groupby_person_attribute}"
            self.index_fields.append("class")
            self.logger.debug(f"Index fields={self.index_fields}")
        self.simulation_name += "_counts.csv"


class DurationBreakdownComparison(CsvComparison):

    plot_types = ["bar", "line"]

    def __init__(self, config, mode, benchmark_data_path=None, **kwargs):
        super().__init__(config, mode=mode, **kwargs)
        self.benchmark_data_path = benchmark_data_path

    requirements = ['trip_duration_breakdown']
    valid_modes = ['all']
    index_fields = ['duration']
    value_field = 'trips'
    simulation_name = 'trip_duration_breakdown_all.csv'
    weight = 1


class DurationModeBreakdownComparison(CsvComparison):

    plot_types = ["bar", "line"]

    def __init__(self, config, mode, benchmark_data_path=None, **kwargs):
        super().__init__(config, mode=mode, **kwargs)
        self.benchmark_data_path = benchmark_data_path

    requirements = ['trip_duration_breakdown']
    valid_modes = ['all']
    index_fields = ['duration', 'mode']
    value_field = 'trips'
    simulation_name = 'trip_duration_breakdown_mode.csv'
    weight = 1


class DurationDestinationActivityBreakdownComparison(CsvComparison):

    plot_types = ["bar", "line"]

    def __init__(self, config, mode, benchmark_data_path=None, **kwargs):
        super().__init__(config, mode=mode, **kwargs)
        self.benchmark_data_path = benchmark_data_path

    requirements = ['trip_duration_breakdown']
    valid_modes = ['all']
    index_fields = ['duration', 'd_act']
    value_field = 'trips'
    simulation_name = 'trip_duration_breakdown_d_act.csv'
    weight = 1


class EuclideanDistanceBreakdownComparison(CsvComparison):

    plot_types = ["bar", "line"]
    requirements = ['trip_euclid_distance_breakdown']
    valid_modes = ['all']

    index_fields = ['euclidean_distance']
    value_field = 'trips'
    simulation_name = 'trip_euclid_distance_breakdown_all.csv'
    weight = 1


class EuclideanDistanceModeBreakdownComparison(CsvComparison):

    plot_types = ["bar", "line"]
    requirements = ['trip_euclid_distance_breakdown']
    valid_modes = ['all']

    index_fields = ['euclidean_distance', 'mode']
    value_field = 'trips'
    simulation_name = 'trip_euclid_distance_breakdown_mode.csv'
    weight = 1


class LinkCounterComparison(BenchmarkTool):

    requirements = ['link_vehicle_counts']
    options_enabled = True
    weight = 1

    def __str__(self):
        return f'{self.__class__.__name__}: {self.mode}: {self.name}: {self.benchmark_data_path}'

    def __init__(self, config, mode, benchmark_data_path=None, **kwargs) -> None:
        """
        Link volume benchmarker for json formatted {mode: {id: {dir: {links: [], counts: {}}}}}.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(
            config=config,
            mode=mode,
            benchmark_data_path = benchmark_data_path,
            **kwargs
            )

        self.mode = mode
        self.logger.debug(
            f"Initiating: {str(self)} with name: {self.name}"
            )

        with open(self.benchmark_data_path) as json_file:
            self.counts = json.load(json_file)

        missing_counters = 0
        total_counters = 0

        mode_counts = self.counts.get(self.mode)
        if self.mode is None:
            raise UserWarning(
                f"Mode: {self.mode} not found in {str(self)}"
            )

        for counter_id, counter_location in mode_counts.items():

            if counter_id == "TOTAL":
                continue

            for direction, counter in counter_location.items():
                total_counters += 1

                links = counter['links']
                if not links:
                    missing_counters += 1
                    self.logger.warning(
                        f"Benchmark data has no links - suggests error with Bench (i.e. MATSIM network has not matched to BM)."
                        )

        # Check for number of missing BM links. Note this is a fault with the BM (ie missing links)
        missing = missing_counters / total_counters
        self.logger.debug(f"{missing*100}% of BMs are missing snapped links.")
        if total_counters == missing_counters:
            raise UserWarning(
                f"No links found for {str(self)}"
            )
        if missing > 0.5:
            self.logger.warning(
                f"{str(self)} has more than 50% ({missing*100}%) BMs with no links."
                )

    def build(self, resource: dict, write_path: Optional[str]=None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        :return: Dictionary of scores {'name': float}
        """

        self.logger.debug(f'building {str(self)}')

        # extract benchmark mode count
        mode_counts = self.counts.get(self.mode)

        if not mode_counts:
            self.logger.warning(
                f"{self.mode} not available, returning score of one"
            )
            return {'counters': 1}

        # extract counters
        counter_ids = mode_counts.keys()
        total_counters = len(counter_ids)
        if not len(counter_ids):
            self.logger.warning(
                f"no benchmarks found for {self.mode}, returning score of one"
            )
            return {'counters': 1}

        # Extract simulation results
        # Build paths and load appropriate volume counts from previous workstation
        results_name = f"link_vehicle_counts_{self.mode}.csv"
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path)
        results_df.index = results_df.link_id.map(str)  # indices converted to strings
        results_df = results_df[[str(h) for h in range(24)]]  # just keep hourly counts

        # build benchmark results
        snaps = 0
        failed_snaps = 0
        bm_results = []
        bm_summary = []
        bm_scores = []

        for counter_id, counter_location in mode_counts.items():

            if counter_id == "TOTAL":
                continue

            for direction, counter in counter_location.items():

                links = counter['links']
                if links==[]:
                    continue # some links are empty lists, we skip them
                links = [str(link) for link in links]  # force all ids to strings
                bm_hours = [str(h) for h in list(counter['counts'])]
                counts_array = np.array(list(counter['counts'].values()))

                sim_result = np.array([0.0 for _ in range(len(bm_hours))])

                # check if count times are available
                if not set(bm_hours) <= set(results_df.columns):
                    raise UserWarning(
                        f"""Counter: {counter_id}, direction: {direction}:
                        {bm_hours} not available in results.columns:
                        {results_df.columns}"""
                    )

                # combine mode link counts
                for link_id in links:
                    if link_id not in results_df.index:
                        failed_snaps += 1
                        self.logger.warning(
                            f" Missing model link: {link_id}, zero filling count for benchmark: "
                            f"{counter_id}"
                        )
                    else:
                        snaps += 1
                        sim_result += np.array(results_df.loc[str(link_id), bm_hours])

                if not sum(sim_result):
                    found = False
                    continue # this will ignore failed joins and zero counters
                else:
                    found = True

                # calc score
                counter_diff = (sim_result - counts_array) ** 2

                if not sum(counter_diff):
                    counter_score = 0
                elif sum(counts_array):
                    counter_score = sum(counter_diff) / sum(counts_array)
                else:
                    counter_score = 1
                    self.logger.warning(
                        f"Zero size benchmark: {counter_id} link: {link_id}, returning 1"
                    )
                bm_scores.append(counter_score)

                # build result lines for df
                result_line = {
                    'mode': self.mode,
                    'found': found,
                    'counter_id': counter_id,
                    'direction': direction,
                    'links': ','.join(links),
                    'score': counter_score,

                }

                for i, time in enumerate(bm_hours):
                    result_line[f"sim_{str(time)}"] = sim_result[i]

                for i, time in enumerate(bm_hours):
                    result_line[f"bm_{str(time)}"] = counts_array[i]

                for i, time in enumerate(bm_hours):
                    result_line[f"diff_{str(time)}"] = sim_result[i] - counts_array[i]

                bm_results.append(result_line)

                # build summary for df
                result_line = {
                    'source': 'simulation',
                    'mode': self.mode,
                    'counter_id': counter_id,
                    'direction': direction,
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = sim_result[i]

                bm_summary.append(result_line)

                result_line = {
                    'source': 'benchmark',
                    'mode': self.mode,
                    'counter_id': counter_id,
                    'direction': direction,
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = counts_array[i]

                bm_summary.append(result_line)

                result_line = {
                    'source': 'difference',
                    'mode': self.mode,
                    'counter_id': counter_id,
                    'direction': direction,
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = sim_result[i] - counts_array[i]

                bm_summary.append(result_line)

        if failed_snaps:
            report = 100 * failed_snaps / (snaps + failed_snaps)
            self.logger.error(
                f" {report}% of links not found in sim results for {self.mode}: {self.name}")

        # build results df
        bm_results_df = pd.DataFrame(bm_results)
        bm_results_summary = pd.DataFrame(bm_summary).groupby('source').sum()

        # write results
        csv_name = f'{self.name}.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_df, csv_path, write_path=write_path)

        # write results
        csv_name = f'{self.name}_summary.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_summary, csv_path, write_path=write_path)

        # plot
        bm_results_summary_df = merge_summary_stats(bm_results_summary)
        bm_results_summary_plot = comparative_plots(bm_results_summary_df)
        plot_name = f'{self.name}_summary.png'
        bm_results_summary_plot.savefig(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)

        # plot normalised by number of counters
        bm_results_normalised_df = bm_results_summary_df
        bm_results_normalised_df['volume']=bm_results_normalised_df['volume'] / total_counters
        bm_results_normalised_plot = comparative_plots(bm_results_normalised_df)
        plot_name = f'{self.name}_summary_normalised.png'
        bm_results_normalised_plot.savefig(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)
        plt.close()

        return {'counters': sum(bm_scores) / len(bm_scores)}


class TransitInteractionComparison(BenchmarkTool):

    requirements = ['stop_passenger_counts']
    options_enabled = True

    def __str__(self):
        return f'{self.__class__.__name__}: {self.mode}: {self.name}: {self.benchmark_data_path}'

    def __init__(self, config, mode, benchmark_data_path=None, **kwargs) -> None:
        """
        PT Interaction (boardings and alightings) benchmarker for json formatted {mode: {id: {dir: {
        nodes: [], counts: {}}}}}.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, benchmark_data_path=benchmark_data_path, **kwargs)

        self.mode = mode

        self.logger.debug(
            f"Initiating {str(self)}"
            )

        with open(self.benchmark_data_path) as json_file:
            self.counts = json.load(json_file)

        missing_counters = 0
        total_counters = 0

        mode_counts = self.counts.get(self.mode)
        if self.mode is None:
            raise UserWarning(
                f"Mode: {self.mode} not found in {str(self)}"
            )

        for counter_id, counter_location in mode_counts.items():

            if counter_id == "TOTAL":
                continue

            for direction, counter in counter_location.items():
                total_counters += 1

                stops = counter['stop_ids']
                if not stops:
                    missing_counters += 1
                    self.logger.debug(
                        f"Benchmark data has no stop/s - suggests error with Bench (i.e. MATSIM network has not matched to BM)."
                        )

        # Check for number of missing BM stops. Note this is a fault with the BM (ie missing stops)
        missing = missing_counters / total_counters
        self.logger.debug(f"{missing*100}% of BMs are missing snapped stops.")
        if total_counters == missing_counters:
            raise UserWarning(
                f"No stops found for {str(self)}"
            )
        if missing > 0.5:
            self.logger.error(
                f"{str(self)} has more than 50% ({missing*100}%) BMs with no stops."
                )

    def build(self, resource: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        :return: Dictionary of scores {'name': float}
        """

        self.logger.debug(f'building {str(self)}')

        # extract benchmark mode count
        mode_counts = self.counts.get(self.mode)

        if not mode_counts:
            self.logger.warning(
                f"{self.mode} not available, returning score of one"
            )
            return {'counters': 1}

        # extract counters
        counter_ids = mode_counts.keys()
        if not len(counter_ids):
            self.logger.warning(
                f"no benchmarks found for {self.mode}, returning score of one"
            )
            return {'counters': 1}

        # Extract simulation results
        # Build paths and load appropriate volume counts from previous workstation
        model_results = {}
        for direction in ["boardings", "alightings"]:
            results_name = f"stop_passenger_counts_{self.mode}_{direction}.csv"
            results_path = os.path.join(self.config.output_path, results_name)
            results_df = pd.read_csv(results_path, index_col=0,dtype={0:str})
            results_df = results_df[[str(h) for h in range(24)]]  # just keep hourly counts
            results_df.index = results_df.index.map(str)  # indices converted to strings
            results_df.index.name = 'stop_id'
            model_results[direction] = results_df

        # build benchmark results
        snaps = 0
        failed_snaps = 0
        bm_results = []
        bm_summary = []
        bm_scores = []

        for counter_id, counter_location in mode_counts.items():

            if counter_id == "TOTAL":
                continue

            for direction, counter in counter_location.items():

                stops = counter['stop_ids']
                bm_hours = [str(h) for h in list(counter['counts'])]
                counts_array = np.array(list(counter['counts'].values()))

                sim_result = np.array([0.0 for _ in range(len(bm_hours))])

                # check if direction available
                if not direction in model_results:
                    raise UserWarning(
                        f"Direction: {direction} not available in model results"
                        )

                # check if count times are available
                if not set(bm_hours) <= set(model_results[direction].columns):
                    raise UserWarning(
                        f"Counter: {counter_id}, direction: {direction}: {bm_hours} not available in "
                        f"results.columns: {model_results[direction].columns}")

                # combine mode stop counts

                for stop_id in stops:
                    if str(stop_id) not in model_results[direction].index:
                        failed_snaps += 1
                        self.logger.warning(
                            f" Missing model stop: {stop_id}, zero filling count for benchmark: "
                            f"{counter_id}"
                        )
                        found = False
                    else:
                        snaps += 1
                        sim_result += np.array(model_results[direction].loc[str(stop_id), bm_hours])
                        found = True

                # calc score
                counter_diff = (sim_result - counts_array) ** 2

                if not sum(counter_diff):
                    counter_score = 0
                elif sum(counts_array):
                    counter_score = sum(counter_diff) / sum(counts_array)
                else:
                    counter_score = 1
                    self.logger.warning(
                        f"Zero size benchmark: {counter_id} stop: {stop_id}, returning 1"
                    )
                bm_scores.append(counter_score)

                # build result lines for df
                result_line = {
                    'mode': self.mode,
                    'found': found,
                    'counter_id': counter_id,
                    'direction': direction,
                    'stops': ','.join(stops),
                    'score': counter_score,

                }

                for i, time in enumerate(bm_hours):
                    result_line[f"sim_{str(time)}"] = sim_result[i]

                for i, time in enumerate(bm_hours):
                    result_line[f"bm_{str(time)}"] = counts_array[i]

                for i, time in enumerate(bm_hours):
                    result_line[f"diff_{str(time)}"] = sim_result[i] - counts_array[i]

                bm_results.append(result_line)

                # build summary for df
                result_line = {
                    'source': 'simulation',
                    'mode': self.mode,
                    'counter_id': counter_id,
                    'direction': direction,
                    'score': counter_score,

                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = sim_result[i]

                bm_summary.append(result_line)

                result_line = {
                    'source': 'benchmark',
                    'mode': self.mode,
                    'counter_id': counter_id,
                    'direction': direction,
                    'score': counter_score,

                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = counts_array[i]

                bm_summary.append(result_line)

                result_line = {
                    'source': 'difference',
                    'mode': self.mode,
                    'counter_id': counter_id,
                    'direction': direction,
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = sim_result[i] - counts_array[i]

                bm_summary.append(result_line)

        if failed_snaps:
            report = 100 * failed_snaps / (snaps + failed_snaps)
            self.logger.warning(f" {report}% of stop_ids not found for bm: {self.name}")

        # build results df
        bm_results_df = pd.DataFrame(bm_results)
        bm_results_summary = pd.DataFrame(bm_summary).groupby('source').sum()

        # write results
        csv_name = f'{self.name}_{self.mode}.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_df, csv_path, write_path=write_path)

        # write results
        csv_name = f'{self.name}_{self.mode}_summary.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_summary, csv_path, write_path=write_path)

        bm_results_summary_df = merge_summary_stats(bm_results_summary)

        bm_results_summary_plot = comparative_plots(bm_results_summary_df)
        plot_name = f'{self.name}_{self.mode}_summary.png'
        bm_results_summary_plot.savefig(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)
        plt.close()

        return {'counters': sum(bm_scores) / len(bm_scores)}


class PassengerStopToStop(BenchmarkTool):

    name = None
    benchmark_data_path = None
    requirements = ['stop_to_stop_passenger_counts']

    def __str__(self):
        return f'{self.__class__.__name__}: {self.mode}: {self.name}: {self.benchmark_data_path}'

    def __init__(self, config, mode, **kwargs) -> None:
        """
        PT Volume count (between stops) benchmarker for json formatted
        {mode: {o: {d: {
        o_stop_ids: [],
        d_stop_ids: [],
        counts: {},
        line: str
        }}}}.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, **kwargs)

        self.mode = mode

        self.logger.debug(
            f"Initiating {str(self)}"
            )

        with open(self.benchmark_data_path) as json_file:
            self.counts = json.load(json_file)

        missing_counters = 0
        total_counters = 0

        mode_counts = self.counts.get(self.mode)
        if self.mode is None:
            raise UserWarning(
                f"Mode: {self.mode} not found in {str(self)}"
            )

        for od, data in mode_counts.items():
            for _, count_data in data.items():
                total_counters += 1

                o_stops = count_data['o_stop_ids']
                d_stops = count_data['d_stop_ids']
                if not o_stops or not d_stops:
                    missing_counters += 1
                    self.logger.debug(
                        f"Benchmark {od} data has no stop/s - suggests error with Bench (i.e. MATSIM network has not matched to BM)."
                        )

        # Check for number of missing BM stops. Note this is a fault with the BM (ie missing stops)
        missing = missing_counters / total_counters
        self.logger.debug(f"{missing*100}% of BMs are missing snapped stops.")
        if total_counters == missing_counters:
            raise UserWarning(
                f"No stops found for {str(self)}"
            )
        if missing > 0.5:
            self.logger.error(
                f"{str(self)} has more than 50% ({missing*100}%) BMs with no stops."
                )

    def build(self, resource: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        :return: Dictionary of scores {'name': float}
        """

        self.logger.info(f'building {str(self)}')

        # extract benchmark mode count
        mode_counts = self.counts.get(self.mode)

        if not mode_counts:
            self.logger.warning(
                f"{self.mode} not available, returning distance of one"
            )
            return {'counters': 1}

        # extract counters
        ods = mode_counts.keys()
        if not len(ods):
            self.logger.warning(
                f"no benchmarks found for {self.mode}, returning distance of one"
            )
            return {'counters': 1}

        # Extract simulation results
        # Build paths and load appropriate volume counts from previous workstation
        results_name = f"stop_to_stop_passenger_counts_{self.mode}.csv"
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path, index_col=False,dtype = {0:str})
        results_df.origin = results_df.origin.map(str)  # indices converted to strings
        results_df.destination = results_df.destination.map(str)  # indices converted to strings
        results_df = results_df.set_index(["origin", "destination"])
        # results_df = results_df[[str(h) for h in range(24)]]  # just keep hourly counts
        # results_df.index = results_df.index.map(str)  # indices converted to strings
        # results_df.index.name = 'stop_id'

        # build benchmark results
        snaps = 0
        failed_snaps = 0
        bm_results = []
        bm_summary = []
        bm_scores = []

        for od, data in mode_counts.items():
            if od == "TOTAL":
                continue
            for _, count_data in data.items():
                o_stops = count_data['o_stop_ids']
                d_stops = count_data['d_stop_ids']
                bm_hours = list(count_data['counts'])
                counts_array = np.array(list(count_data['counts'].values()))
                line = count_data['line']

                sim_result = np.array([0.0 for _ in range(len(bm_hours))])

                # check if count times are available
                if not set(bm_hours) <= set(results_df.columns):
                    raise UserWarning(
                        f"Hours: {bm_hours} not available in "
                        f"results.columns: {results_df.columns}")

                # combine mode stop counts
                for o_id in o_stops:
                    for d_id in d_stops:
                        if not results_df.index.isin([(o_id, d_id)]).any():
                            failed_snaps += 1
                            self.logger.warning(
                                f" Missing link: {o_id}->{d_id}, "
                                "zero filling count for benchmark: "
                                f"{od}"
                            )
                        else:
                            snaps += 1
                            sim_result = sim_result + np.array(
                                results_df.loc[(o_id, d_id), bm_hours]
                            )

                if not sum(sim_result):
                    found = False
                else:
                    found = True

                # calc score
                counter_diff = np.absolute(sim_result - counts_array)

                if not sum(counter_diff):
                    counter_score = 0
                elif sum(counts_array):
                    counter_score = sum(counter_diff) / sum(counts_array)
                else:
                    counter_score = 1
                    self.logger.warning(
                        f"Zero size benchmark: {od}, {o_stops}->{d_stops}, returning 1"
                    )
                bm_scores.append(counter_score)

                # build result lines for df
                result_line = {
                    'mode': self.mode,
                    'found': found,
                    'counter_id': od,
                    'line': line,
                    'o': ','.join(o_stops),
                    'd': ','.join(d_stops),
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[f"sim_{str(time)}"] = sim_result[i]

                for i, time in enumerate(bm_hours):
                    result_line[f"bm_{str(time)}"] = counts_array[i]

                for i, time in enumerate(bm_hours):
                    result_line[f"diff_{str(time)}"] = sim_result[i] - counts_array[i]

                bm_results.append(result_line)

                # build summary for df
                result_line = {
                    'source': 'simulation',
                    'mode': self.mode,
                    'counter_id': od,
                    'line': line,
                    'o': ','.join(o_stops),
                    'd': ','.join(d_stops),
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = sim_result[i]

                bm_summary.append(result_line)

                result_line = {
                    'source': 'benchmark',
                    'mode': self.mode,
                    'counter_id': od,
                    'line': line,
                    'o': ','.join(o_stops),
                    'd': ','.join(d_stops),
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = counts_array[i]

                bm_summary.append(result_line)

                result_line = {
                    'source': 'difference',
                    'mode': self.mode,
                    'counter_id': od,
                    'line': line,
                    'o': ','.join(o_stops),
                    'd': ','.join(d_stops),
                    'score': counter_score,
                }

                for i, time in enumerate(bm_hours):
                    result_line[time] = sim_result[i] - counts_array[i]

                bm_summary.append(result_line)

        if failed_snaps:
            report = 100 * failed_snaps / (snaps + failed_snaps)
            self.logger.warning(f" {report}% of stop_ids not found for bm: {self.name}")

        # build results df
        bm_results_df = pd.DataFrame(bm_results)
        bm_results_summary = pd.DataFrame(bm_summary).groupby('source').sum()

        # write results
        csv_name = f'{self.name}_{self.mode}.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_df, csv_path, write_path=write_path)

        # write results
        csv_name = f'{self.name}_{self.mode}_summary.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_summary, csv_path, write_path=write_path)

        bm_results_summary_df = merge_summary_stats(bm_results_summary)
        bm_results_summary_plot = comparative_plots(bm_results_summary_df)
        plot_name = f'{self.name}_{self.mode}_summary.png'
        bm_results_summary_plot.savefig(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)
        plt.close()

        return {'counters': sum(bm_scores) / len(bm_scores)}

# ================= Input/Output Plan Comparison BMs  ====================

class PlanComparisonTripStart(CsvComparison):
    simulation_name = 'trip_logs_all_trips.csv'
    requirements = ['trip_logs']
    valid_modes = ['all']
    index_fields = ['agent', 'seq']
    value_field = 'start_s'
    output_value_fields = ['start_s_bm', 'start_s_sim']
    plot = False


class InputPlanComparisonTripStart(PlanComparisonTripStart):
    requirements = ['trip_logs', 'input_trip_logs']
    unsafe_load = True

    def __init__(self, config, **kwargs) -> None:
        self.benchmark_data_path = os.path.join(
            config.output_path,
            'input_trip_logs_all_trips.csv'
        )

        super().__init__(config=config, mode="all")

        data_path_from_config = kwargs.get('benchmark_data_path', None)
        if data_path_from_config is not None:
            self.logger.warning(
                f'InputPlanComparison tool overriding {data_path_from_config} with {self.benchmark_data_path}'
            )


class PlanComparisonTripDuration(CsvComparison):
    simulation_name = 'trip_logs_all_trips.csv'
    requirements = ['trip_logs']
    valid_modes = ['all']
    index_fields = ['agent', 'seq']
    value_field = 'duration_s'
    output_value_fields = ['duration_s_bm', 'duration_s_sim']
    plot = False


class InputPlanComparisonTripDuration(PlanComparisonTripDuration):
    requirements = ['trip_logs', 'input_trip_logs']
    unsafe_load = True

    def __init__(self, config, **kwargs) -> None:

        self.benchmark_data_path = os.path.join(
            config.output_path,
            'input_trip_logs_all_trips.csv'
        )

        super().__init__(config, **kwargs)

        data_path_from_config = kwargs.get('benchmark_data_path', None)
        if data_path_from_config is not None:
            self.logger.warning(
                f'InputPlanComparison tool overriding {data_path_from_config} with {self.benchmark_data_path}'
            )


class PlanComparisonActivityStart(CsvComparison):
    simulation_name = 'trip_logs_all_activities.csv'
    requirements = ['trip_logs']
    valid_modes = ['all']
    index_fields = ['agent', 'seq']
    value_field = 'start_s'
    output_value_fields = ['start_s_bm', 'start_s_sim']
    plot = False


class InputPlanComparisonActivityStart(PlanComparisonActivityStart):
    requirements = ['trip_logs', 'input_trip_logs']
    unsafe_load = True

    def __init__(self, config, **kwargs) -> None:
        self.benchmark_data_path = os.path.join(
            config.output_path,
            'input_trip_logs_all_activities.csv'
        )

        super().__init__(config, **kwargs)

        data_path_from_config = kwargs.get('benchmark_data_path', None)
        if data_path_from_config is not None:
            self.logger.warning(
                f'InputPlanComparison tool overriding {data_path_from_config} with {self.benchmark_data_path}'
            )


class PlanComparisonActivityDuration(CsvComparison):
    simulation_name = 'trip_logs_all_activities.csv'
    requirements = ['trip_logs']
    valid_modes = ['all']
    index_fields = ['agent', 'seq']
    value_field = 'duration_s'
    output_value_fields = ['duration_s_bm', 'duration_s_sim']
    plot = False


class InputPlanComparisonActivityDuration(PlanComparisonActivityDuration):
    requirements = ['trip_logs', 'input_trip_logs']
    unsafe_load = True

    def __init__(self, config, **kwargs) -> None:
        self.benchmark_data_path = os.path.join(
            config.output_path,
            'input_trip_logs_all_activities.csv'
        )

        super().__init__(config, **kwargs)

        data_path_from_config = kwargs.get('benchmark_data_path', None)
        if data_path_from_config is not None:
            self.logger.warning(
                f'InputPlanComparison tool overriding {data_path_from_config} with {self.benchmark_data_path}'
            )


class InputModeComparison(BenchmarkTool):

    requirements = ['input_trip_logs', 'trip_logs']
    unsafe_load = True
    options_enabled = True
    weight = 1
    plot = True

    def __str__(self):
        return f'{self.__class__.__name__}: {self.mode}: {self.name}: {self.benchmark_data_path}'

    def __init__(self, config, benchmark_data_path=None, **kwargs) -> None:
        """
        Creates a table and confusion matrix for input modes compared to post-simulation modes.
        :param config: Config object
        :returns score: dict, {'pct': float}
        """

        # from input_plan_handler
        self.benchmark_data_path = os.path.join(
            config.output_path,
            'input_trip_logs_all_trips.csv'
        )

        # from plan_handler
        self.simulation_data_path = os.path.join(
            config.output_path,
            'trip_logs_all_trips.csv'
        )

        super().__init__(config, **kwargs)

        # Always uses Elara output, issue warning if bm path specified
        # TODO remove this requirement and move all tools into separate module
        data_path_from_config = kwargs.get('benchmark_data_path', None)
        if data_path_from_config is not None:
            self.logger.warning(
                f'InputPlanComparison tool overriding {data_path_from_config} with {self.benchmark_data_path}'
            )

    def build(self, resources: dict, write_path: Optional[str]=None) -> dict:
        usecols = ['agent', 'seq', 'mode']
        indexcols = ['agent', 'seq']

        trips_input = pd.read_csv(self.benchmark_data_path, usecols=usecols, header=0).set_index(indexcols)
        trips_output = pd.read_csv(self.simulation_data_path, usecols=usecols, header=0).set_index(indexcols)

        trips_input.rename({'mode': 'prev_mode'}, axis=1, inplace=True)
        trips_output.rename({'mode': 'new_mode'}, axis=1, inplace=True)

        results_table = trips_input.join(trips_output, how='left')

        # handle unjoined records (plans cannot be completed in full)
        # avoiding nan column name
        results_table['new_mode'].fillna('unmatched', inplace=True)

        # build confusion matrix representations
        results_matrix_counts = results_table.value_counts(subset=['prev_mode', 'new_mode']).unstack()
        results_matrix_counts.fillna(0, inplace=True) # unobserved pairs = 0
        results_matrix_pcts = results_matrix_counts.div(results_matrix_counts.sum(1), axis=0)

        # write results
        csv_name = f'{self.name}.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(results_table, csv_path, write_path=write_path)

        csv_name = f'{self.name}_counts_matrix.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(results_matrix_counts, csv_path, write_path=write_path)

        csv_name = f'{self.name}_pct_matrix.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(results_matrix_pcts, csv_path, write_path=write_path)

        # score = percent correct
        count_no_shift = len(results_table.loc[results_table.prev_mode == results_table.new_mode])
        score = {'pct': (count_no_shift / len(results_table))}

        if self.plot:
            self.plot_heatmap(results_matrix_pcts)

        return score

    def plot_heatmap(
        self,
        df,
        result_name="_matrix_pct",
        cmap='summer',
        figsize=(12,12),
        val_label_size=12
    ) -> None:

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        ax.imshow(df, cmap=cmap, alpha=0.7)

        # set up ticks and lables
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(list(df.columns))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(list(df.index))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # add labels
        for i, idx in enumerate(df.index):
            for j, col in enumerate(df.columns):
                val = df.loc[idx, col]
                val_string = f"{val:.2%}"
                set_val = ax.text(
                    j, i, val_string, ha="center", va="center",
                    color='k', fontweight='normal', fontsize=val_label_size
                )

        plt.ylabel("Original Mode", fontsize=14)
        plt.xlabel("Simulation Mode", fontsize=14)

        plt.tight_layout()

        plt.savefig(os.path.join(self.config.output_path,'benchmarks', f'{self.name}{result_name}.png'))


# ========================== Old style BMs below ==========================

class PointsCounter(BenchmarkTool):

    name = None
    benchmark_data_path = None
    requirements = ['volume_counts']

    def __init__(self, config, mode, groupby_person_attribute=None, **kwargs) -> None:
        """
        Points Counter parent object used for highways traffic counter networks (ie 'coils' or
        'loops').
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.mode = mode

        with open(self.benchmark_data_path) as json_file:
            self.link_counts = json.load(json_file)

        if self.mode not in self.link_counts.keys():
            self.logger.warning(
                f"{self.mode} not available in benchmark data: {self.benchmark_data_path}"
            )

    def build(self, resource: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        :return: Dictionary of scores {'name': float}
        """

        self.logger.info(f'building {str(self)}')

        # extract benchmark mode count
        mode_benchmark = self.link_counts.get(self.mode)

        if not mode_benchmark:
            self.logger.warning(
                f"{self.mode} not available, returning score of one"
            )
            return {'counters': 1}

        # extract counters
        counter_ids = mode_benchmark.keys()
        if not len(counter_ids):
            self.logger.warning(
                f"no benchmarks found for {self.mode}, returning score of one"
            )
            return {'counters': 1}

        # Extract simulation results
        # Build paths and load appropriate volume counts from previous workstation
        results_name = "link_vehicle_counts_{}.csv".format(self.mode)
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path)
        results_df.index = results_df.link_id.map(str)

        results_df = results_df[[str(h) for h in range(24)]]  # just keep counts

        # build benchmark results
        bm_results = []
        bm_scores = []

        for counter_id, links in mode_benchmark.items():

            for link_id, counter in links.items():

                direction = counter['dir']
                bm_counts = np.array(counter['counts'])

                if link_id not in results_df.index:
                    self.logger.warning(
                        f"Zero filling sim results for benchmark: {counter_id} link: {link_id}"
                    )
                    sim_result = np.array([0 for _ in range(24)])
                else:
                    sim_result = np.array(results_df.loc[link_id])

                # calc score
                link_diff = np.absolute(sim_result - bm_counts)
                if sum(bm_counts):
                    link_score = sum(link_diff) / sum(bm_counts)
                else:
                    link_score = 1
                    self.logger.warning(
                        f"Zero size benchmark: {counter_id} link: {link_id}, returning 1"
                    )
                bm_scores.append(link_score)

                # build result lines for df
                sim_result_line = {
                    'source': 'simulation',
                    'counter_id': counter_id,
                    'direction': direction,
                    'link_id': link_id,
                    'score': link_score,
                    'mode': self.mode,
                }
                for h in range(24):
                    sim_result_line[h] = sim_result[h]

                bm_result_line = {
                    'source': 'benchmark',
                    'counter_id': counter_id,
                    'direction': direction,
                    'link_id': link_id,
                    'score': link_score,
                    'mode': self.mode,
                }
                for h in range(24):
                    bm_result_line[h] = bm_counts[h]

                bm_results.extend([sim_result_line, bm_result_line])

        # build results df
        bm_results_df = pd.DataFrame(bm_results)
        bm_results_summary = bm_results_df.groupby('source').sum()

        # write results
        csv_name = f'{self.name}_bm.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_df, csv_path, write_path=write_path)

        # write results
        csv_name = 'f{self.name}_bm_summary.csv'
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_summary, csv_path, write_path=write_path)

        bm_results_summary_df = merge_summary_stats(bm_results_summary)
        bm_results_summary_plot = comparative_plots(bm_results_summary_df)
        plot_name = f'{self.name}_{self.mode}_summary.png'
        bm_results_summary_plot.savefig(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)
        plt.close()

        return {'counters': sum(bm_scores) / len(bm_scores)}


class Cordon(BenchmarkTool):

    cordon_counter = None
    cordon_path = None

    directions = {'in': 1, 'out': 2}
    year = 2016
    hours = None
    modes = None

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs) -> None:
        """
        Cordon parent object used for cordon benchmarks. Initiated with CordonCount
        objects as required.
        :param config: Config object
        :param mode: str, mode
        """
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

        self.cordon_counts = []

        self.mode = mode

        counts_df = pd.read_csv(self.benchmark_data_path)
        links_df = pd.read_csv(self.cordon_path, index_col=0)

        if not self.hours:
            self.hours = range(self.config.time_periods)

        for direction_name, dir_code in self.directions.items():
            self.cordon_counts.append(self.cordon_counter(
                self,
                direction_name,
                dir_code,
                counts_df,
                links_df
            ))

    def build(self, resource: dict, write_path: Optional[str]=None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        Collects scoring from CordonCount objects.
        :return: Dictionary of scores {'in': float, 'out': float}
        """

        self.logger.info(f'building {str(self)}')

        # Build paths and load appropriate volume counts
        results_name = "link_vehicle_counts_{}.csv".format(self.mode)
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path, index_col=0)
        results_df.index.name = 'link_id'

        # get scores and write outputs
        scores = {}
        for cordon_count in self.cordon_counts:
            scores[cordon_count.direction] = cordon_count.output_and_score(
                results_df, write_path=write_path
            )
        return scores


class CordonDirectionCount(BenchmarkTool):

    def __init__(self, parent, direction_name, dir_code, counts_df, links_df, **kwargs):
        """
        Cordon count parent object for counts in or out of a cordon. Includes
        methods for calculating hourly or aggregated period counts.
        :param direction_name: String
        :param dir_code: Int
        :param counts_df: DataFrame of all benchmark counts for cordon
        :param links_df: DataFrame of cordon-count to links
        """
        super().__init__(config=None, mode="all", groupby_person_attribute=None, **kwargs)

        self.cordon_name = parent.name
        self.config = parent.config
        self.year = parent.year
        self.hours = parent.hours
        self.mode = parent.mode

        self.direction = direction_name
        self.dir_code = dir_code
        self.counts_df = counts_df

        # Get cordon links
        self.link_ids = self.get_links(links_df, dir_code)

    @staticmethod
    def get_links(links_df, direction_code):
        """
        Filter given DataFrame for direction and return list of unique link ids.
        :param links_df: DataFrame
        :param direction_code: Int
        :return: List
        """
        df = links_df.loc[links_df.dir == direction_code, :]
        return list(set(df.link))

    def __str__(self):
        return 'f{cls}'

    def get_counts(self, counts_df):
        """
        Builds array of total counts by hour.
        :param counts_df: DataFrame
        :return:
        """
        counts_array = np.zeros(len(self.hours))

        df = counts_df.loc[counts_df.Year == self.year, :]
        assert(len(df)),\
            f'No {self.cordon_name} benchmark counts left from after filtering by {self.year}'

        df = df.loc[df.Hour.isin(self.hours), :]
        assert(len(df)),\
            f'No {self.cordon_name} BM counts left from after filtering by hours:{self.hours}'

        df = df.loc[counts_df.Direction == self.dir_code, :]
        site_indexes = list(set(df.Site))

        for site_index in site_indexes:
            site_df = df.loc[df.Site == site_index, :]
            hour_counts = np.array(site_df.sort_values('Hour').loc[:, self.mode])
            assert len(hour_counts) == len(self.hours),\
                f'Not extracted the right amount of hours {self.hours}'

            counts_array += hour_counts

        return counts_array

    def counts_to_df(self, array, source='benchmark'):
        """
        Build dataframe from array of hourly counts.
        :param array: np.array
        :param source: String
        :return: DataFrame
        """
        col_names = [str(i) for i in self.hours]
        df = pd.DataFrame(array, index=col_names).T
        df.loc[:, 'source'] = source
        return df

    def get_count(self, counts_df):
        """
        Builds total count for period.
        :param counts_df: DataFrame
        :return:
        """
        count = 0

        df = counts_df.loc[counts_df.Year == self.year, :]
        assert(len(df)),\
            f'No {self.cordon_name} benchmark counts left from after filtering by {self.year}'

        df = df.loc[df.Direction == self.dir_code, :]
        site_indexes = list(set(df.Site))

        for site_index in site_indexes:
            site_df = df.loc[df.Site == site_index, :]

            site_count = site_df.loc[:, self.mode].values.sum()

            count += site_count

        return count

    @staticmethod
    def count_to_df(count, source='benchmark'):
        """
        Build count dataframe from int.
        :param count: Int
        :param source: String
        :return: DataFrame
        """
        col_names = ['counts']
        df = pd.DataFrame([count], index=col_names).T
        df.loc[:, 'source'] = source
        return df


class HourlyCordonDirectionCount(CordonDirectionCount):

    def output_and_score(self, result_df, write_path=None):
        """
        Cordon count for hourly data. Joins all results from different volume counts
        (modal) and extract counts for cordon. Scoring is calculated by summing the
        absolute difference between hourly total counts and model results,
        then normalising by the total of all counts.
        :param result_df: DataFrame object of model results
        :param write_path: str object, optional
        :return: Float
        """
        # collect all results
        assert len(result_df), f"zero length results df at {self.cordon_name}."

        for link_id in self.link_ids:
            if link_id not in result_df.index:
                self.logger.warning("Zero filling results for benchmark")
                result_df.loc[link_id] = 0

        model_results = result_df.loc[result_df.index.isin(self.link_ids), :].copy()
        model_results.loc[:, 'mode'] = self.mode

        # write cordon model results
        csv_name = '{}_{}_model_results.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(model_results, csv_path, write_path=write_path)

        # aggregate for each class
        classes_df = model_results.groupby('subpopulation').sum()  # TODO this is hardcoded - will break for other options

        # filter model results for hours
        select_cols = [str(i) for i in self.hours]
        classes_df = classes_df.loc[:, select_cols]

        # Build model results array for scoring
        results_array = np.array(classes_df.sum())

        # Label and write csv with counts by subpopulation
        classes_df.loc[:, 'source'] = 'model'
        csv_name = '{}_{}_classes.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(classes_df, csv_path, write_path=write_path)

        # Get cordon counts for mode
        counts_array = self.get_counts(self.counts_df)
        count_df = self.counts_to_df(counts_array, )

        # Label and write benchmark csv
        benchmark_df = pd.concat([count_df, classes_df]).groupby('source').sum()

        csv_name = '{}_{}_benchmark.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(benchmark_df, csv_path, write_path=write_path)

        # Calc score
        return sum(np.absolute(results_array - counts_array)) / counts_array.sum()


class PeriodCordonDirectionCount(CordonDirectionCount):

    def output_and_score(self, result_df, write_path=None):
        """
        Cordon count for single period data. Joins all results from different volume counts
        (modal) and extract counts for cordon. Scoring is calculated by summing the
        absolute difference between count and model results, then normalising by the
        total of all counts.
        :param result_df: DataFrame object of model results
        :param write_path: Optional output path overwrite
        :return: Float
        """

        # collect all results
        assert len(result_df), f"zero length results df at {self.cordon_name}."

        for link_id in self.link_ids:
            if link_id not in result_df.index:
                self.logger.warning("Zero filling results for benchmark")
                result_df.loc[link_id] = 0

        model_results = result_df.loc[result_df.index.isin(self.link_ids), :].copy()
        model_results.loc[:, 'mode'] = self.mode

        # write cordon model results
        csv_name = '{}_{}_model_results.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(model_results, csv_path, write_path=write_path)

        # aggregate for each class
        classes_df = model_results.groupby('subpopulation').sum()

        # filter model results for hours
        select_cols = [str(i) for i in self.hours]
        classes_df = classes_df.loc[:, select_cols]

        # Build total result for scoring
        result = classes_df.values.sum()
        result_df = self.count_to_df(result, 'model')

        # Label and write csv with counts by subpopulation
        classes_df.loc[:, 'source'] = 'model'
        csv_name = '{}_{}_classes.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(classes_df, csv_path, write_path=write_path)

        # Get cordon count for mode
        count = self.get_count(self.counts_df)
        count_df = self.count_to_df(count, 'benchmark')

        # Label and write benchmark csv
        benchmark_df = pd.concat([count_df, result_df]).groupby('source').sum()
        csv_name = '{}_{}_benchmark.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(benchmark_df, csv_path, write_path=write_path)

        # Calc score
        return np.absolute(result - count) / count


class TestTownHighwayCounters(PointsCounter):

    name = 'test_highways'
    benchmark_data_path = get_benchmark_data(
        os.path.join('test_town', 'highways', 'test_hw_bm.json')
    )

    requirements = ['link_vehicle_counts']
    valid_modes = ['car', 'bus']
    options_enabled = True

    weight = 1


class SqueezeTownHighwayCounters(PointsCounter):

    name = 'squeeze_town_highways'
    benchmark_data_path = get_benchmark_data(
        os.path.join('squeeze_town', 'highways', 'squeeze_town_highways_bm.json')
    )

    requirements = ['link_vehicle_counts']
    valid_modes = ['car', 'bus']
    options_enabled = True

    weight = 1


# Multimodal Test Scenario

class MultimodalTownModeShare(TripModeSharesComparison):

    requirements = ['mode_shares']
    valid_modes = ['all']
    options_enabled = True

    weight = 1
    benchmark_data_path = get_benchmark_data(
        os.path.join('multimodal_town', 'modestats.csv')
    )


class MultimodalTownCarCounters(PointsCounter):

    name = 'car_count'
    benchmark_data_path = get_benchmark_data(
        os.path.join('multimodal_town', 'highways_car_count_bm.json')
    )

    requirements = ['link_vehicle_counts']
    valid_modes = ['car', 'bus']
    options_enabled = True

    weight = 1


class TestTownHourlyCordon(Cordon):

    requirements = ['link_vehicle_counts']
    valid_modes = ['car']
    options_enabled = True

    weight = 1
    cordon_counter = HourlyCordonDirectionCount
    benchmark_data_path = get_benchmark_data(
        os.path.join('test_town', 'test_town_cordon', '2016_counts.csv')
    )
    cordon_path = get_benchmark_data(
        os.path.join('test_town', 'test_town_cordon', 'test_town_cordon.csv')
    )

    directions = {'in': 1, 'out': 2}
    year = 2016
    hours = None
    modes = ['car', 'bus']


class TestTownPeakIn(Cordon):

    requirements = ['link_vehicle_counts']
    valid_modes = ['car']
    options_enabled = True

    weight = 1
    cordon_counter = PeriodCordonDirectionCount
    benchmark_data_path = get_benchmark_data(
        os.path.join('test_town', 'test_town_peak_cordon', '2016_peak_in_counts.csv')
    )
    cordon_path = get_benchmark_data(
        os.path.join('test_town', 'test_town_peak_cordon', 'test_town_cordon.csv')
    )

    directions = {'in': 1}
    year = 2016
    hours = [7, 8, 9]
    modes = ['car', 'bus']


class BenchmarkWorkStation(WorkStation):
    """
    WorkStation class for building benchmarks.
    """

    tools = {
        # trip mode shares and counts
        "trip_mode_shares_comparison": TripModeSharesComparison,
        "trip_activity_mode_shares_comparison": TripActivityModeSharesComparison,
        "trip_mode_counts_comparison": TripModeCountsComparison,
        "trip_activity_mode_counts_comparison": TripActivityModeCountsComparison,

        # plan mode shares and counts
        "plan_mode_shares_comparison": PlanModeSharesComparison,
        "plan_activity_mode_shares_comparison": PlanActivityModeSharesComparison,
        "plan_mode_counts_comparison": PlanModeCountsComparison,
        "plan_activity_mode_counts_comparison": PlanActivityModeCountsComparison,

        # trip breakdowns - aggregate distribution comparisons
        "euclidean_distance_breakdown_comparison": EuclideanDistanceBreakdownComparison,
        "euclidean_distance_mode_breakdown_comparison": EuclideanDistanceModeBreakdownComparison,
        "duration_breakdown_comparison": DurationBreakdownComparison,
        "duration_mode_breakdown_comparison": DurationModeBreakdownComparison,
        "duration_d_act_breakdown_comparison": DurationDestinationActivityBreakdownComparison,

        # traditional benchmarks - eg cordons etc
        "link_counter_comparison": LinkCounterComparison,
        "transit_interaction_comparison": TransitInteractionComparison,

        # plan comparisons
        "plan_comparison_trip_start": PlanComparisonTripStart,
        "input_plan_comparison_trip_start": InputPlanComparisonTripStart,
        "plan_comparison_trip_duration": PlanComparisonTripDuration,
        "input_plan_comparison_trip_duration": InputPlanComparisonTripDuration,
        "plan_comparison_activity_start": PlanComparisonActivityStart,
        "input_plan_comparison_activity_start": InputPlanComparisonActivityStart,
        "plan_comparison_activity_duration": PlanComparisonActivityDuration,
        "input_plan_comparison_activity_duration": InputPlanComparisonActivityDuration,
        "input_mode_comparison": InputModeComparison,

        # new benchmarks - link speeds, trip durations
        "link_vehicle_speeds_comparison": LinkVehicleSpeedsComparison,
        "trip_durations_comparison": TripDurationsComparison,

        # old style, maintained for posterity and backward compatability:
        "test_town_highways": TestTownHighwayCounters,
        "squeeze_town_highways": SqueezeTownHighwayCounters,
        "multimodal_town_modeshare": MultimodalTownModeShare,
        "multimodal_town_cars_counts": MultimodalTownCarCounters,
        "test_town_cordon": TestTownHourlyCordon,
        "test_town_peak_cordon": TestTownPeakIn,
    }

    benchmarks = {}
    scores_df = None
    meta_score = 0

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # Create output folder if it does not exist
        benchmark_dir = os.path.join(self.config.output_path, 'benchmarks')
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)

    def build(self, spinner=None, write_path=None) -> None:
        """
        Calculates all sub scores from benchmarks, writes to disk and returns
        combined metascore.
        """
        summary = {}
        flat_summary = []

        self.logger.info('--BENCHMARK SCORES--')

        for benchmark_name, benchmark in self.resources.items():

            scores = benchmark.build({}, write_path=write_path)
            weight = benchmark.weight

            sub_summary = {'scores': scores,
                           'weight': weight
                           }
            summary[benchmark_name] = sub_summary

            for name, score in scores.items():
                self.logger.info(f' *** {benchmark_name} {name} = {score} *** ')
                flat_summary.append([benchmark_name, name, score])
                self.meta_score += (score * weight)

        self.logger.info(f' *** Meta Score = {self.meta_score} ***')

        # Write scores
        csv_name = 'benchmark_scores.csv'

        self.scores_df = pd.DataFrame(flat_summary, columns=['benchmark', 'type', 'score'])
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(self.scores_df, csv_path, write_path=write_path)

        summary['meta_score'] = self.meta_score
        json_name = 'benchmark_scores.json'
        json_path = os.path.join('benchmarks', json_name)
        self.write_json(summary, json_path, write_path=write_path)


def merge_summary_stats(bm_results_summary):

    bm_results_summary = bm_results_summary.reset_index()

    data = json.loads(bm_results_summary.to_json(orient='records'))

    results = []

    for record in data:
        record_type = record['source']

        record.pop("score")
        record.pop("source")

        if record_type != "difference":
            for measurement in list(record):
                results.append(
                    {
                    "hour" : int(measurement),
                    "volume" : record[measurement],
                    "type" : record_type
                    })

    return pd.DataFrame(results)


def comparative_plots(results):

    bms = results[results['type'] == 'benchmark']
    sims = results[results['type'] == 'simulation']

    fig, ax = plt.subplots()
    ax.plot(bms['hour'], bms['volume'], label='benchmark', marker ='.', linewidth=1)
    ax.plot(sims['hour'], sims['volume'], label='simulation', marker ='.',linewidth=1)
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Volume")
    ax.legend(loc='best')
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.set_axisbelow(True) 

    return fig
from plotnine import * 
import pandas as pd
import os
import numpy as np
from typing import Optional
import logging
import json

from elara.factory import WorkStation, Tool
from elara import get_benchmark_data


logger = logging.getLogger(__name__)

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

    return ggplot(aes(y="volume",x="hour",color="type"),data=results) + geom_point() + geom_line() + labs(y="Volume", x="Time (hour)")


class BenchmarkTool(Tool):

    options_enabled = True

    def __init__(self, config, option=None):
        super().__init__(config, option)
        self.logger = logging.getLogger(__name__)


class LinkCounter(BenchmarkTool):

    name = None
    benchmark_data_path = None
    requirements = ['volume_counts']

    def __init__(self, config, option) -> None:
        """
        Link volume benchmarker for json formatted {mode: {id: {dir: {links: [], counts: {}}}}}.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

        self.mode = option

        with open(self.benchmark_data_path) as json_file:
            self.counts = json.load(json_file)

        missing_bms = 0
        total_bms = 0
        
        if self.mode not in self.counts.keys():
            self.logger.warning(
                f"{self.mode} not available in benchmark data: {self.benchmark_data_path}"
            )

        mode_counts = self.counts.get(self.mode)
        
        for counter_id, counter_location in mode_counts.items():

            for direction, counter in counter_location.items():

                total_bms = total_bms + 1

                links = counter['links']

                if not links:
                    
                    missing_bms = missing_bms + 1
                    self.logger.warning(f"Benchmarks feature no snapped links - suggests error with Bench (i.e. MATSIM network has not matched to BM).")

        # If more than 33% of BM's are missing, hard fails
        self.logger.warning("{} percentage of BMs are missing snapped links.".format(missing_bms/total_bms*100.0))
        
        if missing_bms / total_bms > 0.66:

            raise UserWarning(f"Exiting for your own good - too many BM's (over 33%) are missing matched links. This is an issue with Bench")

    def build(self, resource: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        :return: Dictionary of scores {'name': float}
        """

        logger.info(f'building {self.__str__()}')

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
        results_name = "{}_volume_counts_{}.csv".format(self.config.name, self.mode)
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path, index_col=0)
        results_df = results_df.groupby(results_df.index).sum()  # remove class dis-aggregation
        results_df = results_df[[str(h) for h in range(24)]]  # just keep hourly counts
        results_df.index.name = 'link_id'
        results_df.index = results_df.index.map(str)

        # build benchmark results
        snaps = 0
        failed_snaps = 0
        bm_results = []
        bm_summary = []
        bm_scores = []

        for counter_id, counter_location in mode_counts.items():

            for direction, counter in counter_location.items():

                links = counter['links']
                if links==[]:
                    continue # some links are empty lists, we skip them
                bm_hours = list(counter['counts'])
                counts_array = np.array(list(counter['counts'].values()))

                sim_result = np.array([0.0 for _ in range(len(bm_hours))])

                # check if count times are available
                if not set(bm_hours) <= set(results_df.columns):
                    raise UserWarning(
                        f"Hours: {bm_hours} not available in results.columns: {results_df.columns}")

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
                        sim_result += np.array(results_df.loc[link_id, bm_hours])

                if not sum(sim_result):
                    found = False
                    continue # this will ignore failed joins and zero counters
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
            self.logger.warning(f" {report}% of links not found for bm: {self.name}")

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

        # plot
        bm_results_summary_df = merge_summary_stats(bm_results_summary)
        bm_results_summary_plot = comparative_plots(bm_results_summary_df)
        plot_name = f'{self.name}_{self.mode}_summary.png'
        bm_results_summary_plot.save(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)

        return {'counters': sum(bm_scores) / len(bm_scores)}


class TestCordon(LinkCounter):

    name = 'test_link_counter'
    benchmark_data_path = get_benchmark_data(
        os.path.join('test_town', 'test_town_cordon', 'test_link_counter.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class IrelandHighwayCountersNew(LinkCounter):

    name = 'ireland_highways_counters_new'
    benchmark_data_path = get_benchmark_data(
        os.path.join('ireland', 'highways', 'ireland_highways_counters_14july20.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class LondonCentralCordon(LinkCounter):

    name = 'london_central_cordon'
    benchmark_data_path = get_benchmark_data(
        os.path.join('london', 'london-GLA', 'central_cordon.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class LondonInnerCordon(LinkCounter):

    name = 'london_inner_cordon'
    benchmark_data_path = get_benchmark_data(
        os.path.join('london', 'london-GLA', 'inner_cordon.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class LondonOuterCordon(LinkCounter):

    name = 'london_outer_cordon'
    benchmark_data_path = get_benchmark_data(
        os.path.join('london', 'london-GLA', 'boundary_cordon.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class LondonThamesScreen(LinkCounter):

    name = 'london_thames_screen'
    benchmark_data_path = get_benchmark_data(
        os.path.join('london', 'london-GLA', 'thames_screen.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class LondonNorthScreen(LinkCounter):

    name = 'london_northern_screen'
    benchmark_data_path = get_benchmark_data(
        os.path.join('london', 'london-GLA', 'northern_screen.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class LondonPeriphScreen(LinkCounter):

    name = 'london_peripheral_screen'
    benchmark_data_path = get_benchmark_data(
        os.path.join('london', 'london-GLA', 'peripheral_screen.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1



class TransitInteraction(BenchmarkTool):

    name = None
    benchmark_data_path = None
    requirements = ['stop_interactions']

    def __init__(self, config, option) -> None:
        """
        PT Interaction (boardings and alightings) benchmarker for json formatted {mode: {id: {dir: {
        nodes: [], counts: {}}}}}.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

        self.mode = option

        with open(self.benchmark_data_path) as json_file:
            self.counts = json.load(json_file)

        if self.mode not in self.counts.keys():
            self.logger.warning(
                f"{self.mode} not available in benchmark data: {self.benchmark_data_path}"
            )

    def build(self, resource: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        :return: Dictionary of scores {'name': float}
        """

        logger.info(f'building {self.__str__()}')

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
            results_name = f"{self.config.name}_{direction}_{self.mode}.csv"
            results_path = os.path.join(self.config.output_path, results_name)
            results_df = pd.read_csv(results_path, index_col=0)
            results_df = results_df.groupby(results_df.index).sum()  # remove class dis-aggregation
            results_df = results_df[[str(h) for h in range(24)]]  # just keep hourly counts
            results_df.index.name = 'link_id'
            model_results[direction] = results_df

        # build benchmark results
        snaps = 0
        failed_snaps = 0
        bm_results = []
        bm_summary = []
        bm_scores = []

        for counter_id, counter_location in mode_counts.items():

            for direction, counter in counter_location.items():

                stops = counter['nodes']
                bm_hours = list(counter['counts'])
                counts_array = np.array(list(counter['counts'].values()))

                sim_result = np.array([0.0 for _ in range(len(bm_hours))])

                # check if count times are available
                if not set(bm_hours) <= set(model_results[direction].columns):
                    raise UserWarning(
                        f"Hours: {bm_hours} not available in "
                        f"results.columns: {model_results[direction].columns}")

                # combine mode link counts
                for stop_id in stops:
                    if stop_id not in model_results[direction].index:
                        failed_snaps += 1
                        self.logger.warning(
                            f" Missing model node: {stop_id}, zero filling count for benchmark: "
                            f"{counter_id}"
                        )
                    else:
                        snaps += 1
                        sim_result += np.array(model_results[direction].loc[stop_id, bm_hours])

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
                        f"Zero size benchmark: {counter_id} link: {stop_id}, returning 1"
                    )
                bm_scores.append(counter_score)

                # build result lines for df
                result_line = {
                    'mode': self.mode,
                    'found': found,
                    'counter_id': counter_id,
                    'direction': direction,
                    'links': ','.join(stops),
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
            self.logger.warning(f" {report}% of links not found for bm: {self.name}")

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

        bm_results_summary_plot.save(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)

        return {'counters': sum(bm_scores) / len(bm_scores)}

class TestPTInteraction(TransitInteraction):

    name = 'test_pt_interaction_counter'
    benchmark_data_path = get_benchmark_data(
        os.path.join('test_town', 'pt_interactions', 'test_interaction_counter.json')
    )

    requirements = ['stop_interactions']
    valid_options = ['bus']
    options_enabled = True

    weight = 1


class LondonRODS(TransitInteraction):

    name = 'london_rods'
    benchmark_data_path = get_benchmark_data(
        os.path.join('london', 'london-GLA', 'board_alight.json')
    )

    requirements = ['stop_interactions']
    valid_options = ['subway']
    options_enabled = True

    weight = 1


# ========================== Old style BMs below ==========================


class PointsCounter(BenchmarkTool):

    name = None
    benchmark_data_path = None
    requirements = ['volume_counts']

    def __init__(self, config, option) -> None:
        """
        Points Counter parent object used for highways traffic counter networks (ie 'coils' or
        'loops').
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

        self.name = self.config.name
        self.mode = option

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

        logger.info(f'building {self.__str__()}')

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
        results_name = "{}_volume_counts_{}.csv".format(self.config.name, self.mode)
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path, index_col=0)

        results_df = results_df.groupby(results_df.index).sum()  # remove class dis-aggregation

        results_df = results_df[[str(h) for h in range(24)]]  # just keep counts

        results_df.index.name = 'link_id'

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
        csv_name = '{}_{}_bm.csv'.format(self.config.name, self.name)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_df, csv_path, write_path=write_path)

        # write results
        csv_name = '{}_{}_bm_summary.csv'.format(self.config.name, self.name)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(bm_results_summary, csv_path, write_path=write_path)

        bm_results_summary_df = merge_summary_stats(bm_results_summary)

        bm_results_summary_plot = comparative_plots(bm_results_summary_df)

        plot_name = f'{self.name}_{self.mode}_summary.png'

        bm_results_summary_plot.save(os.path.join(self.config.output_path,"benchmarks", plot_name), verbose=False)

        return {'counters': sum(bm_scores) / len(bm_scores)}


class Cordon(BenchmarkTool):

    cordon_counter = None
    benchmark_path = None
    cordon_path = None

    directions = {'in': 1, 'out': 2}
    year = 2016
    hours = None
    modes = None

    def __init__(self, config, option) -> None:
        """
        Cordon parent object used for cordon benchmarks. Initiated with CordonCount
        objects as required.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

        self.cordon_counts = []

        self.name = self.config.name
        self.mode = option

        counts_df = pd.read_csv(self.benchmark_path)
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

    def build(self, resource: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        Collects scoring from CordonCount objects.
        :return: Dictionary of scores {'in': float, 'out': float}
        """

        logger.info(f'building {self.__str__()}')

        # Build paths and load appropriate volume counts
        results_name = "{}_volume_counts_{}.csv".format(self.config.name, self.mode)
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

    def __init__(self, parent, direction_name, dir_code, counts_df,
                 links_df):
        """
        Cordon count parent object for counts in or out of a cordon. Includes
        methods for calculating hourly or aggregated period counts.
        :param direction_name: String
        :param dir_code: Int
        :param counts_df: DataFrame of all benchmark counts for cordon
        :param links_df: DataFrame of cordon-count to links
        """
        super().__init__(config=None, option=None)

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
        classes_df = model_results.groupby('class').sum()

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
        classes_df = model_results.groupby('class').sum()

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


class ModeStats(BenchmarkTool):

    benchmark_path = None

    def __init__(self, config, option):
        """
        ModeStat parent object for benchmarking with mode share data.
        :param config: Config object
        :param option: str, mode
        """
        super().__init__(config, option)

        self.benchmark_df = pd.read_csv(self.benchmark_path,
                                        header=None,
                                        names=['mode', 'benchmark'])
        self.benchmark_df.set_index('mode', inplace=True)
        self.name = self.config.name

    def build(self, resource: dict, write_path: Optional[str] = None) -> dict:
        """
        Builds paths for mode share outputs, loads and combines with model for scoring.
        :return: Dictionary of scores
        """
        # Build paths and load appropriate volume counts
        results_name = "{}_mode_shares_all_total.csv".format(self.name)
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path,
                                 header=None,
                                 names=['mode', 'model'])
        results_df.set_index('mode', inplace=True)

        # join results
        summary_df = self.benchmark_df.join(results_df, how='inner')
        summary_df.loc[:, 'diff'] = summary_df.model - summary_df.benchmark

        # write results
        csv_name = '{}_modeshare_results.csv'.format(self.name)
        csv_path = os.path.join('benchmarks', csv_name)
        self.write_csv(summary_df, csv_path, write_path=write_path)

        # get scores and write outputs
        score = sum(np.absolute(np.array(summary_df.benchmark) - np.array(summary_df.model)))

        return {'modeshare': score}


# Highway Counters

class TestHighwayCounters(PointsCounter):

    name = 'test_highways'
    benchmark_data_path = get_benchmark_data(
        os.path.join('test_town', 'highways', 'test_hw_bm.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class IrelandHighwayCounters(PointsCounter):

    name = 'ireland_highways'
    benchmark_data_path = get_benchmark_data(
        os.path.join('ireland', 'highways', 'irish_highways_bm.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


class SqueezeTownHighwayCounters(PointsCounter):

    name = 'squeeze_town_highways'
    benchmark_data_path = get_benchmark_data(
        os.path.join('squeeze_town', 'highways', 'squeeze_town_highways_bm.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


# Multimodal Test Scenario

class MultimodalTownModeShare(ModeStats):

    requirements = ['mode_share']
    valid_options = ['all']
    options_enabled = True

    weight = 1
    benchmark_path = get_benchmark_data(
        os.path.join('multimodal_town', 'modestats.csv')
    )


class MultimodalTownCarCounters(PointsCounter):

    name = 'car_count'
    benchmark_data_path = get_benchmark_data(
        os.path.join('multimodal_town', 'highways_car_count_bm.json')
    )

    requirements = ['volume_counts']
    valid_options = ['car', 'bus']
    options_enabled = True

    weight = 1


# Cordons

class LondonInnerCordonCar(Cordon):

    requirements = ['volume_counts']
    valid_options = ['car']
    options_enabled = True

    weight = 1
    cordon_counter = HourlyCordonDirectionCount
    benchmark_path = get_benchmark_data(
        os.path.join('london', 'inner_cordon', 'InnerCordon2016.csv')
    )
    cordon_path = get_benchmark_data(
        os.path.join('london', 'inner_cordon', 'cordon_links.csv')
    )

    directions = {'in': 1, 'out': 2}
    year = 2016
    hours = None
    modes = ['car']


class DublinCanalCordonCar(Cordon):

    requirements = ['volume_counts']
    valid_options = ['car']
    options_enabled = True

    weight = 1
    cordon_counter = PeriodCordonDirectionCount
    benchmark_path = get_benchmark_data(
        os.path.join('ireland', 'dublin_cordon', '2016_counts.csv')
    )
    cordon_path = get_benchmark_data(
        os.path.join('ireland', 'dublin_cordon', 'dublin_cordon.csv')
    )

    directions = {'in': 1}
    year = 2016
    hours = [7, 8, 9]
    modes = ['car']


class IrelandCommuterStats(ModeStats):

    requirements = ['mode_share']
    valid_options = ['all']
    options_enabled = True

    weight = 1
    benchmark_path = get_benchmark_data(
        os.path.join('ireland', 'census_modestats', '2016_census_modestats.csv')
    )


class TestTownHourlyCordon(Cordon):

    requirements = ['volume_counts']
    valid_options = ['car']
    options_enabled = True

    weight = 1
    cordon_counter = HourlyCordonDirectionCount
    benchmark_path = get_benchmark_data(
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

    requirements = ['volume_counts']
    valid_options = ['car']
    options_enabled = True

    weight = 1
    cordon_counter = PeriodCordonDirectionCount
    benchmark_path = get_benchmark_data(
        os.path.join('test_town', 'test_town_peak_cordon', '2016_peak_in_counts.csv')
    )
    cordon_path = get_benchmark_data(
        os.path.join('test_town', 'test_town_peak_cordon', 'test_town_cordon.csv')
    )

    directions = {'in': 1}
    year = 2016
    hours = [7, 8, 9]
    modes = ['car', 'bus']


class TestTownCommuterStats(ModeStats):

    requirements = ['mode_share']
    valid_options = ['all']
    options_enabled = True

    weight = 1
    benchmark_path = get_benchmark_data(
        os.path.join('test_town', 'census_modestats', 'test_town_modestats.csv')
    )


class BenchmarkWorkStation(WorkStation):
    """
    WorkStation class for building benchmarks.
    """

    tools = {
        "ireland_highways": IrelandHighwayCountersNew,
        "london_board_alight": LondonRODS,
        # "london_boundary_cordon": LondonOuterCordon,
        "london_central_cordon": LondonCentralCordon,
        "london_inner_cordon": LondonInnerCordon,
        # "london_northern_screen": LondonNorthScreen,
        # "london_peripheral_screen": LondonPeriphScreen,
        # "london_thames_screen": LondonThamesScreen,

        "test_pt_interaction_counter": TestPTInteraction,
        "test_town_highways": TestHighwayCounters,
        "squeeze_town_highways": SqueezeTownHighwayCounters,
        "multimodal_town_modeshare": MultimodalTownModeShare,
        "multimodal_town_cars_counts": MultimodalTownCarCounters,
        # "ireland_highways": IrelandHighwayCounters, # replaced with new
        "london_inner_cordon_car": LondonInnerCordonCar,
        "dublin_canal_cordon_car": DublinCanalCordonCar,
        "ireland_commuter_modeshare": IrelandCommuterStats,
        "test_link_cordon": TestCordon,
        "test_town_cordon": TestTownHourlyCordon,
        "test_town_peak_cordon": TestTownPeakIn,
        "test_town_modeshare": TestTownCommuterStats,
    }

    BENCHMARK_WEIGHTS = {
        "test_pt_interaction_counter": 1,
        "london_rods": 1,
        "test_link_cordon": 1,
        "ireland_highways": 1,
        "london_central_cordon": 1,
        "london_inner_cordon": 1,
        "london_outer_cordon": 1,
        "london_thames_screen": 1,

        "test_town_highways": 1,
        "squeeze_town_highways": 1,
        "multimodal_town_modeshare": 1,
        "multimodal_town_cars_counts": 1,
        # "ireland_highways": 1, # replaced with new
        "london_inner_cordon_car": 1,
        "dublin_canal_cordon_car": 1,
        "ireland_commuter_modeshare": 1,
        "test_town_cordon": 1,
        "test_town_peak_cordon": 1,
        "test_town_modeshare": 1
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
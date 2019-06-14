import pandas as pd
import os
import numpy as np
import json
from elara import get_benchmark_data

# TODO this module has some over complex operations on the input data...
# TODO - needs tests and data validation

class Benchmarks:

    benchmarks = {}
    scores_df = None
    meta_score = 0

    def __init__(self, config):
        """
        Wrapper for all benchmarks (ie Cordons ect).
        :param config: Config object
        """

        self.config = config

        # Create output folder if it does not exist
        benchmark_dir = os.path.join(self.config.output_path, 'benchmarks')
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)

        for cordon_name in config.benchmarks:

            # TODO add more explicit input data checking for each benchmark

            self.benchmarks[cordon_name] = BENCHMARK_MAP[cordon_name](
                cordon_name, config)

    def score(self):
        """
        Calculates all sub scores from benchmarks, writes to disk and returns
        combined metascore.
        """
        summary = {}
        flat_summary = []
        for benchmark_name, benchmark in self.benchmarks.items():
            scores = benchmark.output_and_score()
            weight = BENCHMARK_WEIGHTS[benchmark_name]

            sub_summary = {'scores': scores,
                           'weight': weight
                           }
            summary[benchmark_name] = sub_summary

            for name, s in scores.items():
                flat_summary.append([benchmark_name, name, s])
                self.meta_score += (s * weight)

        # Write scores
        self.scores_df = pd.DataFrame(flat_summary, columns=['benchmark', 'type', 'score'])
        csv_name = 'benchmark_scores.csv'
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        self.scores_df.to_csv(csv_path)

        summary['meta_score'] = self.meta_score
        json_name = 'benchmark_scores.json'
        json_path = os.path.join(self.config.output_path, 'benchmarks', json_name)
        with open(json_path, 'w') as outfile:
            json.dump(summary, outfile)


class Cordon:

    cordon_counter = None
    benchmark_path = None
    cordon_path = None

    directions = {'in': 1, 'out': 2}
    year = 2016
    hours = None
    modes = None

    def __init__(self, name, config):
        """
        Cordon parent object used for cordon benchmarks. Initiated with CordonCount
        objects as required.
        :param name: String, cordon name
        :param config: Config
        """
        self.cordon_counts = []
        self.name = name
        self.config = config
        counts_df = pd.read_csv(self.benchmark_path)
        links_df = pd.read_csv(self.cordon_path, index_col=0)

        if not self.hours:
            self.hours = range(self.config.time_periods)

        for direction_name, dir_code in self.directions.items():
            self.cordon_counts.append(self.cordon_counter(
                self.name,
                self.config,
                self.year,
                self.hours,
                direction_name,
                dir_code,
                counts_df,
                links_df
            ))

    def output_and_score(self):
        """
        Builds paths for modal volume count outputs, loads and combines for scoring.
        Collects scoring from CordonCount objects.
        :return: Dictionary of scores {'in': float, 'out': float}
        """

        # Build paths and load appropriate volume counts
        results_dfs = {}
        for mode in self.modes:
            results_name = "{}_volume_counts_{}.csv".format(self.config.name, mode)
            results_path = os.path.join(self.config.output_path, results_name)
            results_df = pd.read_csv(results_path, index_col=0)
            results_df.index.name = 'link_id'
            results_dfs[mode] = results_df

        # get scores and write outputs
        scores = {}
        for cordon_count in self.cordon_counts:
            scores[cordon_count.direction] = cordon_count.output_and_score(results_dfs)
        return scores


class CordonCount:

    def __init__(self, name, config, year, hours, direction_name, dir_code, counts_df, links_df):
        """
        Cordon count parent object for counts in or out of a cordon. Includes
        methods for calculating hourly or aggregated period counts.
        :param name: String
        :param config: Config
        :param direction_name: String
        :param dir_code: Int
        :param counts_df: DataFrame of all benchmark counts for cordon
        :param links_df: DataFrame of cordon-count to links
        """
        self.cordon_name = name
        self.direction = direction_name
        self.config = config
        self.year = year
        self.hours = hours
        self.dir_code = dir_code
        self.counts_df = counts_df

        # Get cordon links
        self.link_ids = self.get_links(links_df, dir_code)

    def get_links(self, links_df, direction_code):
        """
        Filter given DataFrame for direction and return list of unique link ids.
        :param links_df: DataFrame
        :param direction_code: Int
        :return: List
        """
        df = links_df.loc[links_df.dir == direction_code, :]
        return list(set(df.link))

    def get_counts(self, counts_df, modes):
        """
        Builds array of total counts by hour.
        :param counts_df: DataFrame
        :param modes: List of mode Strings
        :return:
        """
        counts_array = np.zeros(len(self.hours))

        df = counts_df.loc[counts_df.Year == self.year, :]
        assert(len(df)), f'No {self.cordon_name} benchmark counts left from after filtering by {self.year}'

        df = df.loc[df.Hour.isin(self.hours), :]
        assert(len(df)), f'No {self.cordon_name} benchmark counts left from after filtering by hours:{self.hours}'

        df = df.loc[counts_df.Direction == self.dir_code, :]
        site_indexes = list(set(df.Site))

        for site_index in site_indexes:
            site_df = df.loc[df.Site == site_index, :]
            hour_counts = np.array(site_df.sort_values('Hour').loc[:, modes]).sum(axis=1)
            assert len(hour_counts) == len(self.hours), f'Not extracted the right amount of hours {self.hours}'

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

    def get_count(self, counts_df, modes):
        """
        Builds total count for period.
        :param counts_df: DataFrame
        :param modes: List of mode Strings
        :return:
        """
        count = 0

        df = counts_df.loc[counts_df.Year == self.year, :]
        assert(len(df)), f'No {self.cordon_name} benchmark counts left from after filtering by {self.year}'

        df = df.loc[df.Direction == self.dir_code, :]
        site_indexes = list(set(df.Site))

        for site_index in site_indexes:
            site_df = df.loc[df.Site == site_index, :]

            site_count = site_df.loc[:, modes].values.sum()

            count += site_count

        return count

    def count_to_df(self, count, source='benchmark'):
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


class HourlyCordonCount(CordonCount):

    def output_and_score(self, results_dfs):
        """
        Cordon count for hourly data. Joins all results from different volume counts
        (modal) and extract counts for cordon. Scoring is calculated by summing the
        absolute difference between hourly total counts and model results,
        then normalising by the total of all counts.
        :param results_dfs: DataFrame object of model results
        :return: Float
        """
        # collect all results
        model_results = pd.DataFrame()
        for mode, result_df in results_dfs.items():
            if len(result_df):
                result_df.loc[:, 'mode'] = mode
                mode_results = result_df.loc[result_df.index.isin(self.link_ids), :].copy()
                model_results = pd.concat([model_results, mode_results], axis=0)

        # write cordon model results
        csv_name = '{}_{}_model_results.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        model_results.to_csv(csv_path)

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
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        classes_df.to_csv(csv_path)

        # Get cordon counts for mode
        counts_array = self.get_counts(self.counts_df, modes=results_dfs.keys())
        count_df = self.counts_to_df(counts_array, )

        # Label and write benchmark csv
        benchmark_df = pd.concat([count_df, classes_df]).groupby('source').sum()

        csv_name = '{}_{}_benchmark.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        benchmark_df.to_csv(csv_path)

        # Calc score
        return sum(np.absolute(results_array - counts_array)) / counts_array.sum()


class PeriodCordonCount(CordonCount):

    def output_and_score(self, results_dfs):
        """
        Cordon count for single period data. Joins all results from different volume counts
        (modal) and extract counts for cordon. Scoring is calculated by summing the
        absolute difference between count and model results, then normalising by the
        total of all counts.
        :param results_dfs: DataFrame object of model results
        :return: Float
        """

        # collect all results
        model_results = pd.DataFrame()
        for mode, result_df in results_dfs.items():
            mode_results = result_df.loc[result_df.index.isin(self.link_ids), :].copy()
            mode_results.loc[:, 'mode'] = mode
            model_results = pd.concat([model_results, mode_results], axis=0)

        # write cordon model results
        csv_name = '{}_{}_model_results.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        model_results.to_csv(csv_path)

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
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        classes_df.to_csv(csv_path)

        # Get cordon count for mode
        count = self.get_count(self.counts_df, modes=results_dfs.keys())
        count_df = self.count_to_df(count, 'benchmark')

        # Label and write benchmark csv
        benchmark_df = pd.concat([count_df, result_df]).groupby('source').sum()
        csv_name = '{}_{}_benchmark.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        benchmark_df.to_csv(csv_path)

        # Calc score
        return np.absolute(result - count) / count


class ModeStats:

    benchmark_path = None

    def __init__(self, name, config):
        """
        ModeStat parent object for benchmarking with mode share data.
        :param name: String
        :param config: Config
        """

        self.benchmark_df = pd.read_csv(self.benchmark_path,
                                        header=None,
                                        names=['mode', 'benchmark'])
        self.benchmark_df.set_index('mode', inplace=True)
        self.name = name
        self.config = config

    def output_and_score(self):
        """
        Builds paths for mode share outputs, loads and combines with model for scoring.
        :return: Dictionary of scores
        """
        # Build paths and load appropriate volume counts
        results_name = "{}_mode_shares_all_total.csv".format(self.config.name)
        results_path = os.path.join(self.config.output_path, results_name)
        results_df = pd.read_csv(results_path,
                                 header=None,
                                 names=['mode', 'model'])
        results_df.set_index('mode', inplace=True)

        # join results
        summary_df = self.benchmark_df.join(results_df, how='inner')
        summary_df.loc[:, 'diff'] = summary_df.model - summary_df.benchmark

        # write results
        csv_name = '{}_results.csv'.format(self.name)
        csv_path = os.path.join(self.config.output_path, 'benchmarks', csv_name)
        summary_df.to_csv(csv_path)

        # get scores and write outputs
        score = sum(np.absolute(np.array(summary_df.benchmark) - np.array(summary_df.model)))

        return {'modeshare': score}


class LondonInnerCordonCar(Cordon):

    cordon_counter = HourlyCordonCount
    benchmark_path = get_benchmark_data(os.path.join('london', 'inner_cordon', 'InnerCordon2016.csv'))
    cordon_path = get_benchmark_data(os.path.join('london', 'inner_cordon', 'cordon_links.csv'))

    directions = {'in': 1, 'out': 2}
    year = 2016
    hours = None
    modes = ['car']


class DublinCanalCordonCar(Cordon):

    cordon_counter = PeriodCordonCount
    benchmark_path = get_benchmark_data(os.path.join('ireland', 'dublin_cordon', '2016_counts.csv'))
    cordon_path = get_benchmark_data(os.path.join('ireland', 'dublin_cordon', 'dublin_cordon.csv'))

    directions = {'in': 1}
    year = 2016
    hours = [7, 8, 9]
    modes = ['car']


class IrelandCommuterStats(ModeStats):

    benchmark_path = get_benchmark_data(os.path.join('ireland', 'census_modestats', '2016_census_modestats.csv'))


####### test cordons for test_town #######

class TestTownHourlyCordon(Cordon):

    cordon_counter = HourlyCordonCount
    benchmark_path = get_benchmark_data(os.path.join('test_town', 'test_town_cordon', '2016_counts.csv'))
    cordon_path = get_benchmark_data(os.path.join('test_town', 'test_town_cordon', 'test_town_cordon.csv'))

    directions = {'in': 1, 'out': 2}
    year = 2016
    hours = None
    modes = ['car', 'bus']


class TestTownPeakIn(Cordon):

    cordon_counter = PeriodCordonCount
    benchmark_path = get_benchmark_data(os.path.join('test_town', 'test_town_peak_cordon', '2016_peak_in_counts.csv'))
    cordon_path = get_benchmark_data(os.path.join('test_town', 'test_town_peak_cordon', 'test_town_cordon.csv'))

    directions = {'in': 1}
    year = 2016
    hours = [7, 8, 9]
    modes = ['car', 'bus']

class TestTownCommuterStats(ModeStats):

    benchmark_path = get_benchmark_data(os.path.join('test_town', 'census_modestats', 'test_town_modestats.csv'))


# maps of benchmarks to Classes and weights for scoring
BENCHMARK_MAP = {"london_inner_cordon_car": LondonInnerCordonCar,
                 "dublin_canal_cordon_car": DublinCanalCordonCar,
                 "ireland_commuter_modeshare": IrelandCommuterStats,
                 "test_town_cordon": TestTownHourlyCordon,
                 "test_town_peak_cordon": TestTownPeakIn,
                 "test_town_modeshare": TestTownCommuterStats}

BENCHMARK_WEIGHTS = {"london_inner_cordon_car": 1,
                     "dublin_canal_cordon_car": 1,
                     "ireland_commuter_modeshare": 1,
                     "test_town_cordon": 1,
                     "test_town_peak_cordon": 1,
                     "test_town_modeshare": 1}


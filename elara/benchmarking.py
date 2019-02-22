import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


class Benchmarks:
    scores_df = None
    meta_score = None
    benchmarks = {}

    def __init__(self, config):

        self.config = config

        for cordon_name in config.benchmarks:
            self.benchmarks[cordon_name] = BENCHMARK_MAP[cordon_name](
                cordon_name, config)

    def score(self):
        scores = {}
        meta_score = 0
        for benchmark_name, benchmark in self.benchmarks.items():
            sub_scores = benchmark.output_and_score()
            scores[benchmark_name] = sub_scores

            # Combine with weights
            weight = BENCHMARK_WEIGHTS[benchmark_name]
            for s in sub_scores.values():
                meta_score += (s * weight)

        # Write scores
        scores_df = pd.DataFrame(scores).T
        csv_name = 'benchmark_scores.csv'
        csv_path = os.path.join(self.config.output_path, csv_name)
        scores_df.to_csv(csv_path)

        self.meta_score = meta_score
        self.scores_df = scores_df

        return meta_score


class InnerCordon:

    cordon_counts = []
    benchmark_path = os.path.join('benchmark_data', 'inner_cordon', 'InnerCordon2016.csv')
    map_path = os.path.join('benchmark_data', 'inner_cordon', 'cordon_links.csv')

    def __init__(self, name, config):
        self.config = config
        counts_df = pd.read_csv(self.benchmark_path)
        links_df = pd.read_csv(self.map_path, index_col=0)

        for direction_name, dir_code in {'in': 1, 'out': 2}.items():
            self.cordon_counts.append(CordonCount(
                name,
                config,
                direction_name,
                dir_code,
                counts_df,
                links_df
            ))

    def output_and_score(self):

        # Build paths and load appropriate volume counts
        # TODO volume counts are just car - need to add buses
        results_dfs = []
        for mode in self.config.handlers['volume_counts']:
            results_name = "{}_volume_counts_{}.csv".format(self.config.name, mode)
            results_path = os.path.join(self.config.output_path, results_name)
            results_df = pd.read_csv(results_path, index_col=0)
            results_df.index.name = 'link_id'
            select_cols = ['class'] + [str(i) for i in range(self.config.time_periods)]
            results_df = results_df.loc[:, select_cols]
            results_dfs.append(results_df)

        # get scores and write outputs
        scores = {}
        for cordon_count in self.cordon_counts:
            scores[cordon_count.direction] = cordon_count.output_and_score(results_dfs)
        return scores


class CordonCount:

    def __init__(self, name, config, direction_name, dir_code, counts_df, links_df):
        self.cordon_name = name
        self.direction = direction_name
        self.config = config

        # Get cordon links and counts
        self.link_ids = self.get_links(links_df, dir_code)
        self.counts_array = self.get_counts(counts_df, dir_code)
        self.count_df = self.counts_to_df(self.counts_array)

    def output_and_score(self, results_dfs):
        # collect all results
        concat_df = pd.DataFrame()
        for result_df in results_dfs:
            concat_df = pd.concat([concat_df, result_df.loc[result_df.index.isin(self.link_ids)]], axis=0)
        classes_df = concat_df.groupby('class').sum()

        # Build array for scoring
        results_array = np.array(classes_df.sum())

        # Label and write csv with counts by subpopulation
        classes_df['source'] = 'model'
        csv_name = '{}_{}_classes.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join(self.config.output_path, csv_name)
        classes_df.to_csv(csv_path)

        # Label and write benchmark csv
        benchmark_df = pd.concat([self.count_df, classes_df]).groupby('source').sum()
        csv_name = '{}_{}_benchmark.csv'.format(self.cordon_name, self.direction)
        csv_path = os.path.join(self.config.output_path, csv_name)
        benchmark_df.to_csv(csv_path)

        # Calc score
        return sum(np.absolute(results_array - self.counts_array)) / self.counts_array.sum()

    def get_links(self, links_df, direction_code):
        df = links_df.loc[links_df.dir == direction_code, :]
        return list(set(df.link))

    def get_counts(self, counts_df, direction_code):
        counts_array = np.zeros((self.config.time_periods))
        df = counts_df.loc[counts_df.Direction == direction_code, :]
        site_indexes = list(set(df.Site))
        for site_index in site_indexes:
            site_df = df.loc[df.Site == site_index, :]
            counts = np.array(site_df.sort_values('Hour').Total)
            assert len(counts) == self.config.time_periods
            counts_array += counts
        return counts_array

    def counts_to_df(self, array):
        col_names = [str(i) for i in range(self.config.time_periods)]
        df = pd.DataFrame(array, index=col_names).T
        df['source'] = 'count'
        return df


BENCHMARK_MAP = {"inner_cordon": InnerCordon}
BENCHMARK_WEIGHTS = {"inner_cordon": 1}

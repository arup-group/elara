import os
import geopandas
import pandas as pd
import logging
from matplotlib import pyplot as plt

from elara.factory import WorkStation, Tool


class PostProcessor(Tool):

    options_enabled = True

    def __init__(self, config, option=None):
        self.logger = logging.getLogger(__name__)
        super().__init__(config, option)

    @staticmethod
    def check_prerequisites():
        return NotImplementedError

    def build(self, resource, write_path=None):
        return NotImplementedError


class PlanTimeSummary(PostProcessor):
    """
    Process Leg Logs into Trip Logs into plan summary.
    """

    requirements = [
        'leg_logs',
    ]
    valid_options = ['all']

    hierarchy = None

    def __str__(self):
        return 'PlanTimeSummary'

    def build(self, resource: dict, write_path=None):
        super().build(resource, write_path=write_path)

        file_name = "leg_log_{}.csv".format(self.option)
        file_path = os.path.join(self.config.output_path, file_name)
        legs_df = pd.read_csv(file_path, index_col=0)

        file_name = "leg_activity_log_{}.csv".format(self.option)
        file_path = os.path.join(self.config.output_path, file_name)
        activity_df = pd.read_csv(file_path, index_col=0)

        leg_figure = self.plot_time_bins(legs_df, 'mode')
        leg_figure.suptitle("Travel Time Bins")
        leg_summary_df = legs_df.describe()

        act_figure = self.plot_time_bins(activity_df, 'act')
        act_figure.suptitle("Activity Time Bins")
        act_summary_df = activity_df.describe()

        # # Export results
        fig_name = "leg_summary_{}.png".format(self.option)
        self.write_png(leg_figure, fig_name, write_path=write_path)
        csv_name = "leg_summary_{}.csv".format(self.option)
        self.write_csv(leg_summary_df, csv_name, write_path=write_path)

        fig_name = "leg_activity_summary_{}.png".format(self.option)
        self.write_png(act_figure, fig_name, write_path=write_path)
        csv_name = "leg_activity_summary_{}.csv".format(self.option)
        self.write_csv(act_summary_df, csv_name, write_path=write_path)

    def time_binner(self, data):
        """
        Bin start and end times and durations, return freq table for 24 hour period, 15min intervals.
        """
        bins = list(range(0, 24*60*60+1, 15*60))
        bins[-1] = 100*60*60
        labels = pd.timedelta_range(start='00:00:00', periods=96, freq='15min')  
        binned = pd.DataFrame(index=pd.timedelta_range(start='00:00:00', periods=96, freq='15min'))
        binned['duration'] = pd.cut(data.duration_s, bins, labels=labels, right=False).value_counts()
        binned['end'] = pd.cut(data.end_s, bins, labels=labels, right=False).value_counts()
        binned['start'] = pd.cut(data.start_s, bins, labels=labels, right=False).value_counts()
        binned = binned / binned.max()
        return binned

    def plot_time_bins(self, data, sub_col):
    
        subs = set(data[sub_col])
        fig, axs = plt.subplots(len(subs), figsize=(12, len(subs)), sharex=True)
        
        for ax, sub in zip(axs, subs):

            binned = self.time_binner(data.loc[data[sub_col] == sub])
            ax.pcolormesh(binned.T, cmap='cool', edgecolors='white', linewidth=1)

            ax.set_xticks([i for i in range(0,97,8)])
            ax.set_xticklabels([f"{h:02}:00" for h in range(0,25,2)])
            ax.set_yticks([0.5,1.5,2.5])
            ax.set_yticklabels(['Duration', 'End time', 'Start time'])
            ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
            for pos in ['right','top','bottom','left']:
                ax.spines[pos].set_visible(False)
            ax.set_ylabel(sub.title(), fontsize='medium')
            
        return fig


class VKT(PostProcessor):
    requirements = ['volume_counts']
    valid_options = ['car', 'bus', 'train', 'subway', 'ferry']

    def __str__(self):
        return f'VKT PostProcessor mode: {self.option}'

    def build(self, resource: dict, write_path=None):
        super().build(resource, write_path=write_path)

        mode = self.option

        file_name = "volume_counts_{}.geojson".format(mode)
        file_path = os.path.join(self.config.output_path, file_name)
        volumes_gdf = geopandas.read_file(file_path)

        # Calculate VKT
        period_headers = generate_period_headers(self.config.time_periods)
        volumes = volumes_gdf[period_headers]
        link_lengths = volumes_gdf["length"].values / 1000  # Conversion to metres
        vkt = volumes.multiply(link_lengths, axis=0)
        vkt_gdf = pd.concat([volumes_gdf.drop(period_headers, axis=1), vkt], axis=1)

        csv_name = "vkt_{}.csv".format(mode)
        geojson_name = "vkt_{}.geojson".format(mode)

        self.write_csv(vkt_gdf, csv_name, write_path=write_path)
        self.write_geojson(vkt_gdf, geojson_name, write_path=write_path)


def generate_period_headers(time_periods):
    """
    Generate a list of strings corresponding to the time period headers in a typical
    data output.
    :param time_periods: Number of time periods across the day
    :return: List of time period numbers as strings
    """
    return list(map(str, range(time_periods)))


def export_geojson(gdf, path):
    """
    Given a geodataframe, export geojson representation to specified path.
    :param gdf: Input geodataframe
    :param path: Output path
    """
    with open(path, "w") as file:
        file.write(gdf.to_json())


class PostProcessWorkStation(WorkStation):

    tools = {
        'plan_summary': PlanTimeSummary,
        'vkt': VKT,
    }

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

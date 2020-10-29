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
        'agent_logs',
    ]
    valid_options = ['all']

    hierarchy = None

    def __str__(self):
        return 'PlanTimeSummary'

    def build(self, resource: dict, write_path=None):
        super().build(resource, write_path=write_path)

        file_name = f"leg_log_{self.option}.csv"
        file_path = os.path.join(self.config.output_path, file_name)
        legs_df = pd.read_csv(file_path, index_col=0)

        file_name = f"activity_log_{self.option}.csv"
        file_path = os.path.join(self.config.output_path, file_name)
        activity_df = pd.read_csv(file_path, index_col=0)

        leg_figure = self.plot_time_bins(legs_df, 'mode')
        leg_figure.suptitle("Travel Time Bins")
        leg_summary_df = legs_df.describe()

        act_figure = self.plot_time_bins(activity_df, 'act')
        act_figure.suptitle("Activity Time Bins")
        act_summary_df = activity_df.describe()

        # # Export results
        fig_name = f"leg_summary_{self.option}.png"
        self.write_png(leg_figure, fig_name, write_path=write_path)
        csv_name = f"leg_summary_{self.option}.csv"
        self.write_csv(leg_summary_df, csv_name, write_path=write_path)

        fig_name = f"activity_summary_{self.option}.png"
        self.write_png(act_figure, fig_name, write_path=write_path)
        csv_name = f"activity_summary_{self.option}.csv"
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


class AgentTripLogs(PostProcessor):
    """
    Process Leg Logs into Trip Logs.
    """

    requirements = [
        'agent_logs',
        'mode_hierarchy'
    ]
    valid_options = ['all']

    hierarchy = None

    def __str__(self):
        return f'AgentTripLogs modes'

    def build(self, resource: dict, write_path=None):
        super().build(resource, write_path=write_path)

        self.hierarchy = resource['mode_hierarchy']

        mode = self.option

        file_name = f"leg_log_{self.option}.csv"
        file_path = os.path.join(self.config.output_path, file_name)
        legs_df = pd.read_csv(file_path)

        # Group by trip and aggregate, applying hierarchy
        def combine_legs(gr):
            first_line = gr.iloc[0]
            last_line = gr.iloc[-1]

            # duration = sum(gr.duration)
            duration_s = sum(gr.duration_s)
            distance = sum(gr.distance)

            modes = list(gr.loc[:, 'mode'])
            primary_mode = self.hierarchy.get(modes)

            return pd.Series(
                {'mode': primary_mode,
                 'ox': first_line.ox,
                 'oy': first_line.oy,
                 'dx': last_line.dx,
                 'dy': last_line.dy,
                 'start': first_line.start,
                 'end': last_line.end,
                 # 'duration': duration,
                 'start_s': first_line.start_s,
                 'end_s': last_line.end_s,
                 'duration_s': duration_s,
                 'distance': distance
                 }
            )

        trips_df = legs_df.groupby(['agent', 'trip']).apply(combine_legs)

        # reset index
        trips_df.reset_index(inplace=True)

        # Export results
        csv_name = f"trip_logs_{mode}.csv"
        self.write_csv(trips_df, csv_name, write_path=write_path)


class VKT(PostProcessor):
    requirements = ['volume_counts']
    valid_options = ['car', 'bus', 'train', 'subway', 'ferry']

    def __str__(self):
        return f'VKT PostProcessor mode: {self.option}'

    def build(self, resource: dict, write_path=None):
        super().build(resource, write_path=write_path)

        mode = self.option

        file_name = f"volume_counts_{mode}.geojson"
        file_path = os.path.join(self.config.output_path, file_name)
        volumes_gdf = geopandas.read_file(file_path)

        # Calculate VKT
        period_headers = generate_period_headers(self.config.time_periods)
        volumes = volumes_gdf[period_headers]
        link_lengths = volumes_gdf["length"].values / 1000  # Conversion to metres
        vkt = volumes.multiply(link_lengths, axis=0)
        vkt_gdf = pd.concat([volumes_gdf.drop(period_headers, axis=1), vkt], axis=1)

        csv_name = f"vkt_{mode}.csv"
        geojson_name = f"vkt_{mode}.geojson"

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
        'trip_logs': AgentTripLogs,
        'vkt': VKT,
    }

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return f'PostProcessing WorkStation'

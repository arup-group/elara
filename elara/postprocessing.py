import os
import geopandas
import pandas as pd
import logging
from matplotlib import axes, pyplot as plt

from elara.factory import WorkStation, Tool


class PostProcessor(Tool):

    options_enabled = True

    def __init__(self, config, mode="all", groupby_person_attribute=None, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(config=config, mode=mode, groupby_person_attribute=groupby_person_attribute, **kwargs)

    def breakdown(self, data, bins, labels, colnames, write_path, groupby_field = None, groupby_data = None, scale = True):
        """
        Bin a data series and export to a csv file
        :params Pandas Series data: A numerical set
        :params list bins: data bins
        :params list labels: data labels
        :params list colnames: the column names of the output file
        :params str groupby_field: the attribute name of the attribute used to cross tabulate (e.g. mode or purpose)
        :params Pandas Series groupby_data: A categorical set to cross tabulate distance by
        :params bool, default True, scale outputs based on config scale

        :returns: None
        """
        # bin
        if groupby_field:
            breakdown_df = pd.concat([data, groupby_data, pd.cut(data, bins=bins, labels=labels)], axis=1)
            breakdown_df.columns = [colnames[1], groupby_field, colnames[0]]
            breakdown_df = breakdown_df.groupby(by=[groupby_field, colnames[0]]).count().sort_index().reset_index()
            csv_breakdown_name = f"{self.name}.csv".replace("all",groupby_field)
        else:
            breakdown_df = pd.cut(data, bins=bins, labels=labels).value_counts().sort_index().reset_index()
            breakdown_df.columns = colnames
            csv_breakdown_name = f"{self.name}.csv"

        # scale counts
        if scale:
            breakdown_df[colnames[1]] /= self.config.scale_factor

        # Export breakdown
        self.write_csv(breakdown_df, csv_breakdown_name, write_path=write_path, compression=self.compression)

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
    valid_modes = ['all']

    hierarchy = None

    def __str__(self):
        return 'PlanTimeSummary'

    def build(self, resource: dict, write_path=None):
        super().build(resource, write_path=write_path)

        file_name = f"leg_logs_{self.mode}_legs.csv"
        file_path = os.path.join(self.config.output_path, file_name)
        legs_df = pd.read_csv(file_path, index_col=0)

        file_name = f"leg_logs_{self.mode}_activities.csv"
        file_path = os.path.join(self.config.output_path, file_name)
        activity_df = pd.read_csv(file_path, index_col=0)

        leg_figure = self.plot_time_bins(legs_df, 'mode')
        leg_figure.suptitle("Travel Time Bins")
        leg_summary_df = legs_df.describe()

        act_figure = self.plot_time_bins(activity_df, 'act')
        act_figure.suptitle("Activity Time Bins")
        act_summary_df = activity_df.describe()

        # # Export results
        fig_name = f"{self.name}_legs.png"
        self.write_png(leg_figure, fig_name, write_path=write_path)
        csv_name = f"{self.name}_legs.csv"
        self.write_csv(leg_summary_df, csv_name, write_path=write_path, compression=self.compression)

        fig_name = f"{self.name}_activities.png"
        self.write_png(act_figure, fig_name, write_path=write_path)
        csv_name = f"{self.name}_activities.csv"
        self.write_csv(act_summary_df, csv_name, write_path=write_path, compression=self.compression)

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

        if len(subs) == 1:
            sub = list(subs)[0]
            fig, ax = plt.subplots(1, figsize=(12, 1))
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

        else:
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


class TripDurationBreakdown(PostProcessor):
    """
    Provide summary breakdowns of trip data:
        - by duration
        - by distance band
        - ...
    """
    requirements = ['trip_logs']
    valid_modes = ['all']

    def build(self, resource: dict, write_path=None):

        super().build(resource, write_path=write_path)

        # read trip logs
        file_name = f"trip_logs_{self.mode}_trips.csv"
        file_path = os.path.join(self.config.output_path, file_name)
        trips_df = pd.read_csv(file_path)

        cross_tab_dict = {"mode": trips_df["mode"],
                            "d_act": trips_df["d_act"],
                            None: None} #cross tabulate by mode, dest purpose of activity, all together

        # duration breakdown
        for key in cross_tab_dict:
            self.breakdown(
                data = trips_df.duration_s / 60,
                bins = [0, 5, 10, 15, 30, 45, 60, 90, 120, 999999],
                labels = ['0 to 5 min', '5 to 10 min', '10 to 15 min', '15 to 30 min', '30 to 45 min', '45 to 60 min', '60 to 90 min', '90 to 120 min', '120+ min'],
                colnames = ['duration', 'trips'],
                write_path = write_path,
                groupby_field = key,
                groupby_data = cross_tab_dict[key]
            )


class TripEuclidDistanceBreakdown(PostProcessor):
    """
    Provide summary breakdowns of trip distance with options to crosstabulate by
        - mode
        - purpose
    """
    requirements = ['trip_logs']
    valid_modes = ['all']

    def build(self, resource: dict, write_path=None):

        super().build(resource, write_path=write_path)
        mode = self.mode

        # read trip logs
        file_name = f"trip_logs_{self.mode}_trips.csv"
        file_path = os.path.join(self.config.output_path, file_name)
        trips_df = pd.read_csv(file_path)

        # euclidean distance breakdown
        trips_df['euclidean_distance'] = ((trips_df.ox - trips_df.dx) ** 2 + (trips_df.oy - trips_df.dy) ** 2) ** 0.5

        cross_tab_dict = {"mode": trips_df["mode"],
                            "d_act": trips_df["d_act"],
                            None: None} #cross tabulate by mode, dest purpose of activity, all together


        for key in cross_tab_dict:
            self.breakdown(
                data = trips_df.euclidean_distance / 1000,
                bins = [0, 1, 5, 10, 25, 50, 100, 200, 999999],
                labels = ['0 to 1 km', '1 to 5 km', '5 to 10 km', '10 to 25 km', '25 to 50 km', '50 to 100 km', '100 to 200 km', '200+ km'],
                colnames = ['euclidean_distance', 'trips'],
                write_path = write_path,
                groupby_field = key,
                groupby_data = cross_tab_dict[key]
        )

class VKT(PostProcessor):

    requirements = ['link_vehicle_counts']

    def __str__(self):
        return f'VKT PostProcessor mode: {self.mode}'

    def build(self, resource: dict, write_path=None):
        super().build(resource, write_path=write_path)

        if self.groupby_person_attribute:
            file_name = f"link_vehicle_counts_{self.mode}_{self.groupby_person_attribute}.geojson"
            file_path = os.path.join(self.config.output_path, file_name)
            volumes_gdf = geopandas.read_file(file_path)
            vkt_gdf = self.calculate_vkt(volumes_gdf)

            csv_name = f"{self.name}_{self.groupby_person_attribute}.csv"
            geojson_name = f"{self.name}_{self.groupby_person_attribute}.geojson"

            self.write_csv(vkt_gdf, csv_name, write_path=write_path, compression=self.compression)
            self.write_geojson(vkt_gdf, geojson_name, write_path=write_path)

        file_name = f"link_vehicle_counts_{self.mode}.geojson"
        file_path = os.path.join(self.config.output_path, file_name)
        volumes_gdf = geopandas.read_file(file_path)
        vkt_gdf = self.calculate_vkt(volumes_gdf)

        csv_name = f"{self.name}.csv"
        geojson_name = f"{self.name}.geojson"
        self.write_csv(vkt_gdf, csv_name, write_path=write_path, compression=self.compression)
        self.write_geojson(vkt_gdf, geojson_name, write_path=write_path)

    def calculate_vkt(self, link_volume_counts):
        """
        Calculate link vehicle kms from link volume counts dataframe.
        """
        period_headers = generate_period_headers(self.config.time_periods)
        link_lengths = link_volume_counts["length"].values / 1000  # Conversion to kilometres
        link_volume_counts[period_headers] = link_volume_counts[period_headers].multiply(link_lengths, axis=0)
        link_volume_counts["total"] = link_volume_counts[period_headers].sum(1) # create new total column
        return link_volume_counts

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
        #'trip_logs': AgentTripLogs,
        'trip_duration_breakdown': TripDurationBreakdown,
        'trip_euclid_distance_breakdown': TripEuclidDistanceBreakdown,
        'vkt': VKT,
    }

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

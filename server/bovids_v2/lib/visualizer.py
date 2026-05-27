__author__ = ["Max Hahn-Klimroth"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

from typing import Any, Dict, List, Tuple
from datetime import datetime, date



import pandas as pd
from matplotlib import pyplot as plt

from func import ensure_directory
from ..config.get_config import BEHAVIOR_COLOR_MAPPING, BEHAVIOR_VISUALIZATION_NAMES, BEHAVIOR_VISUALIZATION_CATEGORY

import plotly.express as px
import plotly.io as io


io.renderers.default = 'svg'



class NightVisualizer:
    """
    - method to plot gantt chart of the behaviors during one night
    """

    # Überschrift: Tiername Datum
    @staticmethod
    def _create_gantt_figure(
        df: pd.DataFrame, x_ticks: List[datetime], x_tick_labels: List[str], title: str, categories: List[str], total: bool
    ):
        # categories for y-axis
        categories = [BEHAVIOR_VISUALIZATION_NAMES[cat] for cat in categories]

        plt.cla()
        plt.clf()

        # create timeline plot from dataframe with the behaviors
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="Resource",
            color="Resource",
            color_discrete_map=BEHAVIOR_COLOR_MAPPING,
            title=title,
            width=1000,
            height=400,
        )

        # update layout for colours and y-axis category order
        fig.update_layout(
            {
                "plot_bgcolor": "rgba(255, 255, 255, 1)",
                "paper_bgcolor": "rgba(255, 255, 255, 1)",
                "yaxis": {
                    "categoryorder": "array",
                    "categoryarray": categories[::-1],
                    "title": "",
                },
            }
        )

        # make a separation line in the plot with all behaviors (total)
        if total:
            total_categories = len(categories)  # number of categories
            top_y = total_categories - 3.5 # position of separation line
            fig.add_shape(
                type="line",
                x0=df["Start"].min(),
                x1=df["Finish"].max(),
                y0=top_y,
                y1=top_y,
                line=dict(color="black", width=2),
            )

        fig.update_layout(showlegend=False) # no legend

        # update x-axis (timeline)
        fig.update_xaxes(
            title="time [h]",
            tickangle=45,
            tickmode="array",
            tickvals=x_ticks,
            ticktext=x_tick_labels,
        )

        return fig


    @staticmethod
    def plot_nocturnal_sequence(
        plot_stly: Dict[str, List[Tuple[datetime, datetime]]],
        plot_subactions: Dict[str, List[Tuple[datetime, datetime]]],
        plot_total: Dict[str, List[Tuple[datetime, datetime]]],
        behaviors_to_plot: List[str],
        behaviors_to_plot_sub: List[str],
        savepath_image_total: str,
        savepath_image_binary: str,
        savepath_image_subactions: str,
        individual_plot_name: str,
        plot_start_datetime: datetime,
        plot_end_datetime: datetime,
        bool_subactions: bool,
    ) -> None:
        """
        Outputs two gantt charts showing the nocturnal behavior sequence
            - one with all behaviors contained in behaviors_to_plot
            - one binary version (only standing and lying and out)
        Ordering on y-axes follows behaviors_to_plot
        """
        # set x tick labels
        x_ticks: List[datetime] = pd.date_range(
            start=plot_start_datetime,
            end=plot_end_datetime,
            periods=int((plot_end_datetime - plot_start_datetime).total_seconds() // 3600)+1,
        ).to_list()
        # set y tick labels
        x_tick_labels: List[str] = [x.strftime("%H:%M") for x in x_ticks]
        start_date: date = date(
            plot_start_datetime.year, plot_start_datetime.month, plot_start_datetime.day
        )

        # set plot title
        title: str = f"{individual_plot_name} - {start_date.strftime('%d.%m.%Y')}"

        # generate binary data using standing, lying and out of view
        binary_plot_data: List[Dict[str, Any]] = []
        for behavior in ["Standing", "Lying", "Out of View"]:
            for start, end in plot_stly[behavior]:
                binary_plot_data.append(
                    dict(
                        Task=BEHAVIOR_VISUALIZATION_NAMES[behavior],
                        Start=start,
                        Finish=end,
                        Resource=BEHAVIOR_VISUALIZATION_NAMES[behavior],
                        CategoryOrder=BEHAVIOR_VISUALIZATION_CATEGORY[behavior],
                    )
                )

        df_binary: pd.DataFrame = pd.DataFrame(binary_plot_data)
        # create figure that contains the actions standing, lying and out of view (binary figure)
        binary_fig = NightVisualizer._create_gantt_figure(
            df_binary, x_ticks, x_tick_labels, title, ["Out of View", "Standing", "Lying"], False
        )
        # safe binary figure
        ensure_directory(savepath_image_binary)
        binary_fig.write_image(savepath_image_binary)
        #io.write_image(binary_fig, savepath_image_binary)

        if bool_subactions:
            # generate subactions data plot
            sub_plot_data: List[Dict[str, Any]] = []
            for behavior in plot_subactions:
                # only actions observed in the used video file are used for the subactions plot
                if behavior not in behaviors_to_plot_sub:
                    continue
                for start, end in plot_subactions[behavior]:
                    sub_plot_data.append(
                        dict(
                            Task=BEHAVIOR_VISUALIZATION_NAMES[behavior],
                            Start=start,
                            Finish=end,
                            Resource=BEHAVIOR_VISUALIZATION_NAMES[behavior],
                            CategoryOrder=BEHAVIOR_VISUALIZATION_CATEGORY[behavior],
                        )
                    )
            df_total: pd.DataFrame = pd.DataFrame(sub_plot_data)
            # create figure containing all subactions
            sub_fig = NightVisualizer._create_gantt_figure(
                df_total, x_ticks, x_tick_labels, title, behaviors_to_plot_sub, False
            )

            # save subactions figure
            ensure_directory(savepath_image_subactions)
            sub_fig.write_image(savepath_image_subactions)

            # generate total data plot
            total_plot_data: List[Dict[str, Any]] = []
            for behavior in plot_total:
                # only actions observed in the used video file are used for the subactions plot
                if behavior not in behaviors_to_plot:
                    continue
                for start, end in plot_total[behavior]:
                    total_plot_data.append(
                        dict(
                            Task=BEHAVIOR_VISUALIZATION_NAMES[behavior],
                            Start=start,
                            Finish=end,
                            Resource=BEHAVIOR_VISUALIZATION_NAMES[behavior],
                            CategoryOrder=BEHAVIOR_VISUALIZATION_CATEGORY[behavior],
                        )
                    )

            df_total: pd.DataFrame = pd.DataFrame(total_plot_data)
            # create figure containing all actions
            total_fig = NightVisualizer._create_gantt_figure(
                df_total, x_ticks, x_tick_labels, title, behaviors_to_plot, True
            )

            # save total figure
            ensure_directory(savepath_image_total)
            total_fig.write_image(savepath_image_total)



class IndividualVisualizer:
    """
    - method to output the activity budget of the individual
    - method to output the activity budget over time of one individual (binary and "total")
    - method to output the distribution of lying cycle length
    - method to output the proportion lying from the first time of lying down (i.e., ACF)
    """

    @staticmethod
    def plot_activity_budget():
        pass

    @staticmethod
    def plot_timeline():
        pass

    @staticmethod
    def plot_lying_aligned_timeline():
        pass

    @staticmethod
    def calculate_individual_overview():
        """
        - concatenate single nights, add individual specific details (e.g., age, sex, ...) as rows
        - output also a sheet with an overview: median + qd, mean + sd per behavior class and phase length (standing, lying, lhd, food, moving), and lying cycle length
        """
        pass


class SpeciesVisualizer:
    """
    - method to output the average activity of the species (by time), divided into individuals by age (binary and total)
    - activity budget divided into individuals by age (bar plot + sem)
    - method to analyse lying cycles -> median of adult individuals, how many short phases per individual -> plot
    --> also output some excel file containing the values
    """

    @staticmethod
    def calculate_species_overview():
        """
        - concatenate individual overview sheets, add species specific details as rows
        - output also an overview sheet: median/mean qd/sd of the median/mean values per individual, distinguished by age
        """
        pass

    @staticmethod
    def plot_one_activity_budget():
        """
        - plot activity budget given the input information
        """
        pass

    @staticmethod
    def plot_all_activity_budgets():
        """
        given the species overview information output activity budgets per age class
        """
        pass


__author__ = ["Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

from typing import List
import pandas as pd
import numpy as np
from server.bovids_v2.config.get_config import  BEHAVIOR_STATISTICS
import datetime
import time

from openpyxl.styles.builtins import percent


def create_phase_csv(df_pp_info: pd.DataFrame, date:str, individual:str, path_phases:str):
    """
    Create .csv file containing all phases including information about: date, individual, start, end, duration,
    behavior and behavior count
    """
    # für beide listen nutzbar, anderer df als input

    # lists to save all information for one phase list
    behavior  = []
    start = df_pp_info["start"].tolist()
    end = df_pp_info["end"].tolist()
    duration = df_pp_info["seq_duration"].tolist()
    date = [date] * len(df_pp_info)
    individual = [individual] * len(df_pp_info)
    behavior_count = []

    # count occurence of each behavior
    behav_count = {behav: 0 for behav in BEHAVIOR_STATISTICS.keys()}

    # iterate over behavior sequence and append lists
    for action in df_pp_info.index:
        behavior.append(BEHAVIOR_STATISTICS[df_pp_info["seq_behavior"][action]])
        behav_count[df_pp_info["seq_behavior"][action]] += 1
        behavior_count.append(behav_count[df_pp_info["seq_behavior"][action]])

    # create a dataframe from all lists
    phases_list = pd.DataFrame(
        {
            "date": date,
            "individual": individual,
            "start": start,
            "end": end,
            "duration": duration,
            "behavior": behavior,
            "behavior_count": behavior_count,
        }
    )

    # save phases list as csv
    phases_list.to_csv(path_phases, index=False)

    return phases_list


def get_cycles_behavior(df_pp_info: pd.DataFrame, date:str, individual:str, behav:int):
    """
    Create a list with the information for all cycles of a specified behavior.
    """
    # lists to save all information for one behavior
    behavior = []
    start = []
    end = []
    duration = []
    date_list = []
    individual_list = []
    perc_standing = []
    perc_lying = []
    cycles = []

    # count duration length for one cycle
    duration_behavior = 0
    duration_standing = 0
    duration_lying = 0
    cycle_count = 0

    # detect if first cycle started
    first_cycle = False
    help_bool = True

    # iterate over behavior sequence
    for action in df_pp_info.index:
        if df_pp_info["seq_behavior"][action] != behav and df_pp_info["seq_behavior"][action] != 0:
            help_bool = True
        # only append lists for correct behavior
        if df_pp_info["seq_behavior"][action] == behav and help_bool:
            date_list.append(date)
            individual_list.append(individual)
            if behav == 1:
                behavior.append("Standing")
            elif behav == 2:
                behavior.append("Lying")
            # only append lists, if first cycle started
            help_bool = False
            if first_cycle:
                duration.append(str(datetime.timedelta(seconds=duration_behavior)))
                end.append(df_pp_info["end"][action - 1])
                duration_behavior = 0  # reset counter
                start.append(df_pp_info["start"][action])
                perc_standing.append(round(duration_standing / (duration_standing + duration_lying), 3) * 100)
                perc_lying.append(round(duration_lying / (duration_standing + duration_lying), 3) * 100)
                duration_standing = 0
                duration_lying = 0
                cycle_count += 1
                cycles.append(cycle_count)
                #help_bool = False

            # detect that first cycle started
            else:
                start.append(df_pp_info["start"][action])
                first_cycle = True
        # count behavior duration if first cycle started
        if first_cycle:
            x = time.strptime(
                df_pp_info["seq_duration"][action].split(",")[0], "%H:%M:%S"
            )
            seconds = datetime.timedelta(
                hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec
            ).total_seconds()
            duration_behavior += seconds
            if df_pp_info["seq_behavior"][action] == 1:
                duration_standing += seconds
            elif df_pp_info["seq_behavior"][action] == 2:
                duration_lying += seconds

    # if there are no cycles, return empy lists
    if not first_cycle:
        return [date_list, individual_list, behavior, start, end, duration, perc_standing, perc_lying, cycles]

    # append values for last cycle
    end.append(df_pp_info["end"].iloc[-1])
    duration.append(str(datetime.timedelta(seconds=duration_behavior)))

    perc_standing.append(round(duration_standing / (duration_standing + duration_lying), 3) * 100)
    perc_lying.append(round(duration_lying / (duration_standing + duration_lying), 3) * 100)
    cycle_count += 1
    cycles.append(cycle_count)

    # save all info in one list
    behavior_list = [
        date_list,
        individual_list,
        behavior,
        start,
        end,
        duration,
        perc_standing,
        perc_lying,
        cycles
    ]


    return behavior_list


def create_cycle_csv(standing_list: List[List], lying_list: List[List], start_action: int, second_action: int,
                     path_stats: str):
    """
    Input Variables
    standing_list: contains individual lists for each information of all standing cycles (date, individual, behavior,
        start, end, duration, percentage standing, percantage lying, amount cycles)
    lying_list: contains individual lists for each information of all lying cycles (date, individual, behavior,
        start, end, duration, percentage standing, percantage lying, amount cycles)

    Create .csv file containing all cycles including information about: date, individual, start, end, duration,
    behavior, percentage standing, percentage lying and cycle count
    """

    # combine both lists into one cycles list


    # lists to save all information for both behaviors
    behavior = []
    start = []
    end = []
    duration = []
    date_list = []
    individual_list = []
    perc_standing = []
    perc_lying = []
    cycles = []

    # set if standing or lying is first cycle to appear
    if start_action == 1 or (start_action == 0 and second_action == 1):
        # set length of longer list (list of the first cycle) to iterate over it
        iteration_length = len(standing_list[0])
        list1 = standing_list
        list2 = lying_list
    elif start_action == 2 or (start_action == 0 and second_action == 2):
        # set length of longer list (list of the first cycle) to iterate over it
        iteration_length = len(lying_list[0])
        list1 = lying_list
        list2 = standing_list


    # iterate over the two separate lists to combine them
    for i in range(iteration_length):
        #if
        # append values from list of first action
        date_list.append(list1[0][i])
        individual_list.append(list1[1][i])
        behavior.append(list1[2][i])
        start.append(list1[3][i])
        end.append(list1[4][i])
        duration.append(list1[5][i])
        perc_standing.append(list1[6][i])
        perc_lying.append(list1[7][i])
        cycles.append(list1[8][i])
        # append values from list of second action
        if i == (iteration_length-1) and len(list1[0]) != len(list2[0]):

            break
        else:
            date_list.append(list2[0][i])
            individual_list.append(list2[1][i])
            behavior.append(list2[2][i])
            start.append(list2[3][i])
            end.append(list2[4][i])
            duration.append(list2[5][i])
            perc_standing.append(list2[6][i])
            perc_lying.append(list2[7][i])
            cycles.append(list2[8][i])

    # save all lists into a dataframe
    cycles_list = pd.DataFrame(
        {
            "date": date_list,
            "individual": individual_list,
            "start": start,
            "end": end,
            "duration": duration,
            "behavior": behavior,
            "percent_standing": perc_standing,
            "percent_lying": perc_lying,
            "cycle_count": cycles,
        }
    )

    cycles_list.to_csv(path_stats, index=False)

    return cycles_list


def create_key_values_csv(cycles: pd.DataFrame, phases: pd.DataFrame, subactions: bool, standing_durations: List[str],
                          lying_durations: List[str], key_values_file: str):
    """
    Input Variables
    cycles: dataframe containing date, individual, start, end, duration, behavior, percentage standing,
    percentage lying and cycle count
    phases: dataframe containing date, individual, start, end, duration, behavior and behavior count

    Create .csv file contatining count and median values for phases and cycles
    - Date
    - Individual
    - Phase Standing Count
    - Phase Lying Count
    - Phase Standing No Food Count
    - Phase Standing Food Count
    - Phase LyingHead Up Count
    - Phase Lying Head Down Count
    - Phase Standing Median
    - Phase Lying Median
    - Phase Standing No Food Median
    - Phase Standing Food Median
    - Phase Lying Head Up Median
    - Phase Lying Head Down Median
    - Cycle Standing Count
    - Cycle Lying Count
    - Cycle Standing Median
    - Cycle Lying Median

    """



    # dict to find the last behavior of each type
    last_behavior = {behav: False for behav in BEHAVIOR_STATISTICS.values()}

    # dict to safe the count of each behavior
    count_behavior = {behav: 0 for behav in BEHAVIOR_STATISTICS.values()}

    # dict to sum up the duration of each behavior
    behavior_duration = {behav: 0 for behav in BEHAVIOR_STATISTICS.values()}

    # dict to safe the percentage per night of each behavior
    percentage_behavior = {}

    # determine the duration for each phase
    for phase in range(len(phases)):
        count_behavior[phases.loc[phase, "behavior"]] += 1
        duration = phases.loc[phase, "duration"]
        x = time.strptime(duration.split(",")[0], "%H:%M:%S")
        seconds = datetime.timedelta(
            hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec
        ).total_seconds()
        behavior_duration[phases.loc[phase, "behavior"]] += seconds

    # calculate median duration for each phase and save in result
    result = {}
    for key in behavior_duration:
        if count_behavior[key] != 0:
            median =  behavior_duration[key] / count_behavior[key]
            minutes = "{}".format(str(datetime.timedelta(seconds=median)))
            result[key] = minutes
        else:
            result[key] = 0

    # percentage behavior - nigth length
    evaluation_start = phases.loc[0, "start"]
    start = time.strptime(evaluation_start.split(",")[0], "%H:%M:%S")
    seconds_start = datetime.timedelta(
        hours=start.tm_hour, minutes=start.tm_min, seconds=start.tm_sec
    ).total_seconds()
    evaluation_end = phases.iloc[-1]["end"]
    end = time.strptime(evaluation_end.split(",")[0], "%H:%M:%S")
    seconds_end = datetime.timedelta(
        hours=end.tm_hour, minutes=end.tm_min, seconds=end.tm_sec
    ).total_seconds()
    if (seconds_end - seconds_start) > 0:
        evaluation_duration_seconds = seconds_end - seconds_start
    else:
        evaluation_duration_seconds = (24*3600) + (seconds_end - seconds_start)


    for behav in behavior_duration:
        percentage_behav = round(behavior_duration[behav] / evaluation_duration_seconds * 100, 1)
        percentage_behavior[behav] = percentage_behav



    # calculate key values for cycles

    # check if there are cycles
    if len(cycles) > 0:

        # find last standing cycle to determin standing count value
        if cycles.iloc[-1]["behavior"] == "Standing":
            cycle_standing = cycles.iloc[-1]["cycle_count"]

            if len(cycles) > 1:
                cycle_lying = cycles.iloc[-2]["cycle_count"]
            # if only one cycle is available set cycle_ying to 0
            else:
                cycle_lying = 0

        # find last lying cycle to determin lying count value
        else:
            cycle_lying = cycles.iloc[-1]["cycle_count"]
            if len(cycles) > 1:
                cycle_standing = cycles.iloc[-2]["cycle_count"]
            # if only one cycle is available set cycle_ying to 0
            else:
                cycle_standing = 0

        # determine duration of standing cycle and calculate median duration
        standing_durations_seconds = []
        for i in standing_durations:
            x = time.strptime(i.split(",")[0], "%H:%M:%S")
            seconds = datetime.timedelta(
                hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec
            ).total_seconds()
            standing_durations_seconds.append(seconds)
        # check if there is a standing cycle
        if len(cycles) == 1 and len(standing_durations_seconds) == 0:
            median_standing_cylce = 0  # median is zero, when there is no standing cycle
        else:
            median_standing_cylce = np.median(standing_durations_seconds)
            median_standing_cylce = "{}".format(str(datetime.timedelta(seconds=median_standing_cylce)))

        # determine duration of lying cycle and calculate median duration
        lying_durations_seconds = []
        for i in lying_durations:
            x = time.strptime(i.split(",")[0], "%H:%M:%S")
            seconds = datetime.timedelta(
                hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec
            ).total_seconds()
            lying_durations_seconds.append(seconds)
        # check if there is a lying cycle
        if len(cycles) == 1 and len(lying_durations_seconds) == 0:
            median_lying_cylce = 0 # median is zero, when there is no lying cycle
        else:
            median_lying_cylce = np.median(lying_durations_seconds)
            median_lying_cylce = "{}".format(str(datetime.timedelta(seconds=median_lying_cylce)))

    # set cycle values to zero, when there are no cycles
    else:
        cycle_standing = 0
        cycle_lying = 0
        median_standing_cylce = 0
        median_lying_cylce = 0


    # save key values as .csv file
    key_values = pd.DataFrame(columns=["date", "individual", "phase/cycle/behavior", "behavior", "count/median/duration/percentage", "value"])
    date = cycles.iloc[0]["date"]
    individual = cycles.iloc[0]["individual"]
    for behavior in BEHAVIOR_STATISTICS.values():
        key_values.loc[len(key_values)] = [date, individual, "Phase", behavior, "Count", count_behavior[behavior]]
        key_values.loc[len(key_values)] = [date, individual, "Phase", behavior, "Median", result[behavior]]
        key_values.loc[len(key_values)] = [date, individual, "Behavior", behavior, "Duration", behavior_duration[behavior]]
        key_values.loc[len(key_values)] = [date, individual, "Behavior", behavior, "Percentage",
                                           percentage_behavior[behavior]]
    key_values.loc[len(key_values)] = [date, individual, "Cycle", "Standing", "Count", cycle_standing]
    key_values.loc[len(key_values)] = [date, individual, "Cycle", "Lying", "Count", cycle_lying]
    key_values.loc[len(key_values)] = [date, individual, "Cycle", "Standing", "Median", median_standing_cylce]
    key_values.loc[len(key_values)] = [date, individual, "Cycle", "Lying", "Median", median_lying_cylce]

    sorted_df = key_values.sort_values(by=["phase/cycle/behavior", "count/median/duration/percentage"])

    sorted_df.to_csv(key_values_file, index=False)


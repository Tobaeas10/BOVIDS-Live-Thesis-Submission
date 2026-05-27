__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from server.bovids_v2.config.get_config import BEHAVIORS_BY_MODE, BEHAVIORS_BY_MODE_2
import datetime

# TODO: Regeln für "Out"-Anteil formulieren und in der Excel-Datei abbilden, zum Beispiel "20% Out in 80% Stehen werden ignoriert, wenn eine einzelne Out-Phase nicht länger als xyz ist."



class PostProcessor:

    rule_set_time: Dict[Tuple[int, int, int], int] = {}
    # (1,2,1): 5: if lying sequence is shorter than 5 minutes, 'ignore' it
    rule_set_behavior: Dict[Tuple[int, int, int], int] = {}
    # (1,2,1): 1: if lying sequence is shorter than given above, cast to standing
    rules_out_correction: Dict = {}

    # lists to stor the original time and behavior sequence and after filtering
    original_behavior_sequence: List[int] = []
    modified_behavior_sequence: List[int] = []
    original_time_sequence: List[int] = []
    modified_time_sequence: List[int] = []

    log_messages: List[str] = []

    def __init__(
        self,
        rule_set: pd.DataFrame,
        rule_name: str,
        seconds_per_interval: int,
        behavior_sequence: List[int],
    ):
        self.original_behavior_sequence, self.original_time_sequence = (
            self.cast_interval_sequence_to_time_behavior_sequence(
                behavior_seq=behavior_sequence,
                seconds_per_interval=seconds_per_interval,
            )
        )

        self.rule_set_time, self.rule_set_behavior = self.set_ruleset(
            rule_name, rule_set
        )

    def ensure_behavior_tuple(self, rule_behavior: dict, behavior_tuple: tuple):
        """
        Method to check for each behavior tupel, if there is a rule.
        If there is not one, this tupel can be skipped.
        """
        # checks if behavior sequence is in rule set
        if behavior_tuple in rule_behavior:
            return True
        else:
            self.log_messages.append("ERROR: Missing rule.")
            return False


    def cast_interval_sequence_to_time_behavior_sequence(
        self, behavior_seq: List[int], seconds_per_interval: int
    ) -> (List[int], List[int]):
        """
        Cast consecutive identical behaviors into one behavior with the according summed time.
        Returns summed time and behavior sequences as lists.
        """

        # create lists to save casted time and behavior
        ret_behavior_sequence: List[int] = []
        ret_time_sequence: List[int] = []

        # check if input is available
        if len(behavior_seq) == 0:
            self.log_messages.append("ERROR: Input behavior sequence is empty.")
            return ret_behavior_sequence, ret_time_sequence

        # initialize current help variables for iteration
        curr_behavior: int = behavior_seq[0]
        curr_count: int = 1

        # iterate over behavior sequence
        for interval_behavior in behavior_seq[1:]:
            # summed time periods for consecutive identical behaviors
            if interval_behavior == curr_behavior:
                curr_count += 1
                continue
            # add summed time and current behavior to return list
            ret_behavior_sequence.append(curr_behavior)
            ret_time_sequence.append(curr_count * seconds_per_interval)
            # update current help variables
            curr_behavior = interval_behavior
            curr_count = 1

        # add summed time and current behavior to return list
        ret_behavior_sequence.append(curr_behavior)
        ret_time_sequence.append(curr_count * seconds_per_interval)

        return ret_behavior_sequence, ret_time_sequence

    def set_ruleset(
        self, rule_name: str, rule_set: pd.DataFrame
    ) -> (Dict[Tuple[int, int, int], int], Dict[Tuple[int, int, int], int]):

        """
        Convert time rules from 'post_processing_rules.xlsx' into ruleset,
        that is used to post process the behavior sequence.
        Returns time and behavior rules as dictionaries.
        """

        # check if ruleset is available
        if rule_set.empty:
            self.log_messages.append("ERROR: Dataframe is empty.")
            return {}, {}

        # dictionary containing: key: given behavior tuple, value: corresponding time rule
        time_dict: dict = {}
        # dictionary containing: key: given behavior tuple, value: corrected behavior
        behavior_dict: dict = {}

        # for each row in the rule dataframe
        for row in rule_set.index:
            # determine previous, current and next behavior and save as tuple
            previous: int = rule_set.loc[row, "previous"]
            current: int = rule_set.loc[row, "current"]
            next: int = rule_set.loc[row, "next"]
            behavior_tuple: tuple = tuple((previous, current, next))
            # determine rule by time and corrected behavior
            current_corrected: int = rule_set.loc[row, "current_corrected"]
            time: int = rule_set.loc[row, rule_name]

            # fill dictionary with behavior tuple and rule
            time_dict[behavior_tuple] = time
            behavior_dict[behavior_tuple] = current_corrected

        return time_dict, behavior_dict

    def filter_short_phases(
        self,
        behavior_seq: List[int],
        time_seq: List[int],
        rule_time: Dict[Tuple[int, int, int], int],
        rule_behavior: Dict[Tuple[int, int, int], int],
    ) -> (List[int], List[int]):

        """
        Phases that are shorter than the ruleset allows, get replaced by the correct behavior.
        Returns corrected behavior and time sequence as lists

        """

        # determines the current behavior in the behavior list
        index: int = 1

        # iterate over behavior sequence to modify behavior and time list so that short phases are removed
        if len(behavior_seq) > 2:
            while True:
                # create behavior tuple, that can be used to compare with rule dictionary
                behavior_tuple: tuple = tuple(
                    (
                        behavior_seq[index - 1],
                        behavior_seq[index],
                        behavior_seq[index + 1],
                    )
                )

                ensured_tuple = self.ensure_behavior_tuple(
                    rule_behavior, behavior_tuple
                )
                if not (ensured_tuple):
                    index += 1
                    # break at the end of the behavior sequence
                    if index >= len(behavior_seq) - 1:
                        break
                    continue

                # if rule from ruleset can be applied
                if time_seq[index] <= rule_time[behavior_tuple]:
                    # if rule includes two identical behaviors as previous and next
                    if behavior_tuple[0] == behavior_tuple[2]:
                        # remove behaviors that are not needed from behavior list
                        behavior_seq.pop(index + 1)
                        behavior_seq.pop(index)
                        # add up all time periods
                        time_seq[index - 1] = (
                            time_seq[index - 1] + time_seq[index] + time_seq[index + 1]
                        )
                        # remove time periods that are not needed from time list
                        time_seq.pop(index + 1)
                        time_seq.pop(index)
                    # if rule includes two different behaviors as previous and next
                    else:
                        # remove behavior at index position that is not needed
                        behavior_seq.pop(index)
                        if rule_behavior[behavior_tuple] == behavior_tuple[0]:
                            # addup time periods
                            time_seq[index - 1] = time_seq[index - 1] + time_seq[index]
                            # remove time period that is not needed from time list
                            time_seq.pop(index)
                        elif rule_behavior[behavior_tuple] == behavior_tuple[2]:
                            # addup time periods
                            time_seq[index + 1] = time_seq[index] + time_seq[index + 1]
                            # remove time periods that are not needed from time list
                            time_seq.pop(index)
                        index += 1
                # if no rule from ruleset can be applied
                else:
                    index += 1

                # break at the end of the behavior sequence
                if index >= len(behavior_seq) - 1:
                    break

        return behavior_seq, time_seq

    def save_post_processed_sequence(
        self,
        behavior_seq: List[int],
        time_seq: List[int],
        output_path_pp: str,
        output_path_pp_info: str,
        eval_start: str
    ):
        """
        Method to safe post processed information as .csv files.
        The first file contains only sequence information for further usage.
        The second file contains additional time information for user.
        
        """

        # file that contains only the post processed sequence with behavior and duration
        pp_seq_df = pd.DataFrame(
            {"seq_behavior": behavior_seq, "seq_duration": time_seq}
        )
        pp_seq_df.to_csv(output_path_pp, index=False)

        # lists to store the timepoints and durations for the behavior sequence
        list_timepoints = []
        list_durations = []

        # create lists with timepoint (in seconds) and duration of each behavior change
        current_timepoint = int(eval_start) * 3600
        list_timepoints.append(current_timepoint)
        for duration in time_seq:
            current_timepoint += duration
            list_timepoints.append(current_timepoint)
            list_durations.append(str(datetime.timedelta(seconds=int(duration))))

        # convert seconds into corresponding time
        for timepoint in range(len(list_timepoints)):
            # check if date line is crossed
            if list_timepoints[timepoint] >= 24 * 3600: # If time is exactly midnight, it used to create a chain of errors, so I made it greater OR EQUAL
                list_timepoints[timepoint] -= 24 * 3600
            list_timepoints[timepoint] = str(datetime.timedelta(seconds=int(list_timepoints[timepoint])))

        # determine start and end timepoint for each behavior
        list_start = list_timepoints[: len(list_timepoints) - 1]
        list_end = list_timepoints[1: len(list_timepoints)]

        # file that contains additional information for each entry
        pp_info_df = pd.DataFrame(
            {
                "seq_behavior": behavior_seq,
                "start": list_start,
                "end": list_end,
                "seq_duration": list_durations,
            }
        )
        pp_info_df.to_csv(output_path_pp_info, index=False)


class PostProcessorSubactions:
    
    # lists to save behavior and time and behavior sequence for StLy, StFo and LHULHD
    stly_behavior: List[int] = []
    stly_time: List[int] = []
    stfo_behavior: List[int] = []
    stfo_time: List[int] = []
    lhulhd_behavior: List[int] = []
    lhulhd_time: List[int] = []

    def __init__(self, stly_behavior, stly_time, interval):
        self.stly_behavior, self.stly_time = self.cast_post_processed_sequence_into_interval_sequence(
            stly_behavior, stly_time, interval
        )

    def cast_post_processed_sequence_into_interval_sequence(
        self, pp_stly_behavior: List[int], pp_stly_time: List[int], interval: int
    ) -> (List[int], List[int]):
        """
        Cast StLy sequence to get a behavior for each interval.
        Returns casted StLy behavior and time as lists
        """

        #lists to save new behavior and time sequence
        stly_behavior = []
        stly_time = []

        # extract behavior for each time interval
        for period in range(len(pp_stly_time)):

            for i in range(int(pp_stly_time[period] / interval)):
                stly_behavior.append(pp_stly_behavior[period])
                stly_time.append(interval)

        return stly_behavior, stly_time

    def incorporate_subactions_sequence(
        self, stly_seq: List[int], stfo_seq: List[int], lhulhd_seq: List[int]
    ):
        """
        Incorporate the corresponding subactions into the behavior sequence.
        Returns casted behavior sequence as list.
        """

        casted = []
        for i in range(len(stly_seq)):
            # if behavior is standing then the corresponding subactions (Food or No Food) get incorporated into behhavior sequence
            if stly_seq[i] == BEHAVIORS_BY_MODE["StLy"]["Standing"]:
                if len(stfo_seq) != 0:
                    # if we have Standing (because of filtering) but no StFo information
                    # TODO: FInd out why causes error on stream service usage sometimes on low batch sizes.
                    if np.isnan(stfo_seq[i]):
                        # if subaction before and after are identical, use this subaction
                        if i > 0 and i < len(stfo_seq) - 1 and stfo_seq[i - 1] == stfo_seq[i + 1]:
                            casted.append(stfo_seq[i - 1])
                        # if subaction before and after are not identical, use No Food
                        else:
                            casted.append(BEHAVIORS_BY_MODE["StFo"]["Standing_no_food"])
                    # if we have StFo information
                    else:
                        casted.append(stfo_seq[i])
                # if we have no StFo information
                else:
                    casted.append(BEHAVIORS_BY_MODE["StLy"]["Standing"])
            # if behavior is lying then the corresponding subactions (LHU or LHD) get incorporated into behhavior sequence
            elif stly_seq[i] == BEHAVIORS_BY_MODE["StLy"]["Lying"]:
                if len(lhulhd_seq) != 0:
                    # if we have Lying (because of filtering) but no LHULHD information
                    if np.isnan(lhulhd_seq[i]):
                        # if subaction before and after are identical, use this subaction
                        if i > 0 and i < len(lhulhd_seq) - 1 and lhulhd_seq[i - 1] == lhulhd_seq[i + 1]:
                            casted.append(lhulhd_seq[i - 1])
                        # if subaction before and after are not identical, use LHU
                        else:
                            casted.append(BEHAVIORS_BY_MODE["LHULHD"]["LHU"])
                    # if we have LHULHD information
                    else:
                        casted.append(lhulhd_seq[i])
                # if we have no LHULHD  information
                else:
                    casted.append(BEHAVIORS_BY_MODE["StLy"]["Lying"])
            # if behavior is Out then Out is appended into casted behavior sequence
            elif stly_seq[i] == BEHAVIORS_BY_MODE_2["StLy"]["Out of View"]:
                casted.append(BEHAVIORS_BY_MODE_2["StLy"]["Out of View"])
        return casted




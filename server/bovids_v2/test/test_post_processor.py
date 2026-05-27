__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

import unittest
import os, sys

import post_processor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from lib.post_processor import PostProcessor, PostProcessorSubactions

# , convert_and_align_subactions
from typing import List
import pandas as pd


class testPostProcessor(unittest.TestCase):

    test_ruleset: pd.DataFrame = pd.DataFrame(  # TODO: zeiten in sekunden?
        {
            "previous": [1, 1, 1, 2, 2, 2],
            "current": [2, 3, 3, 3, 3, 1],
            "next": [1, 1, 2, 1, 2, 2],
            "current_corrected": [1, 1, 1, 1, 2, 2],
            "rule": [10, 10, 10, 10, 10, 10],
        }
    )

    empty_test_ruleset: pd.DataFrame = pd.DataFrame(
        {"previous": [], "current": [], "next": [], "current_corrected": [], "rule": []}
    )

    test_subaction_ruleset: pd.DataFrame = pd.DataFrame(
        {
            "previous": [10, 11, 20, 21],
            "current": [11, 10, 21, 20],
            "next": [10, 11, 20, 21],
            "current_corrected": [10, 11, 20, 21],
            "rule": [10, 10, 10, 10],
        }
    )

    def main(self):
        pass
        # call all test functions

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_cast_interval_sequence_to_time_behavior_sequence_empty(self) -> None:
        interval_sequence: List[int] = []
        seconds_per_interval: int = 5

        post_processor = PostProcessor(
            self.test_ruleset, "rule", seconds_per_interval, interval_sequence
        )
        ret_behavior, ret_times = (
            post_processor.cast_interval_sequence_to_time_behavior_sequence(
                interval_sequence, seconds_per_interval
            )
        )
        self.assertEqual(ret_behavior, [])
        self.assertEqual(ret_times, [])

    def test_cast_interval_sequence_to_time_behavior_sequence_short(self) -> None:

        interval_sequence: List[int] = [1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2]
        seconds_per_interval: int = 5

        post_processor = PostProcessor(
            self.test_ruleset, "rule", seconds_per_interval, interval_sequence
        )
        ret_behavior, ret_times = (
            post_processor.cast_interval_sequence_to_time_behavior_sequence(
                interval_sequence, seconds_per_interval
            )
        )
        self.assertEqual(ret_behavior, [1, 2, 1, 2, 1, 2])
        self.assertEqual(ret_times, [5, 10, 15, 20, 10, 5])

    def test_cast_interval_sequence_to_time_behavior_sequence_long(self) -> None:
        interval_sequence: List[int] = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            3,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
        ]
        seconds_per_interval: int = 5

        post_processor = PostProcessor(
            self.test_ruleset, "rule", seconds_per_interval, interval_sequence
        )
        ret_behavior, ret_times = (
            post_processor.cast_interval_sequence_to_time_behavior_sequence(
                interval_sequence, seconds_per_interval
            )
        )

        self.assertEqual(ret_behavior, [1, 2, 1, 3, 1, 2, 1, 2])
        self.assertEqual(ret_times, [55, 40, 5, 5, 5, 20, 30, 5])

    def test_cast_interval_sequence_to_time_behavior_sequence_subactions(self) -> None:
        interval_sequence: List[int] = [10, 10, 21, 20, 20, 10, 21, 21]
        seconds_per_interval: int = 5

        post_processor = PostProcessor(
            self.test_subaction_ruleset, "rule", seconds_per_interval, interval_sequence
        )
        ret_behavior, ret_times = (
            post_processor.cast_interval_sequence_to_time_behavior_sequence(
                interval_sequence, seconds_per_interval
            )
        )

        self.assertEqual(ret_behavior, [10, 21, 20, 10, 21])
        self.assertEqual(ret_times, [10, 5, 10, 5, 10])

    def test_set_ruleset(self) -> None:
        interval_sequence: List[int] = [1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2]
        interval: int = 5

        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_rule, ret_behavior = post_processor.set_ruleset("rule", self.test_ruleset)

        self.assertEqual(
            ret_rule,
            {
                (1, 2, 1): 10,
                (1, 3, 1): 10,
                (1, 3, 2): 10,
                (2, 3, 1): 10,
                (2, 3, 2): 10,
                (2, 1, 2): 10,
            },
        )
        self.assertEqual(
            ret_behavior,
            {
                (1, 2, 1): 1,
                (1, 3, 1): 1,
                (1, 3, 2): 1,
                (2, 3, 1): 1,
                (2, 3, 2): 2,
                (2, 1, 2): 2,
            },
        )

    def test_set_ruleset_empty(self) -> None:
        interval_sequence: List[int] = [1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2]
        interval: int = 5

        post_processor = PostProcessor(
            self.empty_test_ruleset, "rule", interval, interval_sequence
        )
        ret_rule, ret_behavior = post_processor.set_ruleset(
            "rule", self.empty_test_ruleset
        )

        self.assertEqual(ret_rule, {})
        self.assertEqual(ret_behavior, {})

    def test_set_ruleset_subactions(self) -> None:
        interval_sequence: List[int] = [10, 10, 21, 20, 20, 10, 21, 21]
        interval: int = 5

        post_processor = PostProcessor(
            self.test_subaction_ruleset, "rule", interval, interval_sequence
        )
        ret_rule, ret_behavior = post_processor.set_ruleset(
            "rule", self.test_subaction_ruleset
        )

        self.assertEqual(
            ret_rule,
            {(10, 11, 10): 10, (11, 10, 11): 10, (20, 21, 20): 10, (21, 20, 21): 10},
        )
        self.assertEqual(
            ret_behavior,
            {(10, 11, 10): 10, (11, 10, 11): 11, (20, 21, 20): 20, (21, 20, 21): 21},
        )

    def test_filter_short_phases_combine_to_one_phase(self) -> None:
        interval_sequence: List[int] = [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1]
        interval: int = 5

        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [1])
        self.assertEqual(ret_rule, [60])

    def test_filter_short_phases_combine_to_two_phases(self) -> None:
        interval_sequence: List[int] = [1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2]
        interval: int = 5
        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [1, 2])
        self.assertEqual(ret_rule, [35, 25])

    def test_filter_short_phases_two_short_phases(self) -> None:
        interval_sequence: List[int] = [1, 1, 1, 1, 2, 1, 2, 2, 2, 2]
        interval: int = 5
        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [1, 2])
        self.assertEqual(ret_rule, [30, 20])

    def test_filter_short_phases_three_short_phases(self) -> None:
        interval_sequence: List[int] = [1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1]
        interval: int = 5
        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [1])
        self.assertEqual(ret_rule, [55])

    def test_filter_short_phases_no_short_phases(self) -> None:
        interval_sequence: List[int] = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]
        interval: int = 5
        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [1, 2, 1])
        self.assertEqual(ret_rule, [20, 20, 20])

    def test_filter_short_phases_two_long_phases(self) -> None:
        interval_sequence: List[int] = [
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
        ]
        interval: int = 5
        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [1, 2])
        self.assertEqual(ret_rule, [45, 45])

    def test_filter_short_phases_missing_rule(self) -> None:
        interval_sequence: List[int] = [3, 3, 3, 1, 3, 3, 3]
        interval: int = 5

        post_processor = PostProcessor(
            self.test_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [3, 1, 3])
        self.assertEqual(ret_rule, [15, 5, 15])

    def test_filter_short_phases_subactions(self) -> None:
        interval_sequence: List[int] = [21, 21, 20, 21, 21, 10, 10, 11, 10, 10]
        interval: int = 5

        post_processor = PostProcessor(
            self.test_subaction_ruleset, "rule", interval, interval_sequence
        )
        ret_behavior, ret_rule = post_processor.filter_short_phases(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
            post_processor.rule_set_time,
            post_processor.rule_set_behavior,
        )
        self.assertEqual(ret_behavior, [21, 10])
        self.assertEqual(ret_rule, [25, 25])

    def test_save_post_processed_sequence(self) -> None:
        interval_sequence: List[int] = [21, 21, 20, 21, 21, 10, 10, 11, 10, 10]
        interval: int = 5

        post_processor = PostProcessor(
            self.test_subaction_ruleset, "rule", interval, interval_sequence
        )
        pp_seq_df = post_processor.save_post_processed_sequence(
            post_processor.original_behavior_sequence,
            post_processor.original_time_sequence,
        )

        print(pp_seq_df)


class testPostProcessorSubactions(unittest.TestCase):
    def test_cast_post_processed_sequence(self) -> None:
        stly_behavior: List[int] = [1, 2, 1, 2]
        stly_time: List[int] = [10, 15, 5, 10]

        interval: int = 5

        behavior_time = [stly_behavior, stly_time]
        post_processor_subaction = PostProcessorSubactions(
            stly_behavior, stly_time, interval
        )
        stly_behavior, stly_time = (
            post_processor_subaction.cast_post_processed_sequence_into_interval_sequence(
                behavior_time, interval
            )
        )

        self.assertEqual(stly_behavior, [1, 1, 2, 2, 2, 1, 2, 2])
        self.assertEqual(stly_time, [5, 5, 5, 5, 5, 5, 5, 5])

    def test_incorporate_subactions_sequence(self) -> None:
        stly_behavior: List[int] = [1, 2, 1, 2]
        stly_time: List[int] = [10, 15, 5, 10]
        stfo_behavior: List[int] = [10, 11, None, None, None, 10, None, None]
        # stfo_time: List[int] = [5, 5, 15, 5, 10]
        lhulhd_behavior: List[int] = [None, None, 21, 21, 21, 21, None, 21, 21]
        # lhulhd_time: List[int] = [10, 15, 5, 10]

        interval: int = 5

        post_processor_subaction = PostProcessorSubactions(
            stly_behavior, stly_time, interval
        )
        casted = post_processor_subaction.incorporate_subactions_sequence(
            post_processor_subaction.stly_behavior, stfo_behavior, lhulhd_behavior
        )

        self.assertEqual(casted, [10, 11, 21, 21, 21, 10, 21, 21])

    def test_incorporate_subactions_sequence_missing_subaction_unequal(self) -> None:
        stly_behavior: List[int] = [1, 2, 1, 2]
        stly_time: List[int] = [10, 15, 5, 10]
        stfo_behavior: List[int] = [10, 10, None, None, None, 10, None, None]
        # stfo_time: List[int] = [10, 15, 5, 10]
        lhulhd_behavior: List[int] = [None, None, 21, None, 20, None, 21, 21]
        # lhulhd_time: List[int] = [10, 5, 5, 5, 5, 10]

        interval: int = 5

        # subactions_seqs = [stly_behavior, stfo_behavior,  lhulhd_behavior]
        post_processor_subaction = PostProcessorSubactions(
            stly_behavior, stly_time, interval
        )
        casted = post_processor_subaction.incorporate_subactions_sequence(
            post_processor_subaction.stly_behavior, stfo_behavior, lhulhd_behavior
        )

        self.assertEqual(casted, [10, 10, 21, 20, 20, 10, 21, 21])

    def test_incorporate_subactions_sequence_missing_subaction_equal(self) -> None:
        stly_behavior: List[int] = [1, 2, 1, 2]
        stly_time: List[int] = [10, 15, 5, 10]
        stfo_behavior: List[int] = [10, 10, None, None, None, 10, None, None]
        # stfo_time: List[int] = [10, 15, 5, 10]
        lhulhd_behavior: List[int] = [None, None, 21, None, 21, None, 21, 21]
        # lhulhd_time: List[int] = [10, 5, 5, 5, 5, 10]

        interval: int = 5

        post_processor_subaction = PostProcessorSubactions(
            stly_behavior, stly_time, interval
        )
        casted = post_processor_subaction.incorporate_subactions_sequence(
            post_processor_subaction.stly_behavior, stfo_behavior, lhulhd_behavior
        )

        self.assertEqual(casted, [10, 10, 21, 21, 21, 10, 21, 21])

    def test_incorporate_subactions_sequence_empty(self) -> None:
        stly_behavior: List[int] = [1, 2, 1, 2]
        stly_time: List[int] = [10, 15, 5, 10]
        stfo_behavior: List[int] = [10, 10, None, None, None, 10, None, None]
        # stfo_time: List[int] = [10, 15, 5, 10]
        lhulhd_behavior: List[int] = []
        # lhulhd_time: List[int] = []

        interval: int = 5

        post_processor_subaction = PostProcessorSubactions(
            stly_behavior, stly_time, interval
        )
        casted = post_processor_subaction.incorporate_subactions_sequence(
            post_processor_subaction.stly_behavior, stfo_behavior, lhulhd_behavior
        )

        self.assertEqual(casted, [10, 10, 2, 2, 2, 10, 2, 2])


"""
class testConversionActions(unittest.TestCase):
    def test_convert_and_align_subactions(self) -> None:
        test_stly_df :pd.DataFrame = pd.DataFrame({
            'img_name': ['img_1', 'img_2', 'img_3', 'img_4', 'img_5', 'img_6', 'img_7', 'img_8', 'img_9', 'img_10'],
            'Standing': [0.4, 0.2, 0.8, 0.9, 0.3, 0.3, 0.1, 0.9, 0.8, 0.7],
            'Lying': [0.6, 0.8, 0.2, 0.1, 0.7, 0.7, 0.9, 0.1, 0.2, 0.3]
        })
        test_stfo_df: pd.DataFrame = pd.DataFrame({
            'img_name': ['img_3', 'img_4', 'img_8', 'img_9', 'img_10'],
            'Standing_no_food': [0.4, 0.2, 0.8, 0.9, 0.3],
            'Food': [0.6, 0.8, 0.2, 0.1, 0.7]
        })
        test_lhulhd_df: pd.DataFrame = pd.DataFrame({
            'img_name': ['img_1', 'img_2',  'img_5', 'img_6', 'img_7'],
            'LHU': [0.4, 0.2, 0.8, 0.9, 0.3],
            'LHD': [0.6, 0.8, 0.2, 0.1, 0.7]
        })
        stly_seq, stfo_seq, lhulhd_seq = convert_and_align_subactions(test_stly_df, test_stfo_df, test_lhulhd_df)

        self.assertEqual(stly_seq, [2, 2, 1, 1, 2, 2, 2, 1, 1, 1])
        self.assertEqual(stfo_seq, [None, None, 11, 11, None, None, None, 10, 10, 11])
        self.assertEqual(lhulhd_seq, [21, 21, None, None, 20, 20, 21, None, None, None])

    def test_convert_and_align_subactions_empty_df(self) -> None:
        test_stly_df: pd.DataFrame = pd.DataFrame({
            'img_name': ['img_1', 'img_2', 'img_3', 'img_4', 'img_5', 'img_6', 'img_7', 'img_8', 'img_9', 'img_10'],
            'Standing': [0.4, 0.2, 0.8, 0.9, 0.3, 0.3, 0.1, 0.9, 0.8, 0.7],
            'Lying': [0.6, 0.8, 0.2, 0.1, 0.7, 0.7, 0.9, 0.1, 0.2, 0.3]
        })
        test_stfo_df: pd.DataFrame = pd.DataFrame({
            'img_name': ['img_3', 'img_4', 'img_8', 'img_9', 'img_10'],
            'Standing_no_food': [0.4, 0.2, 0.8, 0.9, 0.3],
            'Food': [0.6, 0.8, 0.2, 0.1, 0.7]
        })
        test_lhulhd_df: pd.DataFrame = pd.DataFrame({
            'img_name': [],
            'LHU': [],
            'LHD': []
        })
        stly_seq, stfo_seq, lhulhd_seq = convert_and_align_subactions(test_stly_df, test_stfo_df, test_lhulhd_df)


        self.assertEqual(stly_seq, [2, 2, 1, 1, 2, 2, 2, 1, 1, 1])
        self.assertEqual(stfo_seq, [None, None, 11, 11, None, None, None, 10, 10, 11])
        self.assertEqual(lhulhd_seq, [None, None, None, None, None, None, None, None, None, None])
"""
if __name__ == "__main__":
    # testPostProcessor.main()
    # testPostProcessor.test_set_ruleset()
    testPostProcessor.test_filter_short_phases()

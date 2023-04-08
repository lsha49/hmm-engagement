import numpy.distutils.system_info as sys_info
from datetime import datetime, timedelta
from bisect import bisect_left
import gzip
import logging
from json import loads
from math import ceil
from os import path, listdir
import pandas as pd
import numpy as np
import math
import re
import sys
from collections import Counter

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
pd.options.mode.chained_assignment = None
sys_info.get_info("atlas")


def fetch_start_end_date(course, session_dir, date_csv="coursera_course_dates.csv"):
    """
    Fetch course start end end date (so user does not have to specify them directly).
    :param course: course name.
    :param session_dir: input directory.
    :param date_csv: Path to csv of course start/end dates.
    :return: tuple of datetime objects (course_start, course_end)
    """
    date_df = pd.read_csv(path.join(session_dir, date_csv)).set_index("course")
    course_start = datetime.strptime(date_df.loc[course].start_date, "%m/%d/%y")
    course_end = datetime.strptime(date_df.loc[course].end_date, "%m/%d/%y")
    return course_start, course_end


def has_child(id, df):
    """
    Given an identifier and a DataFrame with the column "parent_id", returns whether or not the DataFrame contains the identifier.
    :param id: string, course resource identifier.
    :param df: DataFrame, must contains "parent_id" column.
    :return:
    """
    return int(id) in df.parent_id


def parent_week(df):
    """
    Given a DataFrame, returns a dictionary that maps DataFrame section_titles to numbers representing weeks. Manually,
    the "Demonstration Section" section_title (0) is taken to be the same as the first week (1)
    :param df: DataFrame, contains "section_title" column, which are assumed to distinguish weeks.
    :return:
    """
    parents = {
        title: week for title, week in zip(
            df.section_title.unique(),
            [week for week in range(0, len(df.section_title.unique()))]
        )
    }

    if "Demonstration Section" in parents.keys():
        parents["Demonstration Section"] += 1

    return parents


def action_classify(action_json):
    """
    Classify the action into predefined action types
    :param action_json: the json format data for this action
    :return: action_type, action_id
    """
    if action_json["key"] == "pageview":
        url = action_json["page_url"]
        # This is removed because it essentially duplicates "video"
        # m1 = re.search(r"/lecture/(.*?(\d+))", url)
        # if m1:
        #     return "lecture", m1.groups()[-1]
        m2 = re.search(r"/forum/(.*?thread_id=(\d+))", url)
        if m2:
            return "forum", m2.groups()[-1]
        m2 = re.search(r"/forum/(.*?forum_id=(\d+))", url)
        if m2:
            return "forum", m2.groups()[-1]
        m2 = re.search(r"/forum/(.*?page=(\d+))", url)
        if m2:
            return "forum", m2.groups()[-1]
        m3 = re.search(r"/wiki/", url)
        if m3:
            return "wiki", "na"
        m4 = re.search(r"/human_grading/(.*?/assessments/(\d+)|index)", url)
        if m4:
            return "hg", "na"
        m5 = re.search(r"/quiz/(.*?quiz_id=(\d+))", url)
        if m5:
            # if line % 1000 == 0:
            #     print("------ quiz -------")
            #     print(action_json)
            return "quiz", m5.groups()[-1]
        return "pageview_other", "na"

    elif action_json["key"] == "user.video.lecture.action":
        url = action_json["page_url"]
        m6 = re.search(r"/lecture/(.*?(\d+))", url)
        if m6:
            return "lecture", m6.groups()[-1]
        return "video_other", "na"

    else:
        logging.info("Not found matching key: {}".format(action_json['key']))
        return "Other", "na"


def scale_timestamp(time, oom=9):
    """
    Alter the timestamp to the correct order of magnitude.
    :param time: timestamp to be altered.
    :param oom: order of magnitude.
    :return:
    """
    return datetime.fromtimestamp(time / math.pow(10, math.floor(math.log(time, 10)) - oom)).timestamp()


def between_timestamp(x, start, end, week):
    """
    Performs a boolean index to find the segment of a course within which a given timestamp occurs.
    :param x: float representing a given timestamp from the clickstream data.
    :param start: numpy array of segment start times.
    :param end: numpy array of segment week times.
    :param week: numpy array of segment names.
    :return:
    """
    return week[(x >= start) & (x <= end)][0]


def add_quiz_section(design):
    """
    Function takes the Code Yourself course design and distributes the weekly quiz assignments to their respective
    sections within the design. It also removes extraneous lecture and video content.
    :param design:
    :return:
    """
    quiz_section_map = {
        "Unit 1 Quiz": 13,
        "Unit 2 Quiz": 16,
        "Unit 3 Quiz": 19,
        "Unit 4 Quiz": 12,
        "Unit 5 Quiz": 22
    }
    relevant_sections = [12, 13, 16, 19, 22]

    for key in quiz_section_map.keys():
        design.loc[design.title == key, "section_id"] = quiz_section_map[key]
        max_order = design[design.section_id == quiz_section_map[key]].order.max()
        design.loc[design.title == key, "order"] = design[design.title == key].order.apply(lambda x: x + round(max_order - x + 1, 0))

    return design[design.section_id.isin(relevant_sections)]


class ExtractCourseDesign:

    def __init__(self, course):
        """
        To create course design:
            1. go through each row in item_sections
                1.1 if quiz: get title from quiz_metadata
                1.2 if lecture: get title from lecture_metadata
                    1.2.1 get quiz_id from lecture_metadata and add corresponding rows from quiz_metadata
        """
        self.course = course
        self.session_dir = path.join("./data/", self.course)
        self.design = pd.DataFrame({
            "item_type": [],
            "item_id": [],
            "section_id": [],
            "order": [],
            "title": []
        })

        self.sections = pd.read_csv(
            path.join(
                self.session_dir,
                "sections.csv"
            ),
            error_bad_lines=False
        )

        self.item_sections = pd.read_csv(
            path.join(
                self.session_dir,
                "item_sections.csv"
            ),
            error_bad_lines=False
        ).sort_values(["section_id", "order"]).reset_index(drop=True)

        self.quiz_metadata = pd.read_csv(
            path.join(
                self.session_dir,
                "quiz_metadata.csv"
            ),
            error_bad_lines=False
        )

        self.quiz_subs = pd.read_csv(
            path.join(
                self.session_dir,
                "quiz_submissions.csv"
            ),
            error_bad_lines=False
        )

        self.lecture_metadata = pd.read_csv(
            path.join(
                self.session_dir,
                "lecture_metadata.csv"
            ),
            error_bad_lines=False
        )

        self.lecture_subs = pd.read_csv(
            path.join(
                self.session_dir,
                "lecture_submissions.csv"
            ),
            error_bad_lines=False
        )

        self.augment_sections()

        if self.course == "code-yourself":
            self.design = add_quiz_section(self.design)

        self.segment_sections()
        self.design.to_csv(path.join(self.session_dir,"course_design.csv"), index=False)

    def augment_sections(self):
        """
        Iterates over item_sections table and builds a course design DataFrame
        :return:
        """
        for idx, row in self.item_sections.iterrows():

            if row["item_type"] == "quiz":
                self.insert_quiz(row)

            elif row["item_type"] == "lecture":
                self.insert_lecture(row)

    def insert_quiz(self, row):
        """
        Given a row representing a quiz item, adds quiz and child quiz items to the course design.
        :param row: Series, contains a single course resource.
        :return:
        """
        quiz = self.quiz_metadata[self.quiz_metadata.item_type == "quiz"]

        logging.info("Quiz {} has {} rows in quiz_metadata, {} logs in quiz_submissions".format(
            row.item_id,
            len(quiz[quiz.item_id == int(row.item_id)]),
            len(self.quiz_subs[self.quiz_subs.type_id == "quiz_{}".format(row.item_id)])
        ))

        if len(quiz[quiz.item_id == int(row.item_id)]) == 1:
            quiz_item = quiz[quiz.item_id == int(row.item_id)]

            self.design = pd.concat([
                self.design,
                pd.DataFrame({
                    "item_type": [quiz_item.item_type.values[0]],
                    "item_id": [int(row.item_id)],
                    "section_id": [int(row.section_id)],
                    "order": [int(row.order)],
                    "title": [quiz_item.title.values[0]]
                })
            ]).reset_index(drop=True)

            if has_child(row.item_id, self.quiz_metadata):
                child_item = quiz[quiz.parent_id == int(row.item_id)]

                self.design = pd.concat([
                    self.design,
                    pd.DataFrame({
                        "item_type": [child_item.item_type.values[0]],
                        "item_id": [int(child_item.item_id.values[0])],
                        "section_id": [int(row.section_id)],
                        "order": [int(row.order) + 0.1],
                        "title": [child_item.title.values[0]]
                    })
                ])

    def insert_lecture(self, row):
        """
        Given a lecture item, adds the lecture, any associated video item, any associated child lecture, and any
        associated child video item to the course design.
        :param row: Series, contains a single course resource.
        :return:
        """
        lecture = self.lecture_metadata[self.lecture_metadata.id == int(row.item_id)]

        logging.info("Lecture {} has {} rows in lecture_metadata, {} logs in lecture_submissions".format(
            row.item_id,
            len(lecture[lecture.id == int(row.item_id)]),
            len(self.lecture_subs[self.lecture_subs.type_id == "lecture_{}".format(row.item_id)])
        ))

        if len(lecture[lecture.id == int(row.item_id)]) == 1:
            lecture_item = lecture[lecture.id == int(row.item_id)]
            lecture_quiz_item = self.quiz_metadata[self.quiz_metadata.item_id == lecture_item.quiz_id.values[0]]

            self.design = pd.concat([
                self.design,
                pd.DataFrame({
                    "item_type": ["lecture"],
                    "item_id": [int(row.item_id)],
                    "section_id": [int(row.section_id)],
                    "order": [int(row.order)],
                    "title": [lecture_item.title.values[0]]
                })
            ]).reset_index(drop=True)

            if len(lecture_quiz_item) == 1:
                self.design = pd.concat([
                    self.design,
                    pd.DataFrame({
                        "item_type": ["video"],
                        "item_id": [int(lecture_quiz_item.item_id.values[0])],
                        "section_id": [int(row.section_id)],
                        "order": [int(row.order) + 0.1],
                        "title": [lecture_quiz_item.title.values[0]]
                    })
                ]).reset_index(drop=True)

            if has_child(lecture_item.id, self.lecture_metadata):
                child_item = self.lecture_metadata[self.lecture_metadata.parent_id == int(row.item_id)]
                child_quiz_item = self.quiz_metadata[self.quiz_metadata.item_id == child_item.quiz_id.values[0]]

                self.design = pd.concat([
                    self.design,
                    pd.DataFrame({
                        "item_type": ["lecture"],
                        "item_id": [int(child_item.id.values[0])],
                        "section_id": [int(row.section_id)],
                        "order": [int(row.order) + 0.2],
                        "title": [child_item.title.values[0]]
                    })
                ])

                if len(child_quiz_item) == 1:
                    self.design = pd.concat([
                        self.design,
                        pd.DataFrame({
                            "item_type": ["video"],
                            "item_id": [int(child_quiz_item.item_id.values[0])],
                            "section_id": [int(row.section_id)],
                            "order": [int(row.order) + 0.3],
                            "title": [child_quiz_item.title.values[0]]
                        })
                    ]).reset_index(drop=True)

    def segment_sections(self):
        """
        Given a complete course design, segments the design into weeks (assumes a 1:1 mapping from the sections table),
        and calculates week start and end timestamps from working backwards from the course end date.
        :return:
        """
        sections = pd.merge(
            self.design,
            self.sections,
            on="section_id"
        )[["item_type", "item_id", "section_id", "order", "title_x", "title_y"]]

        sections.columns = ["item_type", "item_id", "section_id", "order", "title", "section_title"]
        sections["parent_week"] = sections.section_title.apply(lambda x: parent_week(sections)[x])

        start, end = fetch_start_end_date(self.course, self.session_dir)

        intervals = divmod((end - start).days, 7)[0]
        timestamps = [ts.timestamp() for ts in [start + (i * timedelta(days=7)) for i in range(intervals + 1)]]

        segment_ends = {
            section: time for section, time in zip(
                sections.parent_week.unique(),
                timestamps[-len(sections.parent_week.unique()):]
            )
        }

        timestamps = [start + (i * timedelta(days=7)) for i in range(intervals + 1)]
        timestamps = [ts + timedelta(seconds=0.001) for ts in timestamps[-len(sections.parent_week.unique()):-1]]
        timestamps.insert(0, start)

        segment_starts = {
            section: time for section, time in zip(
                sections.parent_week.unique(),
                [ts.timestamp() for ts in timestamps]
            )
        }

        sections["segment_start"] = sections.parent_week.apply(lambda x: segment_starts[x])
        sections["segment_end"] = sections.parent_week.apply(lambda x: segment_ends[x])

        self.design = sections


class ExtractLogs(ExtractCourseDesign):

    def __init__(self, course):
        super().__init__(course)

        self.clickstream = self.extract_clickstream()
        self.aggregate_logs()

    def extract_clickstream(self):
        """
        Iterates over rows in the clickstream logs. Afterwards, lecture view logs are compared to the lecture items in
        the course design. Any logged lecture items missing from the course design are flagged. Then compares quiz logs
        to the quiz items in the course design. Any quiz items missing from the course design are flagged.
        :return:
        """
        cs = [x for x in listdir(self.session_dir) if x.endswith("clickstream_export.gz")][0]
        cs_file = path.join(self.session_dir, cs)

        line_count = 0
        cs_data = []

        with gzip.open(cs_file, "r") as f:
            for line in f:
                try:
                    log_entry = loads(line.decode("utf-8"))
                    type_id = action_classify(log_entry)
                    log_dict = {
                        "session_user_id": log_entry.get("username"),
                        "timestamp": log_entry.get("timestamp"),
                        "item_type": type_id[0],
                        "item_id": type_id[1]
                    }
                    cs_data.append(log_dict)
                except ValueError as e1:
                    logging.warning("Invalid log line {0}: {1}".format(line_count, e1))
                except Exception as e:
                    logging.error("Invalid log line {0}: {1}\n{2}".format(line_count, e, line))
                line_count += 1

        cs = pd.DataFrame(cs_data)

        logging.info("Click-stream has {} lecture items, {} lecture items in course design".format(
            len(set(cs[cs.item_type == "lecture"].item_id)),
            len(set(self.design[self.design.item_type == "lecture"].item_id)),
        ))

        design_lecture_missing = set(cs[cs.item_type == "lecture"].item_id.astype(int)).difference(
            set(self.design[self.design.item_type == "lecture"].item_id.astype(int)))

        for item in design_lecture_missing:
            logging.info("Lecture {} is in click-stream, but not course design. {} logs lost.".format(
                item,
                len(cs[((cs.item_type == "lecture") & (cs.item_id == str(item)))]),
            ))

        logging.info("Click-stream has {} quiz items, {} quiz items in course design".format(
            len(set(cs[cs.item_type == "quiz"].item_id)),
            len(set(self.design[self.design.item_type == "quiz"].item_id)),
        ))

        design_quiz_missing = set(cs[cs.item_type == "quiz"].item_id.astype(int)).difference(
            set(self.design[self.design.item_type == "quiz"].item_id.astype(int)))

        for item in design_quiz_missing:
            logging.info("Quiz {} is in click-stream, but not course design. {} logs lost.".format(
                item,
                len(cs[((cs.item_type == "quiz") & (cs.item_id == str(item)))]),
            ))

        return cs[cs.item_type.isin(["lecture", "quiz"])]

    def aggregate_logs(self):
        """
        Aggregate clickstream and SQL submission logs. Survey items are removed. Course resource access counts are
        calculated, and untouched resources are removed. Parent weeks are calculated and, after timestamps are scaled,
        action weeks are also calculated.
        :return:
        """
        all_logs = pd.concat([
            self.aggregate_quiz(),
            self.aggregate_lecture()
        ])

        all_logs = all_logs[~all_logs.item_type.isin(["survey"])]
        all_logs["type_id"] = all_logs.item_type.map(str) + "_" + all_logs.item_id.map(str)

        self.design["type_id"] = self.design.item_type.map(str) + "_" + self.design.item_id.astype(int).map(str)

        for item in self.design.type_id:
            logging.info("Course resource {} is accessed {} times.".format(
                    item,
                    len(all_logs[all_logs.type_id == item]),
                ))

        self.design[self.design.type_id.isin(all_logs.type_id)].to_csv(
            path.join(self.session_dir, "course_design_accessed.csv"),
            index=False
        )

        type_id_to_parent_week = dict(zip(self.design.type_id, self.design.parent_week))

        all_logs = all_logs[all_logs.type_id.isin(self.design.type_id)]
        all_logs["parent_week"] = all_logs.type_id.apply(lambda x: type_id_to_parent_week[x])

        start = self.design.segment_start.values
        end = self.design.segment_end.values
        week = self.design.parent_week.values

        all_logs["timestamp"] = all_logs.timestamp.apply(lambda x: scale_timestamp(x, 9))
        all_logs = all_logs[((all_logs.timestamp >= start.min()) & (all_logs.timestamp <= end.max()))]

        all_logs["action_week"] = all_logs.timestamp.apply(between_timestamp, args=[start, end, week])

        all_logs.to_csv(
            path.join(self.session_dir, "events.csv"),
            index=False
        )

    def aggregate_quiz(self):
        """
        Quiz items in the clickstream and SQL submissions are aggregated. Items in both are mapped to their respective
        item_id in the quiz_metadata table, from which item_type is known.
        :return:
        """
        id_type = dict(zip(self.quiz_metadata.item_id, self.quiz_metadata.item_type))

        self.quiz_subs["item_id"] = self.quiz_subs.type_id.apply(lambda x: int(x.split("_")[1]))

        quiz_missing = set(self.quiz_subs.item_id).difference(set(self.quiz_metadata.item_id))
        for item in quiz_missing:
            logging.info("Quiz/video quiz {} is in quiz submissions, but not course design. {} logs lost.".format(
                    item,
                    len(self.quiz_subs[self.quiz_subs.item_id == item]),
                ))

        self.quiz_subs = self.quiz_subs[~self.quiz_subs.item_id.isin(quiz_missing)]
        self.quiz_subs["item_type"] = self.quiz_subs.item_id.apply(lambda x: id_type[x])

        quiz_clicks = self.clickstream[self.clickstream.item_type == "quiz"]
        quiz_clicks["item_id"] = quiz_clicks.item_id.apply(lambda x: int(x))

        quiz_clicks_missing = set(quiz_clicks.item_id).difference(set(self.quiz_metadata.item_id))
        for item in quiz_clicks_missing:
            logging.info("Quiz/video quiz {} is in quiz clickstream, but not course design. {} logs lost.".format(
                    item,
                    len(quiz_clicks[quiz_clicks.item_id == item]),
                ))

        quiz_clicks = quiz_clicks[~quiz_clicks.item_id.isin(quiz_missing)]
        quiz_clicks["item_type"] = quiz_clicks.item_id.apply(lambda x: id_type[x])

        all_quiz = pd.concat([
            self.quiz_subs[["item_id", "item_type", "session_user_id", "timestamp"]],
            quiz_clicks[["item_id", "item_type", "session_user_id", "timestamp"]],
        ])

        return all_quiz

    def aggregate_lecture(self):
        """
        Lecture items in the clickstream and SQL submissions are aggregated. Items are mapped to their respective ids
        from lecture_metadata and missing items are flagged.
        :return:
        """
        self.lecture_subs["item_id"] = self.lecture_subs.type_id.apply(lambda x: int(x.split("_")[1]))
        self.lecture_subs["item_type"] = "lecture"

        lecture_missing = set(self.lecture_subs.item_id).difference(set(self.lecture_metadata.id))
        for item in lecture_missing:
            logging.info("Lecture {} is in lecture submissions, but not course design. {} logs lost.".format(
                    item,
                    len(self.lecture_subs[self.lecture_subs.item_id == item]),
                ))

        lecture_clicks = self.clickstream[self.clickstream.item_type == "lecture"]
        lecture_clicks["item_id"] = lecture_clicks.item_id.apply(lambda x: int(x))

        all_lecture = pd.concat([
            self.lecture_subs[["item_id", "item_type", "session_user_id", "timestamp"]],
            lecture_clicks[["item_id", "item_type", "session_user_id", "timestamp"]],
        ])

        return all_lecture

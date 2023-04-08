import argparse
from os import path

from feature_extraction.feature_extractor import ExtractLogs
from feature_extraction.sql_utils import extract_coursera_sql_data
from models.topic_modelling import evaluate_topic_models, predict_certification_topic_models, predict_last_action_topic_models, plot_log_perplexity, plot_topic_distribution
from models.hmm import evaluate_hmm, predict_certification_hmm, plot_transition_matrix, plot_action_probabilities, predict_last_action_hmm
from models.baseline import evaluate_baseline_models, predict_last_action_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute feature extraction or model training."
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        help="Mode to run script in (extract, replicate, extend)"
    )
    args = parser.parse_args()

    courses = ["code-yourself", "big-data-edu"]

    if args.mode == "extract":
        # Note that this first function call requires root permissions.
        for course in courses:
            extract_coursera_sql_data(course)
            ExtractLogs(course)

    elif args.mode == "replicate":
        for course in courses:
            course_path = path.join("./out/", course, "PUC")
            evaluate_topic_models(course, course_path)

            predict_certification_topic_models(course, course_path)
            predict_last_action_topic_models(course, course_path)

            plot_log_perplexity(start=0, end=4, output_dir=course_path)
            plot_topic_distribution(start=1, end=8, n_topics=3, course=course, output_dir=course_path)

    elif args.mode == "extend":
        # for idx, course in enumerate(courses):
        #     course_path = path.join("./out/", course, "HMM")
        #
        #     evaluate_hmm(course, course_path)
        #     predict_last_action_hmm(course_path, course_path)
        #
        #     plot_transition_matrix(output_dir=course_path)
        #     plot_action_probabilities(output_dir=course_path)

        for idx, course in enumerate(courses):
            course_path = path.join("./out/", course, "HMM")
            alt_course = [c for c in courses if c != course][0]
            predict_last_action_hmm(path.join("./out/", alt_course, "HMM"), course_path)
            assert 1 == 0

    elif args.mode == "baseline":
        for course in courses:
            course_path = path.join("./out/", course, "baseline")
        #     evaluate_baseline_models(course, course_path)
            predict_last_action_baseline(course, course)

        # for idx, course in enumerate(courses):
        #     alt_course = [c for c in courses if c != course][0]
        #     predict_last_action_baseline(alt_course, course)

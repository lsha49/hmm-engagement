import numpy.distutils.system_info as sys_info
from os import path, mkdir
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.WARNING)
sys_info.get_info("atlas")


def evaluate_baseline_models(course, output_dir):
    """
    Train baseline topic models for a given course and session, saving results to the passed output directory.
    :param course: input directory
    :param output_dir: output directory.
    :return:
    """
    session_dir = path.join("./data/", course)

    event_logs = pd.read_csv(
        path.join(
            session_dir,
            "events.csv"
        )
    )

    week_ids = event_logs.parent_week.unique()
    week_indices = zip([min(week_ids) for _ in range(len(week_ids))], week_ids)

    if not path.exists(path.join(output_dir, "models")):
        mkdir(path.join(output_dir, "models"))

    topic_dict = corpora.Dictionary([[
        "quiz_0", "quiz_1", "quiz_2", "quiz_3", "quiz_4", "quiz_5", "quiz_6", "quiz_7",
        "lecture_0", "lecture_1", "lecture_2", "lecture_3", "lecture_4", "lecture_5", "lecture_6", "lecture_7",
        "video_0", "video_1", "video_2", "video_3", "video_4", "video_5", "video_6", "video_7"
    ]])

    for start, end in week_indices:
        event_logs_slice = event_logs[((event_logs.action_week >= start) & (event_logs.action_week <= end))]

        mapping_dict = dict(zip(event_logs_slice.type_id.values, event_logs_slice.item_type.map(str) + "_" + event_logs_slice.parent_week.map(str)))

        event_logs_slice.type_id = event_logs_slice.type_id.apply(lambda x: mapping_dict[x])
        topics = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        run_k_fold_topic_modelling(
            data=event_logs_slice,
            idx2word=topic_dict,
            topics=topics,
            folds=5,
            start=start,
            end=end,
            output=path.join(output_dir, "models")
        )

    return


def run_k_fold_topic_modelling(data, idx2word, topics, folds, start, end, output):
    """
    Conduct k-fold Topic Modelling following the methodology outlined in Probabilistic Use Cases
    :param data: event logs for the period between start and end.
    :param idx2word: dictionary mapping indices to course design items.
    :param topics: number of topics.
    :param folds: number of CV folds.
    :param start: start date of data.
    :param end: end date of data.
    :param output: output directory.
    :return:
    """
    perplexity = {}

    corpus = extract_corpus(data, max_diff=1800)
    corpus_idx = np.array([idx2word.doc2bow(corpus[student]) for student in corpus.keys()])

    for topic in topics:
        perplexity[topic] = {
            "train": [],
            "test": []
        }
        best_fold = np.inf

        cv = KFold(n_splits=folds, random_state=42)
        for train_idx, test_idx in cv.split(corpus_idx):
            lda_model = LdaModel(
                corpus=corpus_idx[train_idx],
                id2word=idx2word,
                num_topics=topic,
                random_state=np.random.randint(1000),
                update_every=1,
                chunksize=100,
                passes=10,
                alpha="auto",
                per_word_topics=True
            )

            if np.abs(lda_model.log_perplexity(corpus_idx[test_idx])) < np.abs(best_fold):
                best_fold = lda_model.log_perplexity(corpus_idx[test_idx])
                lda_model.save(
                    path.join(
                        output,
                        "{}_{}_{}_topics".format(
                            start,
                            end,
                            topic
                        )
                    )
                )

            perplexity[topic]["train"].append(lda_model.log_perplexity(corpus_idx[train_idx]))
            perplexity[topic]["test"].append(lda_model.log_perplexity(corpus_idx[test_idx]))

        parameters = get_topic_model_parameters(
            LdaModel.load(
                path.join(
                    output,
                    "{}_{}_{}_topics".format(
                        start,
                        end,
                        topic
                    )
                )
            )
        )

        parameters.to_csv(
            path.join(
                output,
                "{}_{}_tm_params.csv".format(
                    start,
                    end
                )
            ),
            index=False,
            header=[True if topic == 3 else False][0],
            mode="a"
        )

        perplexity[topic]["train"] = np.mean(perplexity[topic]["train"])
        perplexity[topic]["test"] = np.mean(perplexity[topic]["test"])

    pd.DataFrame.from_dict(perplexity).to_csv(
        path.join(
            output,
            "{}_{}_tm_perplex.csv".format(
                start,
                end
            )
        ), index=True)

    return


def get_topic_model_parameters(model):
    """
    Extracts parameters from a topic model.
    :param model: gensim LdaModel object.
    :return:
    """
    data_frame = pd.DataFrame.from_dict({
        "n_topics": [],
        "topic": [],
        "word_idx": [],
        "weight": []
    })

    for topic in range(model.num_topics):
        new_frame = pd.DataFrame.from_dict({
            "n_topics": [model.num_topics for _ in model.get_topics()[topic]],
            "topic": [topic + 1 for _ in model.get_topics()[topic]],
            "word_idx": [idx for idx, val in enumerate(model.get_topics()[topic])],
            "weight": [val for idx, val in enumerate(model.get_topics()[topic])]
        })

        data_frame = pd.concat([data_frame, new_frame])

    return data_frame


def extract_corpus(data, max_diff):
    """
    Convert a subset of the event log data into a corpus with student IDs as keys and values being lists. Each list is
    the event log but the occurrence of each event log is multiplied by time stamp differences in seconds, maxed at
    max_diff. This is in keeping with the "Probabilistic Use Cases" paper.
    :param data: dataframe with student ID, type_id, and timestamp.
    :param max_diff: maximum number of seconds which can be spent on a given item.
    :return:
    """
    corpus = {}

    for student in data.session_user_id.unique():
        student_logs = data[data.session_user_id == student].sort_values(["timestamp"])
        student_logs["diff"] = (student_logs.timestamp.shift(-1).fillna(student_logs.timestamp.max()) -
                                student_logs.timestamp).apply(lambda x: round(min(x, max_diff) + 1))
        corpus[student] = [event[0] for event in student_logs[["type_id", "diff"]].values for _ in range(event[1])]

    return corpus


def predict_last_action_baseline(data_course, model_course):
    """
    For all actions up to a given week, predicts if a given week contains the last action for each student.
    :param data_course: data input course.
    :param model_course: model course.
    :return:
    """
    logging.info("Predicting when students take their last action.")

    session_dir = path.join("./data/", data_course)

    event_logs = pd.read_csv(
        path.join(
            session_dir,
            "events.csv"
        )
    )
    event_logs = event_logs[~event_logs.action_week.isna()]

    week_ids = event_logs.parent_week.unique()
    week_indices = zip([min(week_ids) for _ in range(len(week_ids))], week_ids)

    topic_dict = corpora.Dictionary([[
        "quiz_0", "quiz_1", "quiz_2", "quiz_3", "quiz_4", "quiz_5", "quiz_6", "quiz_7",
        "lecture_0", "lecture_1", "lecture_2", "lecture_3", "lecture_4", "lecture_5", "lecture_6", "lecture_7",
        "video_0", "video_1", "video_2", "video_3", "video_4", "video_5", "video_6", "video_7"
    ]])
    topics = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for start, end in week_indices:
        event_logs_slice = remove_already_inactive_users(event_logs, start, end)

        mapping_dict = dict(zip(event_logs_slice.type_id.values, event_logs_slice.item_type.map(str) + "_" + event_logs_slice.parent_week.map(str)))
        event_logs_slice.type_id = event_logs_slice.type_id.apply(lambda x: mapping_dict[x])

        last_action = pd.DataFrame.from_dict({
            "session_user_id": [student for student in event_logs_slice.session_user_id.unique()],
            "outcome": [0 for _ in event_logs_slice.session_user_id.unique()]
        })

        event_logs_slice, last_action = identify_inactive_users(event_logs_slice, last_action, start, end)

        logging.info("From week {} to {}, {} students took their last action.".format(
            start,
            end,
            int(sum(last_action.outcome)))
        )

        run_k_fold_svm(
            data=event_logs_slice,
            outcome=last_action,
            idx2word=topic_dict,
            topics=topics,
            folds=5,
            start=start,
            end=end,
            exp="_last_action_svm_perform_{}".format("SAME" if data_course == model_course else "DIFFERENT"),
            data_course=data_course,
            output=model_course
        )


def remove_already_inactive_users(logs, start, end):
    """
    For a given logs DataFrame, remove all users who had already taken their last recorded action in the previous week.
    :param logs: pandas DataFrame, contains all student actions.
    :param start: string, starting week identifier.
    :param end: string, final week identifier.
    :return:
    """
    if start == end:
        return logs

    active = logs.loc[logs.action_week > end - 1, "session_user_id"].values

    return logs[logs.session_user_id.isin(active)]


def identify_inactive_users(logs, outcome, start, end):
    """
    Returns the student logs from start to end, and the labels reflecting whether or not the next week contains a given
    student's last action in the course.
    :param logs: pandas DataFrame, contains all student actions.
    :param outcome: pandas DataFrame, contains student identifiers and whether or not the next week is their last.
    :param start: string, starting week identifier.
    :param end: string, final week identifier.
    :return:
    """
    active = logs.loc[logs.action_week > end, "session_user_id"].values
    outcome.loc[~outcome.session_user_id.isin(active), "outcome"] = 1

    logs = logs[((logs.action_week >= start) & (logs.action_week <= end))]
    outcome = outcome[outcome.session_user_id.isin(logs.session_user_id)]

    return logs, outcome


def run_k_fold_svm(data, outcome, idx2word, topics, folds, start, end, exp, data_course, output):
    """
    Conduct k-fold SVM certification classification following the methodology outlined in Probabilistic Use Cases
    :param data: event logs for the period between start and end.
    :param outcome: pandas DataFrame containing student identifier and student outcome variable for SVM.
    :param idx2word: dictionary mapping indices to course design items.
    :param topics: number of topics.
    :param folds: number of CV folds.
    :param start: start date of data.
    :param end: end date of data.
    :param exp: string representing the experiment name.
    :param data_course: data course name.
    :param output: model and output directory.
    :return:
    """
    corpus = extract_corpus(data[data.action_week.isin([start, end])], max_diff=1800)

    corpus_idx = np.array([idx2word.doc2bow(corpus[student]) for student in corpus.keys()])
    corpus_keys = [key for key in corpus.keys()]

    performance = {}

    if data_course == "code-yourself" and output == "big-data-edu":
        start += 1
        end += 1
        start_ = 1
        end_ = 8
    elif data_course == "big-data-edu" and output == "code-yourself":
        start += -1
        end += -1
        start_ = 0
        end_ = 4
    elif data_course == "code-yourself":
        start_ = 0
        end_ = 4
    elif data_course == "big-data-edu":
        start_ = 1
        end_ = 8

    for topic in topics:
        performance[topic] = {}
        best_fold = -np.inf

        lda_model = LdaModel.load(
            path.join(
                "./out",
                output,
                "baseline/models",
                "{}_{}_{}_topics".format(
                    start_,
                    end_,
                    topic
                )
            )
        )

        proportions, outcome_arr = extract_proportions(
            model=lda_model,
            corpus=corpus_idx,
            keys=corpus_keys,
            outcome=outcome
        )

        logging.info("Training SVM on weeks {} to {} (topics: {})".format(start, end, topic))

        cv = StratifiedKFold(n_splits=folds, random_state=42)
        for train_idx, test_idx in cv.split(corpus_idx, np.ravel(outcome_arr)):
            svm_model = SVC(
                gamma="auto",
                class_weight="balanced"
            )

            svm_model.fit(
                proportions[train_idx],
                np.ravel(outcome_arr[train_idx])
            )

            train_acc = svm_model.score(proportions[train_idx], np.ravel(outcome_arr[train_idx]))
            train_cm = confusion_matrix(np.ravel(outcome_arr[train_idx]), svm_model.predict(proportions[train_idx]))

            train_f1 = f1_score(np.ravel(outcome_arr[train_idx]), svm_model.predict(proportions[train_idx]))
            test_f1 = f1_score(np.ravel(outcome_arr[test_idx]), svm_model.predict(proportions[test_idx]))

            if test_f1 > best_fold:
                cm = confusion_matrix(np.ravel(outcome_arr[test_idx]), svm_model.predict(proportions[test_idx]))
                performance[topic]["test_f1"] = test_f1
                performance[topic]["test_acc"] = svm_model.score(proportions[test_idx], np.ravel(outcome_arr[test_idx]))
                performance[topic]["test_tpr"] = cm[1][1] / (cm[1][0] + cm[1][1])
                performance[topic]["test_tnr"] = cm[0][0] / (cm[0][0] + cm[0][1])
                performance[topic]["test_auc"] = roc_auc_score(np.ravel(outcome_arr[test_idx]), svm_model.predict(proportions[test_idx]))
                performance[topic]["test_cohen"] = cohen_kappa_score(np.ravel(outcome_arr[test_idx]), svm_model.predict(proportions[test_idx]))

                performance[topic]["train_f1"] = train_f1
                performance[topic]["train_acc"] = train_acc
                performance[topic]["train_tpr"] = train_cm[1][1] / (train_cm[1][0] + train_cm[1][1])
                performance[topic]["train_tnr"] = train_cm[0][0] / (train_cm[0][0] + train_cm[0][1])
                performance[topic]["train_auc"] = roc_auc_score(np.ravel(outcome_arr[train_idx]), svm_model.predict(proportions[train_idx]))
                performance[topic]["train_cohen"] = cohen_kappa_score(np.ravel(outcome_arr[train_idx]), svm_model.predict(proportions[train_idx]))

                best_fold = test_f1

        logging.info("Weeks {} to {} train F1 {}".format(start, end, performance[topic]["train_f1"]))
        logging.info("Weeks {} to {} test F1 {}".format(start, end, performance[topic]["test_f1"]))

    pd.DataFrame.from_dict(performance).to_csv(
        path.join(
            "./out",
            output,
            "baseline",
            "{}_{}{}.csv".format(
                start,
                end,
                exp
            )
        ),
        index=True
    )

    return performance


def extract_proportions(model, corpus, keys, outcome):
    """
    Function returns two arrays: one containing the proportions of each topic, the other containing outcome variables.
    :param model: trained LDA model.
    :param corpus: processed event logs.
    :param keys: corpus keys (student IDs).
    :param outcome: pandas DataFrame containing student identifier and student outcome variable for SVM.
    :return:
    """
    proportions = {"topic_{}".format(topic + 1): [] for topic in range(model.num_topics)}
    proportions["session_user_id"] = []
    proportions["outcome"] = []

    for idx, doc in enumerate(corpus):
        doc_topics = model.get_document_topics(doc)

        [proportions["topic_{}".format(item[0] + 1)].append(item[1]) for item in doc_topics]
        proportions["session_user_id"].append(keys[idx])
        proportions["outcome"].append(outcome[outcome.session_user_id == keys[idx]].values[0][1])

        max_len = max([len(proportions[key]) for key in proportions.keys()])
        [proportions[key].append(0) for key in proportions.keys() if len(proportions[key]) < max_len]

    prop = pd.DataFrame.from_dict(proportions)

    return prop[[col for col in prop.columns if col not in ["session_user_id", "outcome"]]].values, \
        prop[["outcome"]].values
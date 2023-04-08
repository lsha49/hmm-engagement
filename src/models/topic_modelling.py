import numpy.distutils.system_info as sys_info
from os import path, mkdir
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.WARNING)
sys_info.get_info("atlas")


def evaluate_topic_models(course, output_dir):
    """
    Train topic models for a given course and session, saving results to the passed output directory.
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
    design = pd.read_csv(
        path.join(
            session_dir,
            "course_design_accessed.csv"
        )
    )

    week_ids = event_logs.parent_week.unique()
    week_indices = zip([min(week_ids) for _ in range(len(week_ids))], week_ids)

    topic_dict = corpora.Dictionary([design.type_id.values.tolist()])
    topics = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    if not path.exists(output_dir):
        mkdir(output_dir)
        mkdir(path.join(output_dir, "models"))

    for start, end in week_indices:
        run_k_fold_topic_modelling(
            data=event_logs[((event_logs.action_week >= start) & (event_logs.action_week <= end))],
            idx2word=topic_dict,
            topics=topics,
            folds=5,
            start=start,
            end=end,
            output=path.join(output_dir, "models"),
        )

    topic_idx_to_design_idx = pd.DataFrame.from_dict(
        {idx: design.index[design["type_id"] == topic_dict[idx]][0] for idx in topic_dict},
        orient="index"
    ).reset_index(drop=False)
    topic_idx_to_design_idx.columns = ["topic_idx", "design_idx"]
    topic_idx_to_design_idx.to_csv(
        path.join(output_dir, "topic_idx_to_design_idx.csv"),
        index=False
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


def plot_log_perplexity(start, end, output_dir):
    """
    Plot the log perplexity of topic models from a given <start, end> slice of a course.
    :param start: week from which event logs start.
    :param end: week from which event logs end.
    :param output_dir: output directory.
    :return:
    """
    params = pd.read_csv(
        path.join(
            output_dir,
            "{}_{}_tm_perplex.csv".format(
                int(start),
                int(end)
            )
        ),
        index_col=[0]
    )

    params = params.transpose().reset_index(drop=False)
    params.columns = ["n_topics", "Test Data", "Training Data"]
    params = pd.melt(params, ["n_topics"])
    params["n_topics"] = params.n_topics.astype("int32")
    params.columns = ["Number of Topics", "", "Mean Log Perplexity"]

    ax = sns.lineplot(
        x="Number of Topics",
        y="Mean Log Perplexity",
        hue="",
        palette={"Training Data": "seagreen", "Test Data": "magenta"},
        style="",
        data=params
    )

    print("----- Log perplexity across use-cases -----")
    print("------------------ Test -------------------")
    x, y = ax.get_lines()[0].get_data()

    for item in zip(x.tolist(), y.tolist()):
        print(item)

    print("------------------ Train ------------------")
    x, y = ax.get_lines()[1].get_data()

    for item in zip(x.tolist(), y.tolist()):
        print(item)

    ax.get_figure().savefig(
        path.join(
            output_dir,
            "{}_{}_log_perplexity.svg".format(
                start,
                end
            )
        )
    )

    return


def plot_topic_distribution(start, end, n_topics, course, output_dir):
    """
    Plot the topic model distribution for a given number of topics
    :param start: week from which event logs start.
    :param end: week from which event logs end.
    :param n_topics: number of topics for which to plot.
    :param course: input directory
    :param output_dir: output directory.
    :return:
    """
    topics = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert n_topics in topics, "n_topics must be in {}".format(str(topics))

    params = pd.read_csv(
        path.join(
            output_dir,
            "{}_{}_tm_params.csv".format(
                start,
                end
            )
        )
    )

    idx_dict = pd.read_csv(
        path.join(
            output_dir,
            "topic_idx_to_design_idx.csv"
        )
    )

    design = pd.read_csv(
        path.join(
            "./data/",
            course,
            "course_design_accessed.csv"
        )
    )
    type_dict = design.type_id.str.rsplit("_", n=1, expand=True)[0].reset_index(drop=False)[0].to_dict()

    params = params[params.n_topics == n_topics]
    params["course_index"] = params.word_idx.apply(
        lambda x: idx_dict.loc[idx_dict.topic_idx == x, "design_idx"].values[0])

    params["course_type"] = params.course_index.apply(lambda x: type_dict[x])
    params.columns = ["n_topics", "Topic", "word_idx", "Probability", "Course Index", "course_type"]

    pal = {"assignment": "red", "lecture": "seagreen", "quiz": "blue", "video": "yellow", "peergrading": "orange"}

    g = sns.FacetGrid(
        data=params,
        row="Topic",
        hue="course_type",
        palette=pal
    )

    g.map(
        sns.barplot,
        "Course Index",
        "Probability",
        order=params["Course Index"].sort_values().unique()
    )

    for ax in g.axes:
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())

    g.savefig(
        path.join(
            output_dir,
            "{}_{}_{}_topic_distribution.svg".format(
                start,
                end,
                n_topics
            )
        )
    )

    return


def extract_certification(session_dir, output_dir):
    """
    Extract user certification and save to output directory.
    :param session_dir: input directory.
    :param output_dir: output directory.
    :return:
    """
    grades = pd.read_csv(
        path.join(
            session_dir,
            "grades.csv"
        )
    )

    achievement_map = {
        "none": 0,
        "normal": 1,
        "distinction": 1
    }

    grades["certification"] = grades.achievement_level.apply(lambda x: achievement_map[x])
    grades = grades[["session_user_id", "certification"]]

    grades.to_csv(
        path.join(
            output_dir,
            "certification.csv"
        ),
        index=False
    )

    return grades


def predict_certification_topic_models(course, output_dir):
    """
    Converts event logs to a corpus and predicts certification following the Probabilistic Use Cases methodology.
    :param course: input directory.
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
    design = pd.read_csv(
        path.join(
            session_dir,
            "course_design_accessed.csv"
        )
    )
    grades = extract_certification(
        session_dir,
        output_dir
    )

    common_students = [i for i in set(grades.session_user_id.values).intersection(set(event_logs.session_user_id.values))]

    event_logs = event_logs[((~event_logs.action_week.isna()) & (event_logs.session_user_id.isin(common_students)))]
    grades = grades[grades.session_user_id.isin(common_students)]

    week_indices = zip([event_logs.parent_week.min() for _ in range(len(event_logs.parent_week.unique()))],
                       event_logs.parent_week.unique())

    topic_dict = corpora.Dictionary([design.type_id.values.tolist()])
    topics = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for start, end in week_indices:
        run_k_fold_svm(
            data=event_logs[((event_logs.action_week >= start) & (event_logs.action_week <= end))],
            outcome=grades,
            idx2word=topic_dict,
            topics=topics,
            folds=5,
            start=start,
            end=end,
            exp="",
            output=output_dir
        )

    return


def run_k_fold_svm(data, outcome, idx2word, topics, folds, start, end, exp, output):
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
    :param output: output directory.
    :return:
    """
    corpus = extract_corpus(data[data.action_week.isin([start, end])], max_diff=1800)

    corpus_idx = np.array([idx2word.doc2bow(corpus[student]) for student in corpus.keys()])
    corpus_keys = [key for key in corpus.keys()]

    performance = {}

    for topic in topics:
        performance[topic] = {}
        best_fold = -np.inf

        lda_model = LdaModel.load(
            path.join(
                output,
                "models",
                "{}_{}_{}_topics".format(
                    start,
                    end,
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

                performance[topic]["train_f1"] = train_f1
                performance[topic]["train_acc"] = train_acc
                performance[topic]["train_tpr"] = train_cm[1][1] / (train_cm[1][0] + train_cm[1][1])
                performance[topic]["train_tnr"] = train_cm[0][0] / (train_cm[0][0] + train_cm[0][1])

                best_fold = test_f1

        logging.info("Weeks {} to {} train F1 {}".format(start, end, performance[topic]["train_f1"]))
        logging.info("Weeks {} to {} test F1 {}".format(start, end, performance[topic]["test_f1"]))

    pd.DataFrame.from_dict(performance).to_csv(
        path.join(
            output,
            "{}_{}{}_svm_perform.csv".format(
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


def predict_last_action_topic_models(course, output_dir):
    """
    Converts event logs to a corpus and predicts, for iteratively longer weekly slices of the course, whether or not a
    given week will be a student's last week taking any action in the course.
    :param course: input directory.
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
    design = pd.read_csv(
        path.join(
            session_dir,
            "course_design_accessed.csv"
        )
    )

    event_logs = event_logs[~event_logs.action_week.isna()]

    w_start, w_unique = (int(event_logs.parent_week.min()), len(event_logs.parent_week.unique()))
    week_indices = zip([w_start for _ in range(w_start, w_unique)], [i for i in range(w_start, w_unique)])

    topic_dict = corpora.Dictionary([design.type_id.values.tolist()])
    topics = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for start, end in week_indices:
        event_logs_slice = remove_already_inactive_users(event_logs, start, end)

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
            exp="_last_action",
            output=output_dir
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


def print_topic_model_parameters(course):
    """
    Print out the topic model parameters, along with graph colours for Tikz.
    :return:
    """
    topics = pd.read_csv(path.join("./out/", course, "PUC/1_8_tm_params.csv"))
    design = pd.read_csv(path.join("./data/", course, "course_design_accessed.csv"))
    tidx_to_didx = pd.read_csv(path.join("./out/", course, "PUC/topic_idx_to_design_idx.csv"))

    tuc_1 = topics[((topics.n_topics == 3.0) & (topics.topic == 1.0))]
    tuc_2 = topics[((topics.n_topics == 3.0) & (topics.topic == 2.0))]
    tuc_3 = topics[((topics.n_topics == 3.0) & (topics.topic == 3.0))]

    tidx_to_didx["type_id"] = tidx_to_didx.design_idx.apply(lambda x: design.iloc[x]["type_id"])
    tidx_to_didx["item_type"] = tidx_to_didx.design_idx.apply(lambda x: design.iloc[x]["item_type"])

    tidx_to_didx["T1"] = tidx_to_didx.topic_idx.apply(lambda x: round(tuc_1[tuc_1.word_idx == float(x)].weight.values[0], 4))
    tidx_to_didx["T2"] = tidx_to_didx.topic_idx.apply(lambda x: round(tuc_2[tuc_2.word_idx == float(x)].weight.values[0], 4))
    tidx_to_didx["T3"] = tidx_to_didx.topic_idx.apply(lambda x: round(tuc_3[tuc_3.word_idx == float(x)].weight.values[0], 4))

    tidx_to_didx = tidx_to_didx.sort_values(["design_idx"]).reset_index(drop=True)

    for idx, row in tidx_to_didx.iterrows():
        print(row.item_type, idx, row.T1, row.T2, row.T3)

    colour_dict = {"lecture": "{green, fill=green},", "video": "{blue, fill=blue},", "quiz": "{red, fill=red},"}

    for _, row in tidx_to_didx.iterrows():
        print(colour_dict[row.item_type])


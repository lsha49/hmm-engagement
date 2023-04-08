import numpy.distutils.system_info as sys_info
from os import path, mkdir
from collections import Counter
import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.svm import SVC
import scipy.stats as stats
from pyhsmm.util.text import progprint_xrange
import pyhsmm
import seaborn as sns
import pickle
import logging
import sys

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.WARNING)
sys_info.get_info("atlas")


def evaluate_hmm(course, output_dir):
    """
    Train HMMs for a given course and session, saving results to the passed output directory.
    :param course: input directory
    :param output_dir: output directory.
    :return:
    """
    session_dir = path.join("./data/", course)

    if not path.exists(output_dir):
        mkdir(output_dir)

    event_logs = pd.read_csv(
        path.join(
            session_dir,
            "events.csv"
        )
    )
    design = pd.read_csv(
        path.join(
            session_dir,
            "course_design.csv"
        )
    )

    if not path.exists(path.join(output_dir, "certification.csv")):
        grades = extract_certification(
            session_dir,
            output_dir
        )
    else:
        grades = pd.read_csv(
            path.join(output_dir, "certification.csv")
        )

    if not path.exists(path.join(output_dir, "action_vectors.csv")):
        students = [stud for stud in set(event_logs.session_user_id).intersection(set(grades.session_user_id))]

        vectors = extract_student_action_vectors(
            event_logs[event_logs.session_user_id.isin(students)],
            design,
            output_dir
        )
    else:
        vectors = pd.read_csv(
            path.join(output_dir, "action_vectors.csv")
        )

    if not path.exists(path.join(output_dir, "training_data.npy")):
        student_vectors = vectors.student
        vectors = vectors[[col for col in vectors.columns if col not in ["student", "week"]]]

        all_data = add_no_observations(vectors, student_vectors, thresh=1).astype(np.int)
        grades = grades[grades.session_user_id.isin(student_vectors)].certification.values

        train, test, g_train, g_test = train_test_split(
            all_data,
            grades,
            test_size=0.2,
            random_state=42,
            stratify=grades
        )

        np.save(path.join(output_dir, "training_data.npy"), train, allow_pickle=True)
        np.save(path.join(output_dir, "testing_data.npy"), test, allow_pickle=True)

        np.save(path.join(output_dir, "training_grades.npy"), g_train, allow_pickle=True)
        np.save(path.join(output_dir, "testing_grades.npy"), g_test, allow_pickle=True)
    else:
        train = np.load(path.join(output_dir, "training_data.npy"), allow_pickle=True)

    try:
        train_hmm(train, [2, 1], output_dir)
    except Exception as e:
        logging.warning(e)
        sys.exit()


def predict_certification_hmm(output_dir):
    """
    Function loads previously saved train, test, and output arrays, generates state arrays from the pickled HDP-HMM
    model, and trains certification classifiers on iteratively larger slices of the state sequences.
    :param output_dir: output directory.
    :return:
    """
    train_data = np.load(path.join(output_dir, "training_data.npy"), allow_pickle=True)
    train_grades = np.load(path.join(output_dir, "training_grades.npy"), allow_pickle=True)

    test_data = np.load(path.join(output_dir, "testing_data.npy"), allow_pickle=True)
    test_grades = np.load(path.join(output_dir, "testing_grades.npy"), allow_pickle=True)

    model = pickle.load(open(path.join(output_dir, "hmm.pickle"), "rb"))

    train_data = np.stack([model.heldout_state_marginals(train) for train in train_data])
    test_data = np.stack([model.heldout_state_marginals(test) for test in test_data])

    week_indices = zip([0 for _ in range(0, train_data.shape[1])], [i for i in range(0, train_data.shape[1])])

    performance = []

    for start, end in week_indices:
        performance.append(
            run_svm(
                train_data=train_data[:, start:end + 1],
                train_out=train_grades,
                test_data=test_data[:, start:end + 1],
                test_out=test_grades,
                start=start + 1,
                end=end + 1
            )
        )

    pd.concat(
        [pd.DataFrame(perform, index=[0]) for perform in performance]
    ).to_csv(path.join(output_dir, "svm_perform.csv"), index=False)

    return


def predict_last_action_hmm(data_dir, output_dir):
    """
    For all actions up to a given week, predicts if a given week contains the last action for each student.
    :param data_dir: data input directory.
    :param output_dir: output directory.
    :return:
    """
    logging.info("Predicting when students take their last action.")

    train_data = np.load(path.join(data_dir, "training_data.npy"), allow_pickle=True)
    test_data = np.load(path.join(data_dir, "testing_data.npy"), allow_pickle=True)

    week_indices = zip([0 for _ in range(0, train_data.shape[1])], [i for i in range(0, train_data.shape[1])])

    performance = []
    last_action = []

    for start, end in week_indices:
        train_data_slice, test_data_slice = remove_already_inactive_users(train_data, test_data, start, end)

        last_action_train = np.zeros(train_data_slice.shape[0])
        last_action_train[train_data_slice[:, end + 1:, :-1].sum(axis=2).sum(axis=1) == 0] = 1

        last_action_test = np.zeros(test_data_slice.shape[0])
        last_action_test[test_data_slice[:, end + 1:, :-1].sum(axis=2).sum(axis=1) == 0] = 1

        logging.info("From week {} to {}, {} students took their last action.".format(
            start + 1,
            end + 1,
            int(last_action_train.sum() + last_action_test.sum())
        ))

        last_action.append({
            "start": start + 1,
            "end": end + 1,
            "remaining_students": int(last_action_train.shape[0] + last_action_test.shape[0]),
            "dropout": int(last_action_train.sum() + last_action_test.sum())
        })

        if len(set(last_action_train).intersection(set(last_action_test))) > 1:
            train_data_slice = train_data_slice[:, start:end + 1]
            test_data_slice = test_data_slice[:, start:end + 1]

            model = pickle.load(open(path.join(output_dir, "hmm.pickle"), "rb"))

            train_data_slice = np.stack([model.heldout_state_marginals(train) for train in train_data_slice])
            test_data_slice = np.stack([model.heldout_state_marginals(test) for test in test_data_slice])

            performance.append(
                run_svm(
                    train_data=train_data_slice,
                    train_out=last_action_train,
                    test_data=test_data_slice,
                    test_out=last_action_test,
                    start=start + 1,
                    end=end + 1
                )
            )

    if len(performance) > 0:
        pd.concat(
            [pd.DataFrame(perform, index=[0]) for perform in performance]
        ).to_csv(path.join(
            output_dir,
            "last_action_svm_perform_{}.csv".format("SAME" if data_dir == output_dir else "DIFFERENT")
        ), index=False)

    if (len(last_action) > 0) and (data_dir == output_dir):
        pd.DataFrame(last_action).to_csv(path.join(output_dir, "weekly_dropout.csv"), index=False)


def run_svm(train_data, train_out, test_data, test_out, start, end):
    """
    Conduct SVM certification classification on training and test sets.
    :param train_data: numpy array of training data between start and end week.
    :param train_out: numpy array of output variables for training data [0, 1].
    :param test_data: numpy array of test data between start and end week.
    :param test_out: numpy array of output variables for test data [0, 1].
    :param start: integer representing start week.
    :param end: integer representing end week.
    :return:
    """
    performance = {}

    svm_model = SVC(
        gamma="auto",
        class_weight="balanced"
    )

    train_data = np.stack([np.hstack(train) for train in train_data])
    test_data = np.stack([np.hstack(test) for test in test_data])

    logging.info("Training SVM on weeks {} to {}".format(start, end))

    svm_model.fit(train_data, train_out)

    train_cm = confusion_matrix(np.ravel(train_out), svm_model.predict(train_data))
    performance["train_acc"] = svm_model.score(train_data, train_out)
    performance["train_tpr"] = train_cm[1][1] / (train_cm[1][0] + train_cm[1][1])
    performance["train_tnr"] = train_cm[0][0] / (train_cm[0][0] + train_cm[0][1])
    performance["train_f1"] = f1_score(np.ravel(train_out), svm_model.predict(train_data))
    performance["train_auc"] = roc_auc_score(np.ravel(train_out), svm_model.predict(train_data))
    performance["train_kappa"] = cohen_kappa_score(np.ravel(train_out), svm_model.predict(train_data))

    test_cm = confusion_matrix(np.ravel(test_out), svm_model.predict(test_data))
    performance["test_acc"] = svm_model.score(test_data, test_out)
    performance["test_tpr"] = test_cm[1][1] / (test_cm[1][0] + test_cm[1][1])
    performance["test_tnr"] = test_cm[0][0] / (test_cm[0][0] + test_cm[0][1])
    performance["test_f1"] = f1_score(np.ravel(test_out), svm_model.predict(test_data))
    performance["test_auc"] = roc_auc_score(np.ravel(test_out), svm_model.predict(test_data))
    performance["test_kappa"] = cohen_kappa_score(np.ravel(test_out), svm_model.predict(test_data))

    performance["start"] = start
    performance["end"] = end

    logging.info("Weeks {} to {} train F1 {}".format(start, end, performance["train_f1"]))
    logging.info("Weeks {} to {} test F1 {}".format(start, end, performance["test_f1"]))

    return performance


def extract_student_action_vectors(logs, design, output):
    """
    Takes a course event log DataFrame and, for each student and each week, calculates action vectors.
    :param logs: event log DataFrame of student interactions with the course.
    :param design: DataFrame containing the course design.
    :param output: output directory.
    :return:
    """
    design_weeks = design.parent_week.unique()

    action_vector = {string.format(item): [] for string, item in
                     itertools.product(["{}_current", "{}_past", "{}_future"], design.item_type.unique().tolist())}

    action_vector["student"] = []
    action_vector["week"] = []

    def extract_student_vector(student):
        """
        For a given student, extract weekly action vectors.
        :param student: student identifier string.
        :return:
        """
        student_vector = action_vector
        student_events = logs[logs.session_user_id == student].sort_values(["timestamp"])

        student_events["tot"] = (student_events.timestamp.shift(-1).fillna(student_events.timestamp.max()) - student_events.timestamp).apply(
            lambda x: round(min(x, 1800) + 1))

        tot = pd.DataFrame(student_events.groupby(["type_id", "action_week"])["tot"].agg("sum")).reset_index(drop=False)

        student_events = pd.merge(
            pd.concat(
                [student_events[student_events.action_week == week].drop_duplicates(subset="type_id", keep="first") for
                 week in student_events.action_week.unique()]
            ),
            tot,
            how="left",
            left_on=["type_id", "action_week"],
            right_on=["type_id", "action_week"]
        )

        for week in design_weeks:
            student_vector = extract_action_vector_tot(
                student_events[student_events.action_week == week],
                student_vector,
                student,
                week
            )

        # ABOVE IS TIME ON TASK. BELOW IS BINARY ACCESS COUNTS
        # student_events = pd.concat(
        #     [student_events[student_events.action_week == week].drop_duplicates(subset="type_id", keep="first") for
        #      week in student_events.action_week.unique()]
        # )

        # for week in design_weeks:
        #     student_vector = extract_action_vector(
        #         student_events[student_events.action_week == week],
        #         student_vector,
        #         student,
        #         week
        #     )

        return student_vector

    student_vectors = Parallel(n_jobs=10, backend="threading")(
        delayed(extract_student_vector)(student) for student in logs.session_user_id.unique()
    )

    student_vectors = pd.concat([pd.DataFrame.from_dict(student) for student in student_vectors])
    student_vectors.to_csv(
        path.join(output, "action_vectors.csv"),
        index=False
    )

    return student_vectors


def extract_action_vector(logs, vector, student, week):
    """
    For a given set of weekly student logs, calculate whether the actions types taken were past, present, or future.
    :param logs: DataFrame of student logs for a given week.
    :param vector: empty action vector for every possible past, present, and future action type.
    :param student: string representing student identifier.
    :param week: int representing week identifier.
    :return:
    """
    vector = copy.deepcopy(vector)
    max_length = max([len(vector[key]) for key in vector.keys()]) + 1

    vector["student"].append(student)
    vector["week"].append(week)

    [vector["{}_current".format(type)].append(count) for type, count in
     Counter(logs[logs.parent_week == logs.action_week].item_type).items()]

    [vector["{}_past".format(type)].append(count) for type, count in
     Counter(logs[logs.parent_week < logs.action_week].item_type).items()]

    [vector["{}_future".format(type)].append(count) for type, count in
     Counter(logs[logs.parent_week > logs.action_week].item_type).items()]

    [vector[key].append(0) for key in vector.keys() if len(vector[key]) < max_length]

    return vector


def extract_action_vector_tot(logs, vector, student, week):
    """
    For a given set of weekly student logs, calculate whether the actions types taken were past, present, or future.
    :param logs: DataFrame of student logs for a given week.
    :param vector: empty action vector for every possible past, present, and future action type.
    :param student: string representing student identifier.
    :param week: int representing week identifier.
    :return:
    """
    vector = copy.deepcopy(vector)
    max_length = max([len(vector[key]) for key in vector.keys()]) + 1

    vector["student"].append(student)
    vector["week"].append(week)

    [vector["{}_current".format(type)].append(count) for type, count in
     [(row[1].item_type, row[1].tot_y) for row in
      logs[logs.parent_week == logs.action_week].groupby(["item_type"])["tot_y"].agg("sum").reset_index(drop=False).iterrows()]]

    [vector["{}_past".format(type)].append(count) for type, count in
     [(row[1].item_type, row[1].tot_y) for row in
      logs[logs.parent_week < logs.action_week].groupby(["item_type"])["tot_y"].agg("sum").reset_index(drop=False).iterrows()]]

    [vector["{}_future".format(type)].append(count) for type, count in
     [(row[1].item_type, row[1].tot_y) for row in
      logs[logs.parent_week > logs.action_week].groupby(["item_type"])["tot_y"].agg("sum").reset_index(drop=False).iterrows()]]

    [vector[key].append(0) for key in vector.keys() if len(vector[key]) < max_length]

    return vector


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
        path.join(output_dir, "certification.csv"),
        index=False
    )

    return grades


def train_hmm(training_data, params, output_dir):
    """
    Function fits a HDP-HMM on training data and pickles the resulting model.
    :param training_data: numpy array of shape (students, weeks, actions).
    :param params: tuple containing obs and dur hyper-parameters.
    :param output_dir: output directory.
    :return:
    """
    np.random.seed(404)

    max_states = 10
    vocab_len = training_data.shape[2]

    obs_hyper_params = {
        "a_0": params[0],
        "b_0": params[1],
        "K": vocab_len
    }

    obs_distns = [pyhsmm.distributions.MultinomialAndConcentration(**obs_hyper_params) for _ in range(max_states)]

    posterior_model = pyhsmm.models.WeakLimitStickyHDPHMM(
        kappa=50,
        alpha_a_0=1.,
        alpha_b_0=1. / 4,
        gamma_a_0=1.,
        gamma_b_0=1. / 4,
        init_state_concentration=10.,
        obs_distns=obs_distns
    )

    for idx, student in enumerate(training_data):
        posterior_model.add_data(student)

    for _ in progprint_xrange(250):                                                     # TODO check this has converged
        posterior_model.resample_model(num_procs=6)

    with open(path.join(output_dir, "hmm.pickle"), "wb") as outfile:
        pickle.dump(posterior_model, outfile, protocol=-1)

    return


def add_no_observations(vectors, students, thresh):
    """
    Standardizes student activity across weeks by determining an activity threshold, and calculating which students do
    not exceed it. A "no_observations" column is added which, for each week, represents the difference between a
    student's action count and the threshold.
    :param vectors: DataFrame containing action counts across the action vector vocabulary.
    :param students: Series containing student IDs matching the rows in vectors.
    :param thresh: float representing the percentile of actions counts to use or an integer representing the threshold.
    :return:
    """
    vectors = vectors.copy()

    vectors = vectors.reset_index(drop=True)
    students = students.reset_index(drop=True)

    row_sums = vectors.sum(axis=1)
    if thresh < 1:
        threshold = row_sums.quantile(thresh)
    else:
        threshold = thresh

    no_observations = (threshold - vectors.sum(axis=1)).apply(lambda x: max(x, 0))
    no_observations.name = "no_obs"

    vectors = pd.merge(vectors, no_observations, left_index=True, right_index=True)

    return np.array([vectors.iloc[students[students == student].index, :].values for student in students.unique()])


def plot_transition_matrix(output_dir):
    """
    Function loads in a pickled HMM object and plots a transition table.
    :param output_dir: output directory.
    :return:
    """
    if not path.exists(path.join(output_dir, "plots")):
        mkdir(path.join(output_dir, "plots"))

    model = pickle.load(
        open(path.join(output_dir, "hmm.pickle"), "rb")
    )

    plt.figure()

    ax = sns.heatmap(
        model.trans_distn.trans_matrix,
        linewidths=0.5,
        cmap="YlOrRd"
    )

    ax.xaxis.set_ticks_position("top")

    ax.figure.savefig(
        path.join(output_dir, "plots", "trans_mat.svg")
    )

    plt.close()

    t_mat = model.trans_distn.trans_matrix
    print("to", "from", "val")
    for _from, _ in enumerate(t_mat):
        for _to, _ in enumerate(t_mat[_from, :]):
            print(_to, _from, t_mat[_from, _to].round(4))
        print("")

    return


def plot_action_probabilities(output_dir, normalised=True):
    """
    Function loads in a pickled HMM object and plots the action probabilities that characterise each state.
    :param output_dir: output directory.
    :param normalised: boolean whether or not to normalise action probabilities.
    :return:
    """
    if not path.exists(path.join(output_dir, "plots")):
        mkdir(path.join(output_dir, "plots"))

    model = pickle.load(
        open(path.join(output_dir, "hmm.pickle"), "rb")
    )

    state_counts = Counter(np.concatenate(model.stateseqs))
    action_weights = np.array([d.weights for d in model.obs_distns])

    if normalised:
        action_weights /= np.sum(action_weights, axis=0)[None, :]

    labels = [col for col in pd.read_csv(path.join(output_dir, "action_vectors.csv")).columns if
              col not in ["student", "week"]]
    labels.append("no_observation")

    print("-------- State Action Distributions -------")
    for k, w in enumerate(action_weights):
        plt.bar(height=w, x=np.arange(len(w)))
        plt.xticks(labels=labels, ticks=np.arange(len(labels)), rotation=90)
        plt.title("State {}: {}".format(k, state_counts[k]))
        plt.tight_layout()
        plt.savefig(path.join(output_dir, "plots", "action_probs_state_{}.svg".format(k)))
        plt.close()

        print("---------------- State {} ------------------".format(k + 1))
        for item in zip(np.arange(1, len(w) + 1).tolist(), w.tolist()):
            print(item)


def remove_already_inactive_users(train, test, start, end):
    """
    Given a start and end week, remove users who became inactive for the rest of the course starting in week end - 1.
    :param train: numpy array, containing training data.
    :param test: numpy array, containing testing data.
    :param start: integer, representing starting week.
    :param end: integer, representing ending week.
    :return:
    """
    if start == end:
        return train, test

    train = train[train[:, end:, :-1].sum(axis=2).sum(axis=1) > 0]
    test = test[test[:, end:, :-1].sum(axis=2).sum(axis=1) > 0]

    return train, test


def print_course_access_distribution(course):
    """
    Prints the values of the course resource access distribution for Tikz.
    :return:
    """
    vectors = pd.read_csv(path.join("./out/", course, "HMM/action_vectors.csv"))
    vectors = vectors[[col for col in vectors.columns if col not in ["student", "week"]]]
    vec = vectors.values
    x, y = sns.distplot(vec.sum(axis=1)).get_lines()[0].get_data()

    for item in zip(x.tolist(), y.tolist()):
        print(item)

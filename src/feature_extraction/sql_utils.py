import os
import subprocess
import shutil
import pandas as pd


DATABASE_NAME = "course"


def execute_mysql_query_into_csv(query, file, database_name=DATABASE_NAME, delimiter=","):
    """
    Execute a mysql query into a file.
    :param query: valid mySQL query as string.
    :param file: csv filename to write to.
    :param database_name: name of database to use.
    :param delimiter: type of delimiter to use in file.
    :return: None
    """
    # mysql_to_csv_cmd = """ | tr '\t' '{}' """.format(delimiter)  # string to properly format result of mysql query
    # command = '''mysql -u root -proot {} -e"{}"'''.format(database_name, query)
    # command += """{} > {}""".format(mysql_to_csv_cmd, file)
    ## BEGIN test
    formatted_query = """{} INTO OUTFILE '{}' FIELDS TERMINATED BY '{}' ;""".format(query, file, delimiter)
    command = '''mysql -u root -proot {} -e"{}"'''.format(database_name, formatted_query)
    ##END test
    subprocess.call(command, shell=True)
    return


def load_mysql_dump(dumpfile, database_name=DATABASE_NAME):
    """
    Load a mySQL data dump into DATABASE_NAME.
    :param file: path to mysql database dump
    :return:
    """
    command = '''mysql -u root -proot {} < {}'''.format(database_name, dumpfile)
    subprocess.call(command, shell=True)
    return


def initialize_database(database_name=DATABASE_NAME):
    """
    Start mySQL service and initialize mySQL database with database_name.
    :param database_name: name of database.
    :return: None
    """
    # start mysql server
    subprocess.call("service mysql start", shell=True)
    # create database
    subprocess.call('''mysql -u root -proot -e "CREATE DATABASE {}"'''.format(database_name), shell=True)
    return


def remove_database(database_name=DATABASE_NAME):
    """
    Remove mySQL database with database_name then restart the server.
    :param database_name: name of database.
    :return: None
    """
    # remove database
    subprocess.call('''mysql -u root -proot -e "DROP DATABASE {}"'''.format(database_name), shell=True)
    # restart mysql server
    subprocess.call("service mysql restart", shell=True)
    return


def extract_coursera_sql_data(course,
                              sections_filename="sections.csv",
                              items_section_filename="item_sections.csv",
                              quiz_metadata_filename="quiz_metadata.csv",
                              lecture_metadata_filename="lecture_metadata.csv",
                              assignments_filename="assignment_submissions.csv",
                              quiz_filename="quiz_submissions.csv",
                              lecture_filename="lecture_submissions.csv",
                              grades_filename="grades.csv"):
    """
    Initialize the mySQL database, load files, and execute queries to deposit csv files of data into /input/course/session directory.
    :param course: string containing the name of the data subdirectory.
    :param sections_filename: name of csv to write sections to.
    :param items_section_filename: name of csv to write item sections to.
    :param quiz_metadata_filename: name of csv to write quiz metadata to.
    :param lecture_metadata_filename: name of csv to write lecture metadata to.
    :param assignments_filename: name of csv to write assignments to.
    :param quiz_filename: name of csv to write quizzes to.
    :param lecture_filename: name of csv to write lectures to.
    :param grades_filename: name of csv to write grades to.
    :return:
    """

    # Paths for reading results
    course_session_dir = os.path.join("./data/", course)

    # Path for writing mysql results
    mysql_default_output_dir = "/var/lib/mysql/{}/".format(DATABASE_NAME)

    sections_fp = os.path.join(course_session_dir, sections_filename)
    items_section_fp = os.path.join(course_session_dir, items_section_filename)
    quiz_metadata_filename_fp = os.path.join(course_session_dir, quiz_metadata_filename)
    lecture_metadata_filename_fp = os.path.join(course_session_dir, lecture_metadata_filename)
    assignments_fp = os.path.join(course_session_dir, assignments_filename)
    quiz_fp = os.path.join(course_session_dir, quiz_filename)
    lecture_fp = os.path.join(course_session_dir, lecture_filename)
    grades_fp = os.path.join(course_session_dir, grades_filename)

    general_sql_dump = \
        [x for x in os.listdir(course_session_dir) if "anonymized_general" in x][0]
    hash_mapping_sql_dump = \
        [x for x in os.listdir(course_session_dir) if "hash_mapping" in x][0]
    forum_sql_dump = \
        [x for x in os.listdir(course_session_dir) if "anonymized_forum" in x][0]

    # Create DATABASE_NAME database and populate with course +/ session sql dumps
    initialize_database()
    load_mysql_dump(os.path.join(course_session_dir, general_sql_dump))
    load_mysql_dump(os.path.join(course_session_dir, forum_sql_dump))
    load_mysql_dump(os.path.join(course_session_dir, hash_mapping_sql_dump))

    # Execute items sections query and send to csv
    query = """
            SELECT
                id,
                title,
                display_order
            FROM
                sections
            """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, sections_filename))

    query = """
            SELECT 
                * 
            FROM 
                items_sections
            """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, items_section_filename))

    # Execute quiz metadata query and send to csv
    query = """
                SELECT 
                    id,
                    parent_id,
                    title,
                    quiz_type,
                    soft_close_time 
                FROM 
                    quiz_metadata
                WHERE
                    deleted = 0
                """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, quiz_metadata_filename))

    # Execute lecture metadata query and send to csv
    query = """
                SELECT 
                    id,
                    parent_id,
                    quiz_id,
                    title 
                FROM 
                    lecture_metadata
                WHERE
                    deleted = 0
                """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, lecture_metadata_filename))

    # Execute assignment submission query and send to csv
    query = """
                SELECT 
                    CONCAT('assignment_', assignment_submission_metadata.item_id) AS item_type,
                    assignment_submission_metadata.session_user_id, 
                    assignment_submission_metadata.submission_time
                FROM 
                    assignment_submission_metadata 
                """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, assignments_filename))

    # Execute quiz submission query and send to csv
    query = """
                SELECT 
                    CONCAT('quiz_', quiz_submission_metadata.item_id) AS item_type,
                    session_user_id, 
                    submission_time 
                FROM 
                    quiz_submission_metadata
                """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, quiz_filename))

    # Execute lecture query and send to csv
    query = """
                SELECT 
                    CONCAT('lecture_', lecture_submission_metadata.item_id) AS item_type,
                    session_user_id, 
                    submission_time
                FROM 
                    lecture_submission_metadata
                """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, lecture_filename))

    # Execute grades query and send to csv
    query = """
                SELECT
                    session_user_id,
                    normal_grade,
                    distinction_grade,
                    achievement_level
                FROM
                    course_grades
                """
    execute_mysql_query_into_csv(query, file="{}{}".format(mysql_default_output_dir, grades_filename))

    # Move files to intended location
    shutil.move(os.path.join(mysql_default_output_dir, sections_filename), sections_fp)
    change_columns(sections_fp, ["section_id", "title", "display_order"])

    shutil.move(os.path.join(mysql_default_output_dir, items_section_filename), items_section_fp)
    change_columns(items_section_fp, ["item_type", "item_id", "section_id", "order"])

    shutil.move(os.path.join(mysql_default_output_dir, quiz_metadata_filename), quiz_metadata_filename_fp)
    change_columns(quiz_metadata_filename_fp, ["item_id", "parent_id", "title", "item_type", "timestamp"])

    shutil.move(os.path.join(mysql_default_output_dir, lecture_metadata_filename), lecture_metadata_filename_fp)
    change_columns(lecture_metadata_filename_fp, ["id", "parent_id", "quiz_id", "title"])

    shutil.move(os.path.join(mysql_default_output_dir, assignments_filename), assignments_fp)
    change_columns(assignments_fp, ["type_id", "session_user_id", "timestamp"])

    shutil.move(os.path.join(mysql_default_output_dir, quiz_filename), quiz_fp)
    change_columns(quiz_fp, ["type_id", "session_user_id", "timestamp"])

    shutil.move(os.path.join(mysql_default_output_dir, lecture_filename), lecture_fp)
    change_columns(lecture_fp, ["type_id", "session_user_id", "timestamp"])

    shutil.move(os.path.join(mysql_default_output_dir, grades_filename), grades_fp)
    change_columns(grades_fp, ["session_user_id", "normal_grade", "distinction_grade", "achievement_level"])

    remove_database()

    return


def change_columns(csvfile, columns):
    """
    Add course and session columns to csvfile.
    :param csvfile: path to csv.
    :param columns: columns.
    :return:
    """
    df = pd.read_csv(
        csvfile,
        delimiter=',',
        quotechar='"',
        escapechar='\\',
        error_bad_lines=False,
        dtype=object,
        header=None
    )
    df.columns = columns
    df.to_csv(
        csvfile,
        index=False
    )
    return

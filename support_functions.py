import sys
import os
import re
import sqlite3
import json
import pickle
from tqdm import tqdm
import numpy as np

# path to https://github.com/taoyds/test-suite-sql-eval
sys.path.append("/Users/somov-od/Documents/phd/projects/sql_testing_suite")

import evaluation
from sklearn.model_selection import train_test_split
import uncertainty_data_path_consts as data_constants

EXECUTION_CACHE=dict()


def create_split(scores_matrix, target_matrix):
    """
    Split the scores and target matrices into development and test sets.

    Args:
        scores_matrix (numpy.ndarray): A matrix of scores where each column corresponds to a seed.
        target_matrix (numpy.ndarray): A matrix of target values where each column corresponds to a seed.

    Returns:
        tuple: A tuple containing:
            - dev_scores_matrix (numpy.ndarray): Development set scores.
            - dev_target_matrix (numpy.ndarray): Development set targets.
            - test_scores_matrix (numpy.ndarray): Test set scores.
            - test_target_matrix (numpy.ndarray): Test set targets.
    """
    seed_number = range(scores_matrix.shape[1])
    dev_scores_matrix, test_scores_matrix = [], []
    dev_target_matrix, test_target_matrix = [], []
    for i in seed_number:
        dev_scores, test_scores, dev_target, test_target = train_test_split(scores_matrix[:, i], target_matrix[:, i],
                                                                            test_size=0.66, random_state=42)
        dev_scores_matrix.append(dev_scores)
        test_scores_matrix.append(test_scores)
        dev_target_matrix.append(dev_target)
        test_target_matrix.append(test_target)

    dev_scores_matrix = np.array(dev_scores_matrix).T
    dev_target_matrix = np.array(dev_target_matrix).T
    test_scores_matrix = np.array(test_scores_matrix).T
    test_target_matrix = np.array(test_target_matrix).T

    return dev_scores_matrix, dev_target_matrix, test_scores_matrix, test_target_matrix


def create_exec_match_dict(test_list, preds_dict, split_name):
    """
    Create a dictionary of execution match results for each query in the test set.

    Args:
        test_list (list): List of test samples.
        preds_dict (dict): Dictionary of predicted SQL queries.
        split_name (str): Name of the dataset split.

    Returns:
        dict: A dictionary mapping sample IDs to execution match results (0 for success, 1 for failure).
    """
    exec_per_query = dict()
    for sample in tqdm(test_list):
        sample_id = sample['id']
        gold_query = sample['sql']
        pred_query = preds_dict[sample_id]['sql']

        cache_string = f"{gold_query}|{pred_query}|{split_name}"
        if cache_string not in EXECUTION_CACHE:
            if split_name != 'ehrsql':
                db_id = sample['db_id']
                db_path = os.path.join(data_constants.PAUQ_DB_PATH, db_id, db_id + ".sqlite")
                exec_match_result = eval_exec_match(gold_query, pred_query, db_path)
            else:
                exec_match_result = eval_ehrsql_match(gold_query, pred_query, data_constants.EHRSQL_MIMIC_PATH)

            EXECUTION_CACHE[cache_string] = exec_match_result
        else:
            exec_match_result = EXECUTION_CACHE[cache_string]

        exec_per_query[sample_id] = 0 if exec_match_result == 1 else 1

    return exec_per_query


def read_gold_dataset_test(split_name, split_gold_path_dict):
    """
    Read the gold dataset for a given split.

    Args:
        split_name (str): Name of the dataset split.
        split_gold_path_dict (dict): Dictionary mapping split names to their file paths.

    Returns:
        list: A list of samples, each containing 'id', 'question', 'sql', and 'db_id'.
    """
    split_path = split_gold_path_dict[split_name]
    split = json.load(open(split_path, 'r'))
    split_list = []
    if split_name != 'ehrsql':
        for sample in split:
            new_sample = {"id": sample['id'],
                          'question': sample['question'],
                          "sql": sample['query'],
                          "db_id": sample['db_id']}
            split_list.append(new_sample)
    else:
        # Path to EHRSQL test data questions
        questions_list = json.load(
            open("/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/data/mimic_iv/test/data.json",
                 'r'))['data']
        questions_dict = {sample['id']: sample['question'] for sample in questions_list}
        for key in split:
            new_sample = {"id": key,
                          "sql": split[key],
                          "question": questions_dict[key],
                          "db_id": 'mimic_iv'}
            split_list.append(new_sample)

    return split_list


def read_preds_dataset_test(split_name, model_name, seed, split_prediction_path_dict):
    """
    Read the predicted dataset for a given split, model, and seed.

    Args:
        split_name (str): Name of the dataset split.
        model_name (str): Name of the model.
        seed (int): Seed value used for the experiment.
        split_prediction_path_dict (dict): Dictionary mapping models and splits to their prediction paths.

    Returns:
        dict: Dictionary of predictions for the given split, model, and seed.
    """
    preds = None
    model_preds_dict = split_prediction_path_dict.get(model_name)
    if model_preds_dict:
        split_pred_path = model_preds_dict.get(split_name)
        if split_pred_path:
            prediction_path = split_pred_path.format(seed=seed)
            preds = pickle.load(open(prediction_path, 'rb'))
    return preds


def make_scores_array(splits_file, preds_dict):
    """
    Create an array of prediction scores for the given split.

    Args:
        splits_file (list): List of samples from the split.
        preds_dict (dict): Dictionary of predicted scores.

    Returns:
        list: List of prediction scores.
    """
    scores_list = []
    for sample in splits_file:
        sample_id = sample['id']
        prediction_score = preds_dict[sample_id]['score']
        scores_list.append(prediction_score)
    return scores_list


def make_execution_result_array(splits_file, prediction_dict, split_name):
    """
    Create an array of execution results for the given split.

    Args:
        splits_file (list): List of samples from the split.
        prediction_dict (dict): Dictionary of predicted SQL queries.
        split_name (str): Name of the dataset split.

    Returns:
        list: List of execution results (0 for success, 1 for failure).
    """
    execution_list = []
    exec_result_dict = create_exec_match_dict(splits_file, prediction_dict, split_name)
    for sample in splits_file:
        sample_id = sample['id']
        execution_status = exec_result_dict[sample_id]
        execution_list.append(execution_status)
    return execution_list


def make_numpy_arrays(split_name, model_name, seed_list, split_prediction_path_dict, split_gold_path_dict):
    """
    Create numpy arrays of prediction scores and execution results for a given split and model.

    Args:
        split_name (str): Name of the dataset split.
        model_name (str): Name of the model.
        seed_list (dict): Dictionary mapping models to their list of seeds.
        split_prediction_path_dict (dict): Dictionary mapping models and splits to their prediction paths.
        split_gold_path_dict (dict): Dictionary mapping splits to their gold dataset paths.

    Returns:
        tuple: A tuple containing:
            - prediction_scores_matrix (numpy.ndarray): Matrix of prediction scores.
            - execution_result_matrix (numpy.ndarray): Matrix of execution results.
    """
    split_test = read_gold_dataset_test(split_name, split_gold_path_dict)
    awailable_splits = seed_list[model_name]

    prediction_scores_matrix = []
    execution_result_matrix = []
    for seed in awailable_splits:
        prediction_file = read_preds_dataset_test(split_name, model_name, seed, split_prediction_path_dict)
        if prediction_file:
            scores_array = make_scores_array(split_test, prediction_file)
            prediction_scores_matrix.append(scores_array)

            execution_array = make_execution_result_array(split_test, prediction_file, split_name)
            execution_result_matrix.append(execution_array)

    if len(prediction_scores_matrix) != 0:
        prediction_scores_matrix = np.array(prediction_scores_matrix).T
        execution_result_matrix = np.array(execution_result_matrix).T

        print(prediction_scores_matrix.shape, execution_result_matrix.shape)
        return prediction_scores_matrix, execution_result_matrix
    else:
        print(f'No prediction for {model_name} for {split_name}!')
        return None, None



def parse_sql(query, db_path):
    """
    Parse an SQL query into its components.

    Args:
        query (str): The SQL query.
        db_path (str): Path to the database.

    Returns:
        dict: Parsed SQL query components.
    """
    schema = evaluation.Schema(evaluation.get_schema(db_path))
    try:
        parsed_query = evaluation.get_sql(schema, query)
    except:
        parsed_query = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
        }
    return parsed_query


def eval_exec_match(gold_query, pred_query, db_path):
    """
    Evaluate the execution match between a gold SQL query and a predicted SQL query.

    Args:
        gold_query (str): The gold SQL query.
        pred_query (str): The predicted SQL query.
        db_path (str): Path to the database.

    Returns:
        int: 1 if the execution matches, 0 otherwise.
    """
    return evaluation.eval_exec_match(db=db_path, p_str=pred_query, g_str=gold_query,
                                      plug_value=False, keep_distinct=False,
                                      progress_bar_for_each_datapoint=False,
                                      run_async=False)


def eval_ehrsql_match(gold_query, pred_query, db_path):
    """
    Evaluate the execution match for EHRSQL queries.

    Args:
        gold_query (str): The gold SQL query.
        pred_query (str): The predicted SQL query.
        db_path (str): Path to the database.

    Returns:
        bool: True if the execution matches, False otherwise.
    """
    __current_time = "2100-12-31 23:59:00"
    __precomputed_dict = {
        'temperature': (35.5, 38.1),
        'sao2': (95.0, 100.0),
        'heart rate': (60.0, 100.0),
        'respiration': (12.0, 18.0),
        'systolic bp': (90.0, 120.0),
        'diastolic bp': (60.0, 90.0),
        'mean bp': (60.0, 110.0)
    }

    def post_process_sql(query):
        """
        Post-process the SQL query to standardize formatting and replace placeholders.

        Args:
            query (str): The SQL query.

        Returns:
            str: The processed SQL query.
        """
        query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()
        query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')

        if "current_time" in query:
            query = query.replace("current_time", f"'{__current_time}'")
        if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
            vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
            vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
            vital_name_list = list(
                set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper',
                                                                                       vital_upper_expr)))
            if len(vital_name_list) == 1:
                processed_vital_name = vital_name_list[0].replace('_', ' ')
                if processed_vital_name in __precomputed_dict:
                    vital_range = __precomputed_dict[processed_vital_name]
                    query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr,
                                                                                         f"{vital_range[1]}")

        query = query.replace("%y", "%Y").replace('%j', '%J')
        return query

    def process_answer(ans):
        """
        Process the query answer to a standardized format.

        Args:
            ans (list): The query result.

        Returns:
            str: The processed answer.
        """
        return str(sorted([str(ret) for ret in ans[:100]]))  # Check only up to 100th record

    def execute(sql, db_path, skip_indicator='null'):
        """
        Execute an SQL query on the database.

        Args:
            sql (str): The SQL query.
            db_path (str): Path to the database.
            skip_indicator (str): Indicator to skip execution.

        Returns:
            str: The processed query result or skip indicator.
        """
        if sql != skip_indicator:
            con = sqlite3.connect(db_path)
            con.text_factory = lambda b: b.decode(errors="ignore")
            cur = con.cursor()
            result = cur.execute(sql).fetchall()
            con.close()
            return process_answer(result)
        else:
            return skip_indicator

    def execute_query(sql1, sql2, db_path):
        """
        Execute two SQL queries and compare their results.

        Args:
            sql1 (str): The first SQL query.
            sql2 (str): The second SQL query.
            db_path (str): Path to the database.

        Returns:
            dict: Dictionary containing the results of both queries.
        """
        try:
            result1 = execute(sql1, db_path)
        except:
            result1 = 'error1'
        try:
            result2 = execute(sql2, db_path)
        except:
            result2 = 'error2'
        result = {'real': result1, 'pred': result2}
        return result

    gold_query = post_process_sql(gold_query)
    pred_query = post_process_sql(pred_query)
    query_result = execute_query(gold_query, pred_query, db_path)

    exec_result = (query_result['real'] == query_result['pred'])
    return exec_result


def parse_complex_nested_dict_values(d):
    """
    Parse and extract values from a complex nested dictionary.

    Args:
        d (dict): The nested dictionary.

    Yields:
        Any: Extracted values from the dictionary.
    """
    service_values = ['table_unit', 'and', 'or', 'desc', 'asc']
    if isinstance(d, dict):
        for value in d.values():
            if isinstance(value, dict) or isinstance(value, list) or isinstance(value, tuple):
                yield from parse_complex_nested_dict_values(value)
            elif value not in service_values:
                yield value
    elif isinstance(d, list) or isinstance(d, tuple):
        for value in d:
            if isinstance(value, dict) or isinstance(value, list) or isinstance(value, tuple):
                yield from parse_complex_nested_dict_values(value)
            elif value not in service_values:
                yield value
    elif value not in service_values:
        yield d


def filter_list(l):
    """
    Filter a list to exclude boolean values and convert other values to strings.

    Args:
        l (list): The list to filter.

    Returns:
        list: The filtered list.
    """
    extracted_value = []
    for t in l:
        if isinstance(t, bool):
            continue
        elif isinstance(t, str):
            extracted_value.append(t)
        elif isinstance(t, float):
            extracted_value.append(str(t))
    return extracted_value


def get_sql_variables(query, db_path):
    """
    Extract attributes, tables, and values from an SQL query.

    Args:
        query (str): The SQL query.
        db_path (str): Path to the database.

    Returns:
        list: List of extracted SQL variables.
    """
    sql_dict = parse_sql(query, db_path)
    dict_values = parse_complex_nested_dict_values(sql_dict)
    extracted_vals = filter_list(dict_values)
    return extracted_vals



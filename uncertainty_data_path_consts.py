# Paths to the gold test splits for different datasets
GOLD_TEST_SPLIT_PATH = {
    "pauq_xsp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/data/raw_splits/pauq/pauq_xsp_test.json",
    "template_ssp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/data/raw_splits/my_cp_splits/template_ssp_test.json",
    "tsl_ssp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/data/raw_splits/my_cp_splits/tsl_ssp_test.json",
    "ehrsql": "/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/data/mimic_iv/test/label.json"
}

# Paths to the model predictions for different datasets and models
SPLITS_PREDICTIONS_PATH = {
    't5-large': {
        "pauq_xsp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/experiments/t5-large_pauq_xsp_s{seed}/pauq_xsp_test_inference_result.pkl",
        "template_ssp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/experiments/t5-large_template_ssp_s{seed}/template_ssp_test_inference_result.pkl",
        "tsl_ssp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/experiments/t5-large_tsl_ssp_s{seed}/tsl_ssp_test_inference_result.pkl",
        "ehrsql": "/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/training_trials/t5-large_s{seed}/ehrsql_test_for_t5_inference_result.pkl"
    },
    't5-3b': {
        "pauq_xsp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/experiments/t5-3b_pauq_xsp_s{seed}/pauq_xsp_test_inference_result.pkl",
        "template_ssp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/experiments/t5-3b_template_ssp_s{seed}/template_ssp_test_inference_result.pkl",
        "tsl_ssp": "/Users/somov-od/Documents/phd/projects/naacl_cp_t5/experiments/t5-3b_tsl_ssp_s{seed}/tsl_ssp_test_inference_result.pkl",
        "ehrsql": "/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/training_trials/t5-3b_s{seed}/ehrsql_test_for_t5_inference_result.pkl"
    },
    'dailsql': {
        "pauq_xsp": "/Users/somov-od/Documents/phd/projects/DAILSQL/dailsql_pauq_xsp_s{seed}_predicted_dict.pkl",
        "template_ssp": "/Users/somov-od/Documents/phd/projects/DAILSQL/dailsql_template_ssp_s{seed}_predicted_dict.pkl",
        "tsl_ssp": "/Users/somov-od/Documents/phd/projects/DAILSQL/dailsql_tsl_ssp_s{seed}_predicted_dict.pkl",
        "ehrsql": "/Users/somov-od/Documents/phd/projects/DAILSQL/dailsql_ehrsql_s{seed}_predicted_dict.pkl",
    },
    "llama3_lora": {
        "pauq_xsp": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/pauq_pauq_xsp_s{seed}_lora/pauq_xsp_test_inference_result.pkl",
        "template_ssp": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/pauq_template_ssp_s{seed}_lora/template_ssp_test_inference_result.pkl",
        "tsl_ssp": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/pauq_tsl_ssp_s{seed}_lora/tsl_ssp_test_inference_result.pkl",
        "ehrsql": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/ehrsql_s{seed}_lora_more_epochs/test_inference_result.pkl"
    },
    "llama3_sft": {
        "pauq_xsp": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/pauq_pauq_xsp_s{seed}_sft_1_epoch/pauq_xsp_test_inference_result.pkl",
        "template_ssp": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/pauq_template_ssp_s{seed}_sft_1_epoch/template_ssp_test_inference_result.pkl",
        "tsl_ssp": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/pauq_tsl_ssp_s{seed}_sft_1_epoch/tsl_ssp_test_inference_result.pkl",
        "ehrsql": "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/experiments/ehrsql_s{seed}_sft_3_epoch/test_inference_result.pkl"
    },
}

# List of seeds used for experiments with different models
SEED_LIST = {
    "t5-large": [1, 42, 123],
    "t5-3b": [1, 42, 123],
    "llama3_lora": [1, 42, 123],
    "dailsql": [1],
    "llama3_sft": [1, 42, 123]
}

# Mapping of model names to their display names
MODEL_NAMES = {
    "t5-large": "T5-large",
    "t5-3b": "T5-3B",
    "llama3_lora": "Llama3-8B LoRA",
    "dailsql": "DIAL-SQL",
    "llama3_sft": "Llama3-8B SFT"
}

# Mapping of dataset splits to their display names
SPLITS_NAMES = {
    'pauq_xsp': "PAUQ XSP",
    'template_ssp': "Template SSP split",
    'tsl_ssp': "TSL SSP split",
    "ehrsql": "EHRSQL"
}

# List of dataset names
dataset_names = ['pauq', 'ehrsql']

# List of model names
models_name = ['t5-large', 't5-3b', 'dailsql', 'llama3_lora', 'llama3_sft']

# Paths to the database files for PAUQ and EHRSQL datasets
PAUQ_DB_PATH = "/Users/somov-od/Documents/phd/datasets/pauq/pauq_databases"
EHRSQL_MIMIC_PATH = "/Users/somov-od/Documents/phd/projects/ehrsql-text2sql-solution_statics/data/mimic_iv/mimic_iv.sqlite"
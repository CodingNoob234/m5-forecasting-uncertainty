DATA_BASE_PATH = '../data/m5-forecasting-accuracy/'
DATA_BASE_PATH_UNCERTAINTY = '../data/m5-forecasting-uncertainty/'
SALES_EVALUATION = 'sales_train_evaluation.csv'
SALES_VALIDATION = 'sales_train_validation.csv'
CALENDAR = 'calendar.csv'
SAMPLE_SUBMISSION = 'sample_submission.csv'
SELL_PRICES = 'sell_prices.csv'

PRECOMPUTED_BASE_PATH = '../data/uncertainty/features/'

DAYS: int = 28
QUANTILES: int = [0.005, 0.025, 0.165, 0.25, 0.50, 0.75, 0.835, 0.975, 0.995]

AGG_LEVEL_COLUMNS = {
    "Level1": [], # no grouping, sum of all
    "Level2": ['state_id'],
    "Level3": ['store_id'],
    "Level4": ['cat_id'],
    "Level5": ['dept_id'],
    "Level6": ['state_id', 'cat_id'],
    "Level7": ['state_id', 'dept_id'],
    "Level8": ['store_id', 'cat_id'],
    "Level9": ['store_id', 'dept_id'],
    "Level10": ['item_id'],
    "Level11": ['state_id', 'item_id'],
    "Level12": ['item_id','store_id'],
}

D_CROSS_VAL_START_LIST = [1802, 1830, 1858, 1886, 1914]

TEST_PATH = 'test/'
PREDICTION_BASE_PATH = '../data/uncertainty/temp_submissions/'
SUBMISSION_BASE_PATH = '../data/uncertainty/final_submissions/'

SUB_D_START_VAL: int = 1914
SUB_D_START_EVAL: int = 1914 + 28

# the columns are always included after feature processing
# because they are required in the training and submission format
DROP_FEATURE_COLUMNS: list = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'd', 'sold']
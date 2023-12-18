import pandas as pd
import numpy as np
from utils.configure_logger import configure_logger
configure_logger()
import logging
logger = logging.getLogger(__name__)

def df_to_submission(df: pd.DataFrame):
    """
    Transform df to submission format.
    Columns required: 'id', 'd', 'sold'
    """
    DAYS: int = 28
    df_pivot = df.pivot(index="id", columns="d", values="sold").reset_index(drop=False)
    df_pivot.columns = ["id"] + [f"F{i}" for i,_ in enumerate(range(DAYS),1)]
    return df_pivot

def merge_eval_sold_on_df(df: pd.DataFrame, df_eval: pd.DataFrame):
    """ 
    The validation dataframe does not contain the sold information.
    By merging the sold information from the evaluation set.
    """
    df = df.drop(['sold'], axis = 1)
    df = pd.merge(
        df,
        df_eval[['item_id', 'store_id', 'd', 'sold']],
        how = 'left',
        on = ['item_id', 'store_id', 'd'],
    )
    return df

def sort_df_on_d(df: pd.DataFrame):
    """ 
    Sort df based on d (which has format d_{i}).
    d is a string, and first has to be split
    """
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values(by = 'd_int')
    del df['d_int']
    return df

def _down_cast(df)->pd.DataFrame:
    """ reduce memory usage """
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df

def data_preprocessing(df, calendar, sell_prices, sales_train_nrows=None):
    DAYS = 28
    _, last_date_idx = df.columns[-1].split('_')
    submission_idx = range(int(last_date_idx) + 1, int(last_date_idx) + 1 + DAYS)
    df = pd.concat(
        [
            df, 
            pd.DataFrame(
                [
                    {f"d_{i}": float("NaN") for i in submission_idx} 
                    for _ in range(df.shape[0])
                ]
            )
        ], 
        axis=1
    )
    df = pd.melt(
        frame=df, 
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d", 
        value_name="sold"
    )
    df = pd.merge(
        left=df, 
        right=calendar, 
        how="left", 
        on="d"
    )
    df = pd.merge(
        left=df, 
        right=sell_prices, 
        on=["store_id", "item_id", "wm_yr_wk"], 
        how="left"
    )
    release_dates = sell_prices.groupby(["store_id", "item_id"])["wm_yr_wk"].agg(["min"]).reset_index()
    release_dates.columns = ["store_id", "item_id", "release"]
    df = pd.merge(
        left=df,
        right=release_dates, 
        how="left", 
        on=["store_id", "item_id"]
    )
    return df, submission_idx

class customIter:
    """ 
    tqdm has bugs in jupyter notebooks (at least on my device). This works fine as well.
    ONLY works for pd.Grouper instances
    """
    def __init__(self, x: pd.Grouper):
        self.x: pd.Grouper = x
        
    def __iter__(self):
        i = 0
        tot_iterations = self.x.ngroups
        print(f'{i} / {tot_iterations}', end='\r') # example; 1 / 1000
        for id, group in self.x:
            yield id, group
            i += 1
            print(f'{i} / {tot_iterations}', end='\r')
            
def diff_lists(l1: list, l2: list):
    """ return all elements occurring in l1 but not in l2 """
    return list(set(l1) - set(l2))

def log_status(function):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(function.__name__)
        logger.info('calling')
        return function(*args, **kwargs)
    return wrapper

def add_features(function):
    def wrapper(*args, **kwargs):
        # get params
        if len(args) < 2:
            raise Exception('Provide at least df, features')
        df = args[0]
        features = args[1]
        
        # check instances
        if not isinstance(df, pd.DataFrame):
            error_msg = f'df not of type pd.DataFrame, but {type(df)}'
            raise Exception(error_msg)
        if not isinstance(features, list):
            error_msg = f'features not of type list, but {type(features)}'
            raise Exception(error_msg)
        
        # compute features and add to list
        old_columns = set(df.columns)
        result = function(*args, **kwargs)
        new_columns = set(df.columns)
        features += list(new_columns - old_columns)

        return result, features
    return wrapper
        
def ensemble_submissions(files: list):
    """ 
    Pass a list of strings containing full file paths. 
    The will be read, concatenated and grouped to compute the ensemble mean 
    """
    concat_df = pd.concat([pd.read_csv(file_name) for file_name in files])
    df_pred_avg = concat_df.groupby(['id', 'd'])['pred'].mean().reset_index()
    return df_pred_avg

def ensemble_submissions_uncertainty(files: list):
    concat_df = pd.concat([pd.read_csv(file_name) for file_name in files])
    # the second dot '.' is wrong, should be '_'. 
    # however, fixing this will break the current WSPL calculation
    # thus fix this another time, this is the correct line
    #
    # concat_df['id'] = concat_df['agg_column1'] + '_' + concat_df['agg_column2'] + '_' + concat_df['quantile'].astype(str) + '_' + concat_df['type_of']
    #

    concat_df['id'] = concat_df['agg_column1'] + '_' + concat_df['agg_column2'] + '.' + concat_df['quantile'].astype(str) + '_' + concat_df['type_of']
    df_pred_avg = concat_df.groupby(['id', 'd'])['pred'].mean().reset_index()
    return df_pred_avg

def parse_columns_to_string(c: list):
    """
    Concatenate columns to string
    """
    n = len(c)
    if n == 0:
        return 'Total_X'
    elif c[0] == 'temp_id': 
        return 'Total_X'
    elif n == 1:
        return f'{c[0]}_X'
    return '_'.join(c)

def prefixes_in_column(column:str, prefixes: list):
    """ Check if a column contains any of the prefixes """
    s = sum([column.startswith(prefix) for prefix in prefixes])
    return True if s>0 else False

def store_results_as_json(results, file_path):
    """ Store dictionary containing results under 'file_path' """
    import json
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file) 
        
def load_results_as_json(file_path):
    """ Load file under 'file_path' and parse to dictionary """
    import json
    with open(file_path, 'r') as json_file:
        json_loaded = json.load(json_file) 
    return json_loaded
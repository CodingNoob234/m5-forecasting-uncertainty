import pandas as pd
import numpy as np
import pickle as pkl
import time
from utils.configure_logger import configure_logger
configure_logger()
import logging
logger = logging.getLogger(__name__)

def df_to_submission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform df to submission format.
    Columns required: 'id', 'd', 'sold'
    """
    DAYS: int = 28
    df_pivot = df.pivot(index="id", columns="d", values="sold").reset_index(drop=False)
    df_pivot.columns = ["id"] + [f"F{i}" for i,_ in enumerate(range(DAYS),1)]
    return df_pivot

def merge_eval_sold_on_df(df: pd.DataFrame, df_eval: pd.DataFrame) -> pd.DataFrame:
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

def sort_df_on_d(df: pd.DataFrame)->pd.DataFrame:
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values(by = 'd_int')
    del df['d_int']
    return df

def WRMSSE(df: pd.DataFrame, load_weights: bool = True):
    """ 
    Calculate WRMSSE 
    The datframe should contain the following columns
    - state_id, store_id, cat_id, dept_id, item_id (for aggregated evaluations)
    - id, d, pred (only for prediction indices)
    - sold for ALL rows in df
    - make sure to drop all rows before release date.
    """
    
    D_PRED = [f"d_{i}" for i in range(1914, 1914 + 28)]
    
    # with open(PATH, "rb") as file:
    #     idx_train, idx_predict, id, features, targets, d_list = pkl.load(file)

    logger.info('reading weights file')
    weights = pd.read_csv('../data/weights_validation.csv')
    # rmsse_denominator = pd.read_csv('../data/rmsse_denominator.csv')
    
    agg_levels_columns = {
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
    
    level_wrmsse_list = []
    for agg_level in agg_levels_columns.keys():
        
        # aggregate specific level
        agg_columns = agg_levels_columns[agg_level] + ['d']
        df_agg = df.groupby(agg_columns).agg({'sold': np.sum, 'pred': np.sum}).reset_index(drop=False)
        
        df_agg['d_int'] = df_agg['d'].apply(lambda x: int(x.split('_')[1]))
        df_agg = df_agg.sort_values('d_int')
        
        # select historic dataframe for denominator calculation and prediction df
        pred_index = df_agg['d'].isin(D_PRED)
        df_hist = df_agg.drop(pred_index, errors = "ignore")
        df_pred = df_agg[pred_index]
        
        # rmsse list
        if agg_level == "Level1":
            rmsse_list = [
                (
                    'Total', 'X', 
                    RMSSE(
                        df_pred['pred'], 
                        df_pred['sold'], 
                        df_hist['sold'],
                    )
                )
            ]        
        else:
            rmsse_list = [
                (
                    list(id) if len(id) == 2 else [id[0], 'X']
                ) + [ 
                    RMSSE(
                        df_p['pred'], 
                        df_p['sold'], 
                        df_h['sold'],
                    )
                ]
                for (id, df_p), (id, df_h) in 
                zip(
                    df_pred.groupby(agg_levels_columns[agg_level]), 
                    df_hist.groupby(agg_levels_columns[agg_level])
                )
            ]
        
        # results to dataframe    
        df_rmsse = pd.DataFrame(rmsse_list, columns=['Agg_Level_1', 'Agg_Level_2', 'RMSSE'])
            
        # print temp results
        # logger.info(f'level: {agg_level} - RMSSE list: ' + str(rmsse_list))
                
        # compute weighted average for rmsse
        level_weights = weights[weights['Level_id'] == agg_level]
        
        # align weights with rmsse results
        df_rmsse = pd.merge(
            df_rmsse,
            level_weights,
            on = ['Agg_Level_1', 'Agg_Level_2'],
            how = 'left'
        )
        
        # compute WRMSSE
        level_wrmsse = np.dot(df_rmsse['RMSSE'], df_rmsse['Weight'])
        logger.info(f"{agg_level} - {level_wrmsse}")
        level_wrmsse_list.append(level_wrmsse)

    return np.mean(level_wrmsse_list)

def RMSSE(y_pred: pd.Series, y_true: pd.Series, y_true_hist: pd.Series)->float:
    """ Calculate RMSSE for one unique 'id' """
    
    n = ((y_pred - y_true)**2).mean(skipna=True)
    d = (y_true_hist.diff().dropna()**2).mean()
    
    return np.sqrt(n / d)


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

def data_preprocessing(df, calendar, sell_prices, sales_train_nrows=None)->tuple:
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

def cross_validation_on_validation_set(
    TEMPLATE_FULL_PATH: str = '../submissions/base_cross_validation_template.parquet',
    PREDICTION_BASE_PATH: str = '../data/submissions/',
    PREDICTION_FILE_NAME: str = 'lgb_multivariate_val_non_transposed_temp.csv',
    apply_max_zero: bool = False,
    df_sub_val: pd.DataFrame = []
):
    """Performs all steps for evaluating of the validation set prediction

    Args:
        TEMPLATE_FULL_PATH (str, optional): _description_. Defaults to '../submissions/base_cross_validation_template.parquet'.
        PREDICTION_BASE_PATH (str, optional): _description_. Defaults to '../data/submissions/'.
        PREDICTION_FILE_NAME (str, optional): _description_. Defaults to 'lgb_multivariate_val_non_transposed_temp.csv'.
        apply_max_zero (bool, optional): _description_. Defaults to False.
    """

    logger.info('reading cross validation template')
    df_sub_merged = pd.read_parquet(TEMPLATE_FULL_PATH)
    logger.info('reading prediction file')
    if not isinstance(df_sub_val, pd.DataFrame):
        df_sub_val = pd.read_csv(PREDICTION_BASE_PATH + PREDICTION_FILE_NAME)
    
    # merge submission in evaluation template
    logger.info('merging both files')
    df_sub_merged = pd.merge(
        df_sub_merged,
        df_sub_val,
        on = ['id', 'd'],
        how = 'left',
        suffixes=('_main', '_replacement')
    )
    df_sub_merged['pred'] = df_sub_merged['pred_replacement']
    df_sub_merged = df_sub_merged.drop(['pred_main', 'pred_replacement'], axis = 1)
    
    # compute WRMSSE
    df_sub_merged_t = df_sub_merged
    if apply_max_zero:
        logger.info('applying max(pred,0)')
        df_sub_merged_t['pred'] = df_sub_merged['pred'].apply(lambda x: max(x,0))

    wrmsse = WRMSSE(df_sub_merged_t)
    logger.info(f'wrmsse: {wrmsse}')

class customIter:
    """ tqdm has bugs in jupyter notebooks. This works just as well """
    def __init__(self, x: pd.Grouper):
        self.x: pd.Grouper = x
        
    def __iter__(self):
        i = 0
        tot_iterations = self.x.ngroups
        for id, group in self.x:
            yield id, group
            i += 1
            print(f'{i} / {tot_iterations}', end='\r')
            
def diff_lists(l1: list, l2: list) -> list:
    """ return all elements occurring in l1 but not in l2 """
    return list(set(l1) - set(l2))

def log_status(function):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(function.__name__)
        logger.info('calling')
        return function(*args, **kwargs)
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

def prefixes_in_column(column, prefixes):
    s = 0
    for prefix in prefixes:
        s += prefix_in_column(column, prefix)
    return True if s>0 else False

def prefix_in_column(column, prefix):
    return 1 if prefix in column else 0
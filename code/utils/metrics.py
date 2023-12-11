from utils.configure_logger import configure_logger
configure_logger()
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

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
QUANTILES = (0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
QUANTILE_COLUMN = 'quantile'

def WSPL(df: pd.DataFrame, D_PRED: list = None):
    """
    Args:
        df (pd.DataFrame): Dataframe with prediction and true sales
        D_PRED (list, optional): _description_. Defaults to None.
    """
    # read weights
    logger.info('reading weights file')
    weights: pd.DataFrame = pd.read_csv('../data/m5-forecasting-accuracy/weights_validation.csv')
    weights.columns = ['Level_id', 'agg_column1', 'agg_column2', 'Weight']
    
    # select prediction and historic indices
    if D_PRED == None:
        D_PRED = [f"d_{i}" for i in range(1914, 1914 + 200)]#28)]

    # aggregate specific level and make sure it is sorted well
    logger.info('sorting df on d ...')
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values('d_int')

    # enter loop
    logger.info('entering loop ...')
    level_wrmsse_list: list = []
    level_wrmsse_dict: dict = {}
    for agg_level, df_agg in df.groupby('Level'):
        try:
            # select historic dataframe for denominator calculation and prediction df
            pred_index: pd.Index = df_agg['d'].isin(D_PRED)
            df_hist = df_agg[~pred_index].reset_index(drop=True)
            df_pred = df_agg[pred_index].reset_index(drop=True)

            # rmsse list
            if agg_level == "Level1":
                rmsse_list = [
                    (
                        'Total', 'X',
                        SPL(
                            df_pred[['pred', 'quantile']], 
                            df_pred[['sold', 'quantile']],
                            df_hist[['sold', 'quantile']],
                        )
                    )
                ]        
            else:
                rmsse_list = [
                    (
                        list(id) if len(id) == 2 else [id[0], 'X']
                    ) + [ 
                        SPL(
                            df_p[['pred', 'quantile']], 
                            df_p[['sold', 'quantile']],
                            df_h[['sold', 'quantile']],
                        )
                    ]
                    for (id, df_p), (id, df_h) in 
                    zip(
                        df_pred.groupby(['agg_column1', 'agg_column2']), 
                        df_hist.groupby(['agg_column1', 'agg_column2'])
                    )
                ]
            
            # results to dataframe    
            df_rmsse = pd.DataFrame(rmsse_list, columns=['agg_column1', 'agg_column2', 'MSPL'])
    
            # compute weighted average for rmsse
            level_weights = weights[weights['Level_id'] == agg_level]
            
            # align weights with rmsse results
            df_rmsse = pd.merge(
                df_rmsse,
                level_weights,
                on = ['agg_column1', 'agg_column2'],
                how = 'left'
            )
            
            # compute WRMSSE
            level_wrmsse = np.dot(df_rmsse['MSPL'], df_rmsse['Weight'])
            logger.info(f"{agg_level} - {level_wrmsse}")
            level_wrmsse_list.append(level_wrmsse)
            level_wrmsse_dict[agg_level] = level_wrmsse
        
        except Exception as e:
            logger.error(e)

    return np.mean(level_wrmsse_list), level_wrmsse_dict

def SPL(df_pred, df_true, df_true_hist: pd.DataFrame):
    # scale to divide pinball loss with
    scale = df_true_hist['sold'].diff().abs().mean(skipna=True) + 1e-3
    # logger.info('scale: ' + str(scale))
    # for each quantile, compute
    all_quantiles = [
        mean_pinball_loss(
            df_true[df_true['quantile'] == 0.005]['sold'], 
            df_pred[df_pred['quantile'] == q]['pred'], 
            alpha = q
        )
        for q in QUANTILES
    ]
    return np.mean(all_quantiles) / scale
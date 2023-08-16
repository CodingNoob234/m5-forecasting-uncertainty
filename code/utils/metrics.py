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
QUANTILES = (0.01, 0.025, 0.175, 0.25, 0.5, 0.75, 0.835, 0.975, 0.99)
QUANTILE_COLUMNS = [
    'pred_{q}' for q in QUANTILES
]

def WSPL(df_dict: dict, D_PRED: list = None):
    """
    Args:
        df_dict (dict): dict with dataframes at each level. Keys are Level{i}
        D_PRED (list, optional): _description_. Defaults to None.
    """
    
    logger.info('reading weights file')
    weights: pd.DataFrame = pd.read_csv('../data/weights_validation.csv')
    
    if D_PRED == None:
        D_PRED = [f"d_{i}" for i in range(1914, 1914 + 28)]

    level_wrmsse_list: list = []
    for agg_level in AGG_LEVEL_COLUMNS.keys():
        
        # aggregate specific level and make sure it is sorted well
        df_agg: pd.DataFrame = df_dict[agg_level]
        df_agg['d_int'] = df_agg['d'].apply(lambda x: int(x.split('_')[1]))
        df_agg = df_agg.sort_values('d_int')
        
        # select historic dataframe for denominator calculation and prediction df
        pred_index: pd.Index = df_agg['d'].isin(D_PRED)
        df_hist = df_agg.drop(pred_index, errors = "ignore")
        df_pred = df_agg[pred_index]
        
        # rmsse list
        if agg_level == "Level1":
            rmsse_list = [
                (
                    'Total', 'X', 
                    SPL(
                        df_pred[QUANTILE_COLUMNS], 
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
                    SPL(
                        df_p[QUANTILE_COLUMNS], 
                        df_p['sold'], 
                        df_h['sold'],
                    )
                ]
                for (id, df_p), (id, df_h) in 
                zip(
                    df_pred.groupby(AGG_LEVEL_COLUMNS[agg_level]), 
                    df_hist.groupby(AGG_LEVEL_COLUMNS[agg_level])
                )
            ]
        
        # results to dataframe    
        df_rmsse = pd.DataFrame(rmsse_list, columns=['Agg_Level_1', 'Agg_Level_2', 'MSPL'])
            
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
        level_wrmsse = np.dot(df_rmsse['MSPL'], df_rmsse['Weight'])
        logger.info(f"{agg_level} - {level_wrmsse}")
        level_wrmsse_list.append(level_wrmsse)

    return np.mean(level_wrmsse_list)
    

def SPL(y_pred, y_true, y_true_hist: pd.Series):
    # scale to divide pinball loss with
    scale = y_true_hist.diff().mean(skipna=True)

    # for each quantile, compute 
    all_quantiles = [
        mean_pinball_loss(y_true, y_pred[c], q)
        for q, c in zip(QUANTILES, QUANTILE_COLUMNS)
    ]
    return np.mean(all_quantiles) / scale
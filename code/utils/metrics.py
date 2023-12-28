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

def WSPL(df: pd.DataFrame, D_PRED: list = [f"d_{i}" for i in range(1914, 1914 + 200)]):
    """
    Args:
        df (pd.DataFrame): Dataframe with prediction and true sales
        D_PRED (list, optional): _description_. Defaults to None.
    """
    # read weights
    logger.info('reading weights file')
    weights: pd.DataFrame = pd.read_csv('../data/m5-forecasting-accuracy/weights_validation.csv')
    weights.columns = ['Level_id', 'agg_column1', 'agg_column2', 'Weight']

    # aggregate specific level and make sure it is sorted well
    logger.info('sorting df on d ...')
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values('d_int')

    # enter loop
    logger.info('entering loop ...')
    # level_wrmsse_list: list = []
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
            
            # align weights with SPL results
            df_rmsse = pd.merge(
                df_rmsse,
                level_weights,
                on = ['agg_column1', 'agg_column2'],
                how = 'left'
            )
            
            # compute WSPL
            level_wrmsse = np.dot(df_rmsse['MSPL'], df_rmsse['Weight'])
            
            # store results
            logger.info(f"{agg_level} - {level_wrmsse}")
            level_wrmsse_list.append(level_wrmsse)
            level_wrmsse_dict[agg_level] = level_wrmsse
        
        except Exception as e:
            logger.error(e)

    return np.mean(list(level_wrmsse_dict.keys())), level_wrmsse_dict
    # return np.mean(level_wrmsse_list), level_wrmsse_dict

def SPL(df_pred, df_true, df_true_hist: pd.DataFrame):
    """ Computes the scales pinball loss for all quantiles """
    # compute factor to scale by
    scale = df_true_hist['sold'].diff().abs().mean(skipna=True) + 1e-3
    
    # for each quantile, compute the Pinball-loss
    idx = df_true['quantile'] == 0.005
    all_quantiles = [
        mean_pinball_loss(
            df_true[idx]['sold'], 
            df_pred[df_pred['quantile'] == q]['pred'], 
            alpha = q
        )
        for q in QUANTILES
    ]
    return np.mean(all_quantiles) / scale

########################### DIEBOLD MARIANO ###########################
def DM_test_pinball(df, h, p_crit: float = 0.05):
    quantile = df['quantile']
    #
    resid_x = df['sold'] - df['pred_x']
    idx = resid_x >= 0
    pinball_resid_x = resid_x
    pinball_resid_x[idx] = resid_x[idx] * (1 - quantile[idx])
    pinball_resid_x[~idx] = resid_x[~idx] * quantile[~idx]
    #
    resid_y = df['sold'] - df['pred_y']
    idx = resid_y >= 0
    pinball_resid_y = resid_y
    pinball_resid_y[idx] = resid_y[idx] * (1 - quantile[idx])
    pinball_resid_y[~idx] = resid_y[~idx] * quantile[~idx]
    #
    df['pinball_resid'] = pinball_resid_x - pinball_resid_y
    #
    agg_dict = {
        'revenue': 'last',
        'pinball_resid': 'mean',
        'Level': 'last'
    }
    df_qtile_avg = df.groupby(['d', 'id_merge']).agg(agg_dict).reset_index(drop=False)
    # df_qtile_avg = df.groupby(['d', 'id_merge'])['pinball_resid']\
    #     .mean().reset_index(drop=False)

    ids = []
    stats = []
    p_values = []
    levels = []
    h0_rejected = []
    for id_merge, df_s in df_qtile_avg.groupby('id_merge'):
        
        # compute cov
        p_s = df_s['pinball_resid']
        mean = p_s.mean()
        T = len(p_s)
        
        def auto_cov(resid, lag, mean):
            resid = list(resid)
            cov = 0
            T = float(len(resid))
            for i in np.arange(0, len(resid)-lag):
                cov += ((resid[i+lag])-mean)*(resid[i]-mean)
            return (1/(T))*cov
        
        gamma = []
        for lag in range(h):
            gamma.append(auto_cov(p_s, lag, mean))
        
        # compute stat
        V_d = (gamma[0] + 2*sum(gamma[1:]))/T
        DM_stat=V_d**(-0.5)*mean
        harvey_adj=( ( T+1-2*h+h*(h-1)) / T ) ** 0.5
        DM_stat = harvey_adj*DM_stat

        # compute p_value
        from scipy.stats import t
        p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
        
        # store results
        levels.append(df_s['Level'].iloc[0])
        ids.append(id_merge)
        stats.append(DM_stat)
        p_values.append(round(p_value,5))
        h0_rejected.append(True if p_value < p_crit else (True if pd.isna(p_value) else False))
        
    return pd.DataFrame({
        'level': levels,
        'ids': ids,
        'stats': stats,
        'p_values': p_values,
        'h0_rejected': h0_rejected
    })
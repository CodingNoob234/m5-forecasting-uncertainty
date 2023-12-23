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
def DM_test_pinball(pred1: pd.DataFrame, pred2: pd.DataFrame, true: pd.DataFrame, h):
    df = pd.merge(
        pred1,
        pred2,
        how = 'inner',
        on = ['quantile', 'd', 'agg_column1', 'agg_column2']
    )
    df = pd.merge(
        df,
        true,
        on = ['d', 'agg_column1', 'agg_column2'],
        how = 'left'
    )
    
    resid = df['sold'] - df['pred']
    quantile = df['quantile']
    idx = resid >= 0
    pinball_resid = resid
    pinball_resid[idx] = resid[idx] * (1 - quantile[idx])
    pinball_resid[~idx] = resid[~idx] * quantile[idx]
    df['pinball_resid'] = pinball_resid
    
    df_qtile_avg = df.groupby(['d', 'agg_column1', 'agg_column2'])['pinball_resid']\
        .mean().reset_index(drop=False)

    ids = []
    stats = []
    p_values = []
    for (agg_column1, agg_column2), df_s in df_qtile_avg.groupby(['agg_column1', 'agg_column2']):
        
        # compute cov
        p_s = df_s['pinball_resid']
        mean = p_s.mean()
        T = len(p_s)
        
        def auto_cov(resid, lag, mean):
            cov = 0
            T = float(len(resid))
            for i in np.arange(0, len(resid)-lag):
                autoCov += ((resid[i+lag])-mean)*(resid[i]-mean)
            return (1/(T))*autoCov
        
        gamma = []
        for lag in range(h):
            gamma.append(auto_cov(p_s, lag, mean))
        
        # compute stat
        V_d = (gamma[0] + 2*sum(gamma[1:]))/T
        DM_stat=V_d**(-0.5)*mean
        harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
        DM_stat = harvey_adj*DM_stat
        
        # compute p_value
        from scipy.stats import t
        p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
        
        # store results
        ids.append(agg_column1 + '_' + agg_column2)
        stats.append(DM_stat)
        p_values.append(p_value)
        
    return pd.DataFrame({
        'ids': ids,
        'stats': stats,
        'p_values': p_values
    })
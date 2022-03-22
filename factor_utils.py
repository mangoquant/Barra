from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd
import numpy as np


def winsorize(df, n):
    """
    n倍MAD去极值

    :param df: pd.DataFrame, index = date, columns = assets
    :param n: int
    :return: pd.DataFrame
    """
    MAD = (df.sub(df.median(axis=1), axis=0).apply(abs)).median(axis=1)
    up = df.median(axis=1) + n * 1.4826 * MAD
    down = df.median(axis=1) - n * 1.4826 * MAD
    return df.clip(lower=down, upper=up, axis=0)


def z_score(df):
    """
    z_score标准化方法

    :param df: pd.DataFrame, index = date, columns = assets
    :return: pd.DataFrame
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def get_factor_ret(factor, ret, market_value, quantile=5):
    """
    获取因子收益率，用多空组合收益率之差表示

    :param factor: pd.DataFrame, index = date, columns = assets
    :param pricing: pd.DataFrame, 交易价格数据
    :param market_value: pd.DataFrame, 股票市值
    :param quantile: int, default = 5, 分组数，取第一组和最后一组市值加权收益率差值作为

    :return: pd.Series, 因子收益率序列
    """

    indexs = factor.index
    ret = ret.loc[indexs, :]
    market_value = market_value.loc[indexs, :]
    
    f_rank = factor.rank(method='first', axis=1)
    q = f_rank.quantile(q=(1./quantile, 1.-1./quantile), axis=1,
                        numeric_only=True, interpolation='linear').T
    up = pd.DataFrame(f_rank.values > q[1.-1./quantile].values.reshape(
        (q.shape[0], 1)), index=f_rank.index, columns=f_rank.columns)
    down = pd.DataFrame(f_rank.values < q[1./quantile].values.reshape(
        (q.shape[0], 1)), index=f_rank.index, columns=f_rank.columns)

    top = (ret[up]*market_value[up]).sum(axis=1)/market_value[up].sum(axis=1)
    bottom = (ret[down]*market_value[down]).sum(axis=1)/market_value[down].sum(axis=1)
    LS = top - bottom
    return LS


def stand_df(df, ticker_list, trade_day):
    """
    标准化df

    Parameters
    ----------
    df : pd.DataFrame
        要标准化的df
    ticker_list : array like
        列名，资产列表
    trade_day : array like
        索引，交易日序列

    Returns
    -------
    refined_df : pd.DataFrame
        标准化后的df

    """
    stand_df = pd.DataFrame(index=trade_day, columns=ticker_list)
    df = df.reindex_like(stand_df[df.index[0]:df.index[-1]])
    df = df.ffill().replace([np.inf, -np.inf], 0)
    return df




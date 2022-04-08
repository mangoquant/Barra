# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:28:08 2022

@author: PCXU
@contact: xupengchengmail@163.com
"""

import os

import pandas as pd
import numpy as np

import configs as cfg
from tools import *


if __name__ == '__main__':
    # 加载数据
    industry_df = pd.read_pickle(cfg.INDUSTRY_PATH)
    ticker_list = list(industry_df.index)

    daily_data = PriceDataBase(pd.read_pickle(os.path.join(
        cfg.DATA_ROOT_PATH, 'daily_info.pkl')))
    trade_day = pd.read_pickle(cfg.DATA_ROOT_PATH+"trade_day.pkl").index
    factor_level1 = PriceDataBase(pd.read_pickle(
        os.path.join(cfg.FACTOR_PATH, 'factor_level1.pkl')))
    # 收益率估计
    close = daily_data.get_table('close')
    ret_M = close.pct_change().replace(
        [np.inf, -np.inf], 0).rolling(21).mean().dropna(axis=0, how='all')

    date = '20091001'
    BO = BarraOptimizer()
    ret = ret_M.loc[date, :]
    assets_pool = ret[~ret.isna()].index
    # 中性化的风格因子
    beta_style = pd.DataFrame(index=ticker_list)
    beta_style['Size'] = factor_level1.get_table('Size').loc[date, :].fillna(0)
    beta_style['Beta'] = factor_level1.get_table('Beta').loc[date, :].fillna(0)
    beta_style['Book-to-Price'] = factor_level1.get_table(
        'Book-to-Price').loc[date, :].fillna(0)
    
    weight = BO.get_weight(ret=ret,
                           assets_pool=assets_pool,
                           beta_industry=industry_df.loc[:,
                                                         industry_df.columns[1:]],
                           beta_style=factor_level1.get_table('Size').loc[date, :])

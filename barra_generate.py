# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:33:10 2022

@author: PCXU
"""


import os

from tqdm import tqdm
import pandas as pd
import numpy as np

from tools.barra_calculator import BarraCalculator
from tools import *
import configs as cfg


if __name__ == '__main__':
    # # 行业分类信息
    # industry_df = pd.read_csv(os.path.join(cfg.DATA_ROOT_PATH, 'industry.csv'),
    #                           index_col=[0])
    # industry_df.index = industry_df.index.astype(str)
    # industry_df = industry_df.sort_index()
    # industry_df.index.name = 'asset'
    # industry_df = pd.merge(industry_df, pd.get_dummies(
    #     industry_df['industry']), left_index=True, right_index=True)
    # industry_df.to_pickle(cfg.INDUSTRY_PATH)

    # 资产列表
    industry_df = pd.read_pickle(cfg.INDUSTRY_PATH)
    ticker_list = list(industry_df.index)

    # # 日频价量数据
    # FR = load_FR_pkl(cfg.FR_PATH)
    # # 交易日
    # trade_day = pd.DataFrame(
    #     index=FR.index.get_level_values(0).unique().sort_values())
    # trade_day.to_pickle(cfg.DATA_ROOT_PATH+"trade_day.pkl")
    # trade_day = trade_day.index

    # daily_data = pd.DataFrame()
    # fields = ['open', 'high', 'low', 'close', 'volume', 'pb', 'total_share']
    # for f in fields:
    #     df = FR[f].unstack()
    #     df.columns = df.columns.astype('str')
    #     df = stand_df(df, ticker_list, trade_day)
    #     daily_data[f] = df.stack()

    # close = daily_data['close'].unstack()
    # total_share = daily_data['total_share'].unstack()
    # market_value = close*total_share
    # daily_data['market_value'] = market_value.stack()
    # daily_data.index.names = ['datetime', 'asset']

    # daily_data.to_pickle(os.path.join(
    #     cfg.DATA_ROOT_PATH, 'daily_info.pkl'))

    daily_data = PriceDataBase(pd.read_pickle(os.path.join(
        cfg.DATA_ROOT_PATH, 'daily_info.pkl')))
    trade_day = pd.read_pickle(cfg.DATA_ROOT_PATH+"trade_day.pkl").index

    # # 财报基本面数据
    # net_income, oper_revenue = load_quarter_data(cfg.QUARTER_DATA, pd.date_range(
    #     start='20040331', end='20210930', freq='3M'), ticker_list)
    # net_income_TTM = get_TTM(net_income)
    # oper_revenue_TTM = get_TTM(oper_revenue)
    # quarter_data = pd.concat(
    #     [oper_revenue_TTM.stack(), net_income_TTM.stack()], axis=1, sort=True)
    # quarter_data.columns = ['oper_revenue_TTM', 'net_income_TTM']
    # quarter_data.index.names = ['datetime', 'asset']
    # quarter_data.to_pickle(os.path.join(
    #     cfg.DATA_ROOT_PATH, 'quarter_info.pkl'))

    quarter_data = PriceDataBase(pd.read_pickle(os.path.join(
        cfg.DATA_ROOT_PATH, 'quarter_info.pkl')))

    market_value = daily_data.get_table('market_value')
    close = daily_data.get_table('close')
    # pb = daily_data.get_table('pb')
    # volume = daily_data.get_table('volume')
    # total_share = daily_data.get_table('total_share')

    ret = close.pct_change().replace([np.inf, -np.inf], 0)
    # net_income = quarter_data.get_table('net_income_TTM')
    # oper_revenue = quarter_data.get_table('oper_revenue_TTM')

    # # 计算二级Barra因子
    # BC = BarraCalculator(rf=cfg.Rf_daily, datetime=trade_day, start='20090401')
    # size = BC.get_size(market_value)
    # beta, hsigma = BC.get_beta_hsigma(ret, market_value, 252)
    # rstr = BC.get_rstr(ret, 504, 126, 21)
    # dastd = BC.get_dastd(ret, 252, 42)
    # cmra = BC.get_cmra(ret, 12)
    # nlsize = BC.get_nlsize(size**3, size)
    # btop = BC.get_btop(pb)
    # stom = BC.get_turnover(volume, total_share, 21)
    # stoq = BC.get_turnover(volume, total_share, 63)
    # stoa = BC.get_turnover(volume, total_share, 252)
    # egro = BC.get_growth_factor(net_income, ticker_list)
    # sgro = BC.get_growth_factor(oper_revenue, ticker_list)

    # factor_level2 = pd.DataFrame()
    # factor_level2['size'] = size.stack()
    # factor_level2['beta'] = beta.stack()
    # factor_level2['rstr'] = rstr.stack()
    # factor_level2['dastd'] = dastd.stack()
    # factor_level2['cmra'] = cmra.stack()
    # factor_level2['hsigma'] = hsigma.stack()
    # factor_level2['nlsize'] = nlsize.stack()
    # factor_level2['btop'] = btop.stack()
    # factor_level2['stom'] = stom.stack()
    # factor_level2['stoq'] = stoq.stack()
    # factor_level2['stoa'] = stoa.stack()
    # factor_level2['egro'] = egro.stack()
    # factor_level2['sgro'] = sgro.stack()
    # factor_level2.to_pickle(os.path.join(cfg.FACTOR_PATH, 'factor_level2.pkl'))

    # factor_level2 = PriceDataBase(pd.read_pickle(
    #     os.path.join(cfg.FACTOR_PATH, 'factor_level2.pkl')))

    # size = z_score(winsorize(factor_level2.get_table('size'), 3))
    # beta = z_score(winsorize(factor_level2.get_table('beta'), 3))
    # rstr = z_score(winsorize(factor_level2.get_table('rstr'), 3))
    # dastd = z_score(winsorize(factor_level2.get_table('dastd'), 3))
    # cmra = z_score(winsorize(factor_level2.get_table('cmra'), 3))
    # hsigma = z_score(winsorize(factor_level2.get_table('hsigma'), 3))
    # nlsize = z_score(winsorize(factor_level2.get_table('nlsize'), 3))
    # btop = z_score(winsorize(factor_level2.get_table('btop'), 3))
    # stom = z_score(winsorize(factor_level2.get_table('stom'), 3))
    # stoq = z_score(winsorize(factor_level2.get_table('stoq'), 3))
    # stoa = z_score(winsorize(factor_level2.get_table('stoa'), 3))
    # egro = z_score(winsorize(factor_level2.get_table('egro'), 3))
    # sgro = z_score(winsorize(factor_level2.get_table('sgro'), 3))

    # # 利用二级Barra因子合成一级因子
    # factor_level1 = pd.DataFrame()
    # factor_level1['Size'] = size.stack()
    # factor_level1['Beta'] = beta.stack()
    # factor_level1['Momentum'] = rstr.stack()
    # factor_level1['Residual_Volatility'] = z_score(
    #     0.74*dastd+0.16*cmra+0.1*hsigma).stack()
    # factor_level1['Non-linear_Size'] = nlsize.stack()
    # factor_level1['Book-to-Price'] = btop.stack()
    # factor_level1['Liquidty'] = z_score(0.35*stom+0.35*stoq+0.3*stoa).stack()
    # factor_level1['Growth'] = z_score(
    #     (0.24/(0.24+0.47))*egro+(0.47/(0.24+0.47))*sgro).stack()
    # factor_level1 = factor_level1.astype('float64')

    # factor_level1.to_pickle(os.path.join(cfg.FACTOR_PATH, 'factor_level1.pkl'))

    factor_level1 = PriceDataBase(pd.read_pickle(
        os.path.join(cfg.FACTOR_PATH, 'factor_level1.pkl')))

    # # 分5组因子多空收益率
    # # 注：此处仅做空第一组，做多第五组，对因子的方向未做判断
    # factor_lst = ['Size', 'Beta', 'Momentum', 'Residual_Volatility',
    #               'Non-linear_Size', 'Book-to-Price', 'Liquidty', 'Growth']
    # Barra_ret_5q = pd.DataFrame()
    # for f in factor_lst:
    #     factor = factor_level1.get_table(f)
    #     Barra_ret_5q[f] = get_factor_ret(factor, ret, market_value)

    # Barra_ret_5q.to_pickle(os.path.join(cfg.FACTOR_RETURN, 'Barra_ret_5q.pkl'))
    Barra_ret_5q = pd.read_pickle(os.path.join(
        cfg.FACTOR_RETURN, 'Barra_ret_5q.pkl'))

    # # 纯因子收益率
    # datetime = factor_level1.data.index.get_level_values(
    #     'datetime').unique().sort_values()
    # factor_name = ['country', '光電業', '其他業', '其他電子業', '化學工業', 
    #                '半導體業', '塑膠工業', '建材營造業', '橡膠工業', '水泥工業', 
    #                '汽車工業', '油電燃氣業', '玻璃陶瓷', '生技醫療業', '紡織纖維', 
    #                '航運業', '觀光事業', '貿易百貨業', '資訊服務業', '通信網路業', 
    #                '造紙工業', '金融保險業', '鋼鐵工業', '電器電纜', '電子通路業', 
    #                '電子零組件業', '電機機械', '電腦及週邊設備業', '食品工業', 
    #                'Size', 'Beta', 'Momentum', 'Residual_Volatility', 
    #                'Non-linear_Size', 'Book-to-Price', 'Liquidty', 'Growth']
    # Barra_ret_net = pd.DataFrame(index=datetime, columns=factor_name)
    # for i in tqdm(datetime):
    #     # Barra因子矩阵
    #     style = factor_level1.data.loc[i, :, :]
    #     style.index = style.index.droplevel(0)
    #     industry = industry_df.loc[style.index, industry_df.columns[1:]]
    #     country = pd.DataFrame(
    #         np.ones((style.shape[0], 1)), index=style.index, columns=['country'])
    #     X = pd.concat([country, industry, style], axis=1).values
    #     X[~np.isfinite(X)] = 0.
    #     # 权重矩阵
    #     W = np.sqrt(market_value.loc[i, style.index].values)
    #     W = np.diag(W/W.sum())
    #     # 约束矩阵
    #     industry_value = industry.mul(
    #         market_value.loc[i, style.index], axis=0).sum(axis=0)
    #     adj_industry_weights = -1 * \
    #         (industry_value[:-1]/industry_value[-1]).values
    #     C = np.eye(1+industry.shape[1]+style.shape[1])
    #     C = np.delete(C, industry.shape[1], axis=1)
    #     C[industry.shape[1], 1:industry.shape[1]] = adj_industry_weights
    #     # 纯因子投资组合权重矩阵
    #     a = np.matmul(X, C)
    #     b = np.matmul(np.matmul(a.T, W), a)
    #     invb = np.linalg.inv(b)
    #     omega = np.matmul(np.matmul(np.matmul(C, invb), a.T), W)
    #     # 纯因子收益率
    #     Barra_ret_net.loc[i, :] = np.matmul(
    #         omega, ret.loc[i, style.index].fillna(0).values)

    # Barra_ret_net.to_pickle(os.path.join(
    #     cfg.FACTOR_RETURN, 'Barra_ret_net.pkl'))
    Barra_ret_net = pd.read_pickle(os.path.join(
        cfg.FACTOR_RETURN, 'Barra_ret_net.pkl'))

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:33:10 2022

@author: PCXU
"""


import os

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm
import pandas as pd
import numpy as np

import configs as cfg
from tools import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_pos_expose(pos_data: pd.DataFrame,
                   asset_expose: pd.DataFrame,
                   industry_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    获取持仓在Barra风格因子上的暴露

    Parameters
    ----------
    pos_data : pd.DataFrame, index=(datetime, asset)
        持仓信息，至少包含weight一列
    asset_expose : pd.DataFrame, index=(datetime, asset), columns=style_factor
        个股在Barra风格因子上的暴露信息
    industry_df: pd.DataFrame, optional, index=asset, columns=industry_name
        行业分类信息，若不为None，则返回在国家、行业和风格因子上的暴露，
        否则，仅返回风格因子暴露
        The default is None.

    Returns
    -------
    pd.DataFrame

    """
    dt = asset_expose.index.get_level_values(0).unique()
    pos_data = pos_data[dt[0]:dt[-1]]
    datetime = pos_data.index.get_level_values(0).unique()
    if type(industry_df) == pd.DataFrame:
        col_names = ['country', ] + \
            list(industry_df.columns)+list(asset_expose.columns)
    else:
        col_names = asset_expose.columns
    pos_expose = pd.DataFrame(index=datetime, columns=col_names)

    for i in tqdm(datetime):
        hold = pos_data.loc[i, 'weight']
        assets = hold.index
        hold = hold.values.reshape(1, -1)

        if type(industry_df) == pd.DataFrame:
            country = np.ones((len(assets), 1))
            industry = industry_df.loc[assets, :]
            style = asset_expose.loc[(i, assets), :].fillna(0).values
            expose = np.hstack((country, industry, style))
        else:
            expose = asset_expose.loc[(i, assets), :].fillna(0).values

        pos_expose.loc[i, :] = np.matmul(hold, expose)
    return pos_expose


def get_his_q(df: pd.DataFrame,
              min_window: int = 252) -> pd.DataFrame:
    """
    返回df的历史分位数

    Parameters
    ----------
    df : pd.DataFrame, index=datetime
        要计算的df
    min_window : int, optional
        计算历史分位数的最小窗口 
        The default is 252.

    Returns
    -------
    pd.DataFrame

    """
    datetime = df.index
    factors = df.columns
    df_q = pd.DataFrame(index=datetime, columns=factors)
    for i in range(df.shape[0]):
        if i >= min_window:
            for f in factors:
                his = df.loc[:datetime[i-1], f]
                now = df.loc[datetime[i], f]
                df_q.loc[datetime[i], f] = (his < now).sum()/i
    return df_q


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

    # # 分20组因子多空收益率
    # # 注：此处仅做空第1组，做多第20组，对因子的方向未做判断
    # factor_lst = ['Size', 'Beta', 'Momentum', 'Residual_Volatility',
    #               'Non-linear_Size', 'Book-to-Price', 'Liquidty', 'Growth']
    # Barra_ret_20q = pd.DataFrame()
    # for f in factor_lst:
    #     factor = factor_level1.get_table(f)
    #     Barra_ret_20q[f] = get_factor_ret(factor, ret, market_value, quantile=20)

    # Barra_ret_20q.to_pickle(os.path.join(cfg.FACTOR_RETURN, 'Barra_ret_20q.pkl'))
    Barra_ret_20q = pd.read_pickle(os.path.join(
        cfg.FACTOR_RETURN, 'Barra_ret_20q.pkl'))

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

    # # 20组多空分析
    # # 持仓信息
    # pos_data = pd.read_excel(os.path.join(
    #     cfg.POSITION_PATH, 'positi...roup20.xlsx'), parse_dates=[0])
    # pos_data['asset'] = pos_data['asset'].astype('str')
    # pos_data.index = pd.MultiIndex.from_frame(pos_data[['datetime', 'asset']])
    # # 持仓暴露
    # pos_expose = get_pos_expose(pos_data, factor_level1.data)
    # pos_ret = pos_expose*Barra_ret_20q.loc[pos_expose.index]

    # hold_analysis = pd.DataFrame()
    # hold_analysis['expose'] = pos_expose.stack()
    # hold_analysis['return'] = pos_ret.stack()
    # hold_analysis.to_pickle(os.path.join(
    #     cfg.POSITION_PATH, 'hold_analysis_20q.pkl'))

    # # 持仓分析
    # hold_analysis = PriceDataBase(pd.read_pickle(
    #     os.path.join(cfg.POSITION_PATH, 'hold_analysis_20q.pkl')))
    # factor_expose = hold_analysis.get_table(
    #     'expose').astype('float64').round(4)
    # factor_ret = hold_analysis.get_table('return').astype('float64').round(4)
    # factor_lst = factor_expose.columns
    # x_label = [t.strftime('%Y-%m-%d') for t in factor_expose.index]
    # factor_expose_q = get_his_q(factor_expose)
    # factor_ret_q = get_his_q(factor_ret)
    # with PdfPages(os.path.join(cfg.POSITION_PATH, 'hold_analysis_20q.pdf')) as pdf:
    #     for f in factor_lst:
    #         fig = plt.figure(figsize=(18, 6))

    #         ax1 = fig.add_subplot(221)
    #         ax1.bar(x_label, factor_expose[f])
    #         ax1.xaxis.set_visible(False)
    #         plt.xlim(-5, len(x_label)+5)
    #         plt.grid(axis="y")
    #         ax1.set_title('portfolio exposure on "{}" factor'.format(f))

    #         ax3 = fig.add_subplot(223)
    #         ax3.scatter(x_label, factor_expose_q[f], s=1, c='r')
    #         ax3.xaxis.set_major_locator(
    #             ticker.MultipleLocator(int(len(x_label)/5)))
    #         plt.xticks(rotation=-45)
    #         plt.xlim(-5, len(x_label)+5)
    #         plt.grid(axis="y")
    #         ax3.set_title(
    #             'historical quantile of "{}" factor exposures'.format(f))

    #         ax2 = fig.add_subplot(222)
    #         ax2.plot(x_label, (1+factor_ret[f]).cumprod())
    #         ax2.xaxis.set_visible(False)
    #         plt.xlim(-5, len(x_label)+5)
    #         plt.grid(axis="y")
    #         ax2.set_title(
    #             'portfolio cumulative return on "{}" factor'.format(f))

    #         ax4 = fig.add_subplot(224)
    #         ax4.scatter(x_label, factor_ret_q[f], s=1, c='r')
    #         ax4.xaxis.set_major_locator(
    #             ticker.MultipleLocator(int(len(x_label)/5)))
    #         plt.xticks(rotation=-45)
    #         plt.xlim(-5, len(x_label)+5)
    #         plt.grid(axis="y")
    #         ax4.set_title(
    #             'historical quantile of "{}" factor return'.format(f))

    #         pdf.savefig(fig)

    # # 纯因子收益率分析分析
    # # 持仓信息
    # pos_data = pd.read_excel(os.path.join(
    #     cfg.POSITION_PATH, 'positi...roup20.xlsx'), parse_dates=[0])
    # pos_data['asset'] = pos_data['asset'].astype('str')
    # pos_data.index = pd.MultiIndex.from_frame(pos_data[['datetime', 'asset']])
    # # 持仓暴露
    # pos_expose = get_pos_expose(
    #     pos_data, factor_level1.data, industry_df.drop('industry', axis=1))
    # pos_ret = pos_expose*Barra_ret_net.loc[pos_expose.index]

    # hold_analysis = pd.DataFrame()
    # hold_analysis['expose'] = pos_expose.stack()
    # hold_analysis['return'] = pos_ret.stack()
    # hold_analysis.to_pickle(os.path.join(
    #     cfg.POSITION_PATH, 'hold_analysis_net.pkl'))

    # 持仓分析
    hold_analysis = PriceDataBase(pd.read_pickle(
        os.path.join(cfg.POSITION_PATH, 'hold_analysis_net.pkl')))
    factor_expose = hold_analysis.get_table(
        'expose').astype('float64').round(4)
    factor_ret = hold_analysis.get_table('return').astype('float64').round(4)
    factor_lst = factor_expose.columns
    x_label = [t.strftime('%Y-%m-%d') for t in factor_expose.index]
    factor_expose_q = get_his_q(factor_expose)
    factor_ret_q = get_his_q(factor_ret)
    with PdfPages(os.path.join(cfg.POSITION_PATH, 'hold_analysis_net.pdf')) as pdf:
        for f in factor_lst:
            fig = plt.figure(figsize=(18, 6))

            ax1 = fig.add_subplot(221)
            ax1.bar(x_label, factor_expose[f])
            ax1.xaxis.set_visible(False)
            plt.xlim(-5, len(x_label)+5)
            plt.grid(axis="y")
            ax1.set_title('portfolio exposure on "{}" factor'.format(f))

            ax3 = fig.add_subplot(223)
            ax3.scatter(x_label, factor_expose_q[f], s=1, c='r')
            ax3.xaxis.set_major_locator(
                ticker.MultipleLocator(int(len(x_label)/5)))
            plt.xticks(rotation=-45)
            plt.xlim(-5, len(x_label)+5)
            plt.grid(axis="y")
            ax3.set_title(
                'historical quantile of "{}" factor exposures'.format(f))

            ax2 = fig.add_subplot(222)
            ax2.plot(x_label, (1+factor_ret[f]).cumprod())
            ax2.xaxis.set_visible(False)
            plt.xlim(-5, len(x_label)+5)
            plt.grid(axis="y")
            ax2.set_title(
                'portfolio cumulative return on "{}" factor'.format(f))

            ax4 = fig.add_subplot(224)
            ax4.scatter(x_label, factor_ret_q[f], s=1, c='r')
            ax4.xaxis.set_major_locator(
                ticker.MultipleLocator(int(len(x_label)/5)))
            plt.xticks(rotation=-45)
            plt.xlim(-5, len(x_label)+5)
            plt.grid(axis="y")
            ax4.set_title(
                'historical quantile of "{}" factor return'.format(f))

            pdf.savefig(fig)

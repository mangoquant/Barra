# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:18:45 2022

@author: PCXU
"""

import statsmodels.api as sm
from typing import List
from tqdm import tqdm
import pandas as pd
import numpy as np


class BarraCalculator:
    """
    这个类用来定义计算Barra因子的相关函数
    """

    def __init__(self,
                 rf: float,
                 datetime: pd.DatetimeIndex,
                 start: str = None,
                 end: str = None,
                 freq: int = 1):
        """


        Parameters
        ----------
        rf : float
            日频无风险利率

        datetime : pd.DatetimeIndex
            时间格式的交易日索引

        start : str, optional
            格式："YYYYMMdd"。The default is None.
            生成因子值的开始时间，默认为尽可能充分利用datetime

        end : str, optional
            格式："YYYYMMdd"。The default is None.
            生成因子值的结束时间，默认为尽可能充分利用datetime

        freq : int, optional
            因子计算频率，间隔freq个交易日计算一次因子值
            The default is 1.

        """
        self.rf = rf
        self.freq = freq
        self.datetime = datetime
        self.start = start if start else self.datetime[0]
        self.end = end if end else self.datetime[-1]
        self.datetime = self.datetime[(
            self.datetime >= self.start) & (self.datetime <= self.end)]
        self.datetime = self.datetime[::self.freq]

    def get_size(self,
                 market_value: pd.DataFrame) -> pd.DataFrame:
        """
        计算市值因子

        Parameters
        ----------
        market_value : pd.DataFrame
            index=datetime, columns=asset
            市值数据

        Returns
        -------
        size : pd.DataFrame

        """
        market_value = market_value.loc[self.datetime, :]

        size = market_value.apply(np.log)
        return size.replace([np.inf, -np.inf], 0)

    def get_beta_hsigma(self,
                        ret: pd.DataFrame,
                        market_value: pd.DataFrame,
                        T: int) -> tuple:
        """
        计算beta和hsigma(beta回归残差年化波动率)因子

        Parameters
        ----------
        ret : pd.DataFrame
            index=datetime, columns=asset
            资产原始日收益率
            ret的开始日期应至少比self.start早T-1期
        market_value : pd.DataFrame
            index=datetime, columns=asset
            市值数据
            market_value的开始日期应至少比self.start早T-1期
        T : int
            回归区间长度

        Returns
        -------
        tuple
            beta : pd.DataFrame
            hsigma : pd.DataFrame

        """
        market_ret = (ret * market_value).sum(axis=1) / \
            market_value.sum(axis=1)
        Re = ret-self.rf
        Re = Re.dropna(axis=0, how='all')
        Rm = market_ret - self.rf

        date_in_src = np.intersect1d(list(Re.index), list(Rm.index))
        beta = pd.DataFrame(index=self.datetime, columns=Re.columns)
        hsigma = pd.DataFrame(index=self.datetime, columns=Re.columns)
        # weights
        alpha = 1-np.exp(np.log(0.5)/63)
        w = [(1-alpha)**(T-i) for i in range(252)]
        w = w/sum(w)

        for end in tqdm(self.datetime):
            ind = np.argwhere(date_in_src == end)[0, 0]
            start = date_in_src[ind-T+1]
            for code in Re.columns:
                y = Re.loc[start:end, code]
                x = Rm[start:end]
                X = sm.add_constant(x)
                wls_model = sm.WLS(y, X, weights=w)
                results = wls_model.fit()
                beta.loc[end, code] = results.params[0]
                hsigma.loc[end, code] = results.resid.std()
        return beta, hsigma

    def get_rstr(self,
                 ret: pd.DataFrame,
                 T: int,
                 halflife: int,
                 L: int) -> pd.DataFrame:
        """
        计算动量因子

        Parameters
        ----------
        ret : pd.DataFrame
            index=datetime, columns=asset
            资产原始日收益率
            ret的开始日期应至少比self.start早T-1期
        T : int
            长期动量周期
        halflife : int
            半衰期
        L : int
            短期动量周期

        Returns
        -------
        rstr : pd.DataFrame

        """
        def func(series, T, halflife, L):
            series = series.ewm(halflife=halflife).mean()
            return series[-1]-series[-(T-L)]

        rt = ((ret+1)/(self.rf+1)).apply(np.log)
        rstr = rt.rolling(T).apply(lambda x: func(x, T, halflife, L))
        return rstr.loc[self.datetime, :]

    def get_dastd(self,
                  ret: pd.DataFrame,
                  T: int,
                  halflife: int) -> pd.DataFrame:
        """
        计算超额收益年化波动率

        Parameters
        ----------
        ret : pd.DataFrame
            index=datetime, columns=asset
            资产原始日收益率
            ret的开始日期应至少比self.start早T-1期
        T : int
            计算区间长度
        halflife : int
            半衰期

        Returns
        -------
        dastd : pd.DataFrame

        """
        re = ret-self.rf
        re = re.sub(re.mean(axis=1), axis=0)
        re = re**2
        dastd = re.rolling(T).apply(
            lambda x: x.ewm(halflife=halflife).mean()[-1])
        return dastd.loc[self.datetime, :]

    def get_cmra(self,
                 ret: pd.DataFrame,
                 M: int) -> pd.DataFrame:
        """
        计算年度超额收益率离差

        Parameters
        ----------
        ret : pd.DataFrame
            index=datetime, columns=asset
            资产原始日收益率
            ret的开始日期应至少比self.start早M*21期
        M : int
            计算区间的月份数

        Returns
        -------
        cmra : pd.DataFrame

        """
        rt = ((ret+1)/(self.rf+1)).apply(np.log)
        ZT = rt.rolling(21*M).sum()
        cmra = ZT.rolling(21*M).apply(lambda x: max(x)-min(x))
        return cmra.loc[self.datetime, :]

    def get_nlsize(self,
                   size3: pd.DataFrame,
                   size: pd.DataFrame) -> pd.DataFrame:
        """
        计算非线性因子

        Parameters
        ----------
        size3 : pd.Dataframe
            index=datetime, columns=asset
            市值因子的立方
        size : pd.DataFrame
            index=datetime, columns=asset
            市值因子

        Returns
        -------
        nlsize : pd.DataFrame

        """
        nlsize = pd.DataFrame(
            index=self.datetime, columns=size.columns)

        for i in tqdm(self.datetime):
            x = size.loc[i, :]
            y = size3.loc[i, :]
            if not x.isna().all():
                X = sm.add_constant(x)
                model = sm.OLS(y, X, missing='drop')
                results = model.fit()
                y_fitted = results.fittedvalues
                nlsize.loc[i, :] = y-y_fitted
        return nlsize

    def get_btop(self,
                 pb: pd.DataFrame) -> pd.DataFrame:
        """
        计算账面市值比因子

        Parameters
        ----------
        pb : pd.DataFrame
            index=datetime, columns=asset
            pb数据

        Returns
        -------
        bp : pd.DataFrame

        """
        pb = pb.loc[self.datetime, :]
        bp = 1./pb
        return bp.replace([np.inf, -np.inf], 0)

    def get_turnover(self,
                     volume: pd.DataFrame,
                     total_share: pd.DataFrame,
                     T: int) -> pd.DataFrame:
        """
        计算换手率因子

        Parameters
        ----------
        volume : pd.DataFrame
            index=datetime, columns=asset
            日成交量数据
            volume的开始日期应至少比self.start早T期
        total_share : pd.DataFrame
            index=datetime, columns=asset
            流通股本数据
            total_share的开始日期应至少比self.start早T期
        T : int
            计算区间长度

        Returns
        -------
        turnover : pd.DataFrame

        """
        turnover = (volume/total_share).rolling(T).sum().apply(np.log)
        return turnover.loc[self.datetime, :]

    def get_growth_factor(self,
                          f_data: pd.DataFrame,
                          asset: List) -> pd.DataFrame:
        """
        计算成长因子

        Parameters
        ----------
        f_data : pd.DataFrame
            index=datetime, columns=asset
            季频财报数据，营业收入/净利润
        asset : List
            资产列表

        Returns
        -------
        factor : pd.DataFrame

        """
        def get_growth_rate(dep):
            x = sm.add_constant(np.array([1, 2, 3, 4, 5]))
            model = sm.OLS(dep, x)
            results = model.fit()
            return results.params[1]/dep.mean()

        f_data = f_data.resample('Y').last()
        factor = f_data.rolling(5).apply(get_growth_rate)
        factor = factor.resample('M').ffill().shift(3)
        factor = factor.dropna(axis=0, how='all')

        stand_df = pd.DataFrame(
            index=pd.date_range(start=factor.index[0], 
                                end=self.datetime[-1]), columns=asset)
        factor = factor.reindex_like(stand_df).ffill()
        return factor.loc[self.datetime, :]

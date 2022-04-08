# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:11:34 2022

@author: PCXU
@contact: xupengchengmail@163.com
"""

from scipy.optimize import minimize
import pandas as pd
import numpy as np


class BarraOptimizer:
    """
    这个类用来定义进行Barra组合优化的相关函数
    """

    def __init__(self,
                 lambd: float = 0,
                 max_w: float = 0.1):
        """


        Parameters
        ----------
        lambd : float, optional
            风险厌恶系数. The default is 0.
        max_w : float, optional
            单资产最大权重. The default is 10.0%.

        """
        self.lambd = lambd
        self.max_w = max_w

    def get_weight(self,
                   ret: pd.Series,
                   assets_pool: pd.Index,
                   beta_industry: pd.DataFrame,
                   beta_style: pd.DataFrame,
                   sigma: pd.DataFrame = None) -> pd.Series:
        """
        获取使风险调整后的收益最大化的权重

        Parameters
        ----------
        ret : pd.Series
            资产收益率的估计
        assets_pool : pd.Index
            股票池
        beta_industry : pd.DataFrame
            index=assets, columns=industry
            要中性化的行业因子暴露
        beta_style : pd.DataFrame
            index=assets, columns=style
            要中性化的风格因子暴露
        sigma : pd.DataFrame, optional
            资产协方差矩阵 The default is None.

        Returns
        -------
        pd.Series

        """
        r = ret[assets_pool].values.reshape((1, -1))
        beta_industry = beta_industry.loc[assets_pool, :].values
        beta_style = beta_style.loc[assets_pool, :].values
        X_ini = np.zeros((1, len(assets_pool)))

        def target_func(r):
            def func(x): return -1*np.matmul(x.reshape(1, -1), r.T)[0, 0]
            return func

        def neutral_func(array):
            def func(x): return np.matmul(x.reshape(1, -1), array)[0, 0]
            return func
        
        # 持仓上限限制
        bnds = tuple((-1*self.max_w, self.max_w) for _ in assets_pool)
        # 货币中性限制
        cons = [{'type': 'eq', 'fun': lambda x: x.sum()},]
        for i in range(beta_industry.shape[1]):
            # 行业因子暴露限制
            cons.append({'type': 'eq', 'fun': neutral_func(
                beta_industry[:, i].reshape(-1, 1))})
        for i in range(beta_style.shape[1]):
            # 风格因子暴露限制
            cons.append({'type': 'eq', 'fun': neutral_func(
                beta_style[:, i].reshape(-1, 1))})
        cons = tuple(cons)
        
        res = minimize(target_func(r), X_ini, method='SLSQP',
                       bounds=bnds, constraints=cons)
        if res.success:
            return pd.DataFrame(res.x,
                                index=assets_pool, columns=['weight'])
        else:
            print(res.message)

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:22:24 2022

@author: PCXU
"""


# 无风险利率
Rf_annual = 0.02
Rf_daily = (Rf_annual+1)**(1/252)-1

# 根目录
ROOT_PATH = 'D:\\work\\实习\\衍盛\\Barra\\'
# 数据根目录
DATA_ROOT_PATH = ROOT_PATH+'data\\'
# 价量数据
PV_PATH = DATA_ROOT_PATH+'daily_prices_2005\\adj_prices.ftr'
# 基本面数据
FR_PATH = DATA_ROOT_PATH+'TWSE\\daily_fundamentals.pkl'
# 行业数据
INDUSTRY_PATH = DATA_ROOT_PATH+'industry_info.pkl'
# 季度数据
QUARTER_DATA = DATA_ROOT_PATH+'quarter_TWSE\\'
# 因子值目录
FACTOR_PATH = DATA_ROOT_PATH+'factor\\'
# 因子收益率
FACTOR_RETURN = FACTOR_PATH+'factor_return\\'

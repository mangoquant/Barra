from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import re


def load_daily_data(ticker_list, dir_path):
    dfs = []
    for ticker in tqdm(ticker_list):
        df = pd.read_csv(dir_path + "{}.csv".format(str(ticker)), thousands=','
                         ).drop(columns=['Unnamed: 0'])
        df['asset'] = str(ticker)
        dfs.append(df)
    else:
        ndf = pd.concat(dfs).reset_index(drop=True)
        ndf['Date'] = ndf['Date'].astype(str)
        ndf['Date'] = pd.to_datetime(ndf['Date'])
        ndf['證券代號'] = ndf['證券代號'].astype('str')
        ndf.index = pd.MultiIndex.from_frame(
            ndf[['Date', '證券代號']], names=('datetime', 'asset'))
        ndf = ndf.dropna(how='all').sort_index()
        ndf = ndf.drop(['Date', '證券代號'], axis=1)
        return ndf


def load_FR_pkl(path):
    df = pd.read_pickle(path)
    df = df[['證券名稱', '開盤價', '最高價', '最低價', '收盤價',
             '成交股數', '股價淨值比', '發行股數']]
    df.columns = ['asset_name', 'open', 'high', 'low', 'close',
                  'volume', 'pb', 'total_share']
    df.index = df.index.set_names(['datetime', 'asset'])
    fields = ['open', 'high', 'low', 'close',
              'volume', 'pb', 'total_share']
    df[fields] = df[fields].astype('str')
    for f in fields:
        df[f] = df[f].apply(lambda x: x.replace(
            ',', '').replace('--', '0.0').replace('-', '0.0'))
    df[fields] = df[fields].astype('float64')
    return df


def load_quarter_data(path, datetime, ticker_list):
    net_income = pd.DataFrame(index=datetime, columns=ticker_list)
    oper_revenue = pd.DataFrame(index=datetime, columns=ticker_list)

    file_lst = os.listdir(path)
    file_lst.sort()
    for i in tqdm(range(len(file_lst))):
        df = pd.read_excel(os.path.join(
            path, file_lst[i]), header=2, keep_default_na=False)
        df = df.iloc[:, [0, 2, 9]]
        df.columns = ['code', 'oper_revenue', 'net_income']
        df['code'] = df['code'].astype('str')
        df['code'] = df['code'].apply(lambda x: x.replace(' ', ''))

        flag = '^[1-9]{1}[0-9]{3}$'
        drop_lst = []
        for j in range(df.shape[0]):
            code = df.loc[j, 'code']
            if code in ticker_list:
                if re.match(flag, code):
                    continue
            drop_lst.append(j)
        df = df.drop(drop_lst, axis=0)
        df = df.replace('', np.nan)

        df[['oper_revenue', 'net_income']] = df[[
            'oper_revenue', 'net_income']].astype('float64')*1000
        df = df.set_index('code')

        net_income.loc[datetime[i], df.index] = df['net_income'].values
        oper_revenue.loc[datetime[i], df.index] = df['oper_revenue'].values

    return net_income.astype('float64'), oper_revenue.astype('float64')


def get_TTM(df):
    # 计算单季度数据
    df_Q1 = df[df.index.month == 3]

    df_Q2 = df[df.index.month == 6]
    Q2_np = df_Q2.values-df_Q1.values
    Q2_np = pd.DataFrame(Q2_np, index=df_Q2.index, columns=df_Q2.columns)

    df_Q3 = df[df.index.month == 9]
    Q3_np = df_Q3.values-df_Q2.values
    Q3_np = pd.DataFrame(Q3_np, index=df_Q3.index, columns=df_Q3.columns)

    df_Q4 = df[df.index.month == 12]
    Q4_np = df_Q4.values-df_Q3.values[0:-1, :]
    Q4_np = pd.DataFrame(Q4_np, index=df_Q4.index, columns=df_Q4.columns)

    df_Q = pd.concat([df_Q1, Q2_np, Q3_np, Q4_np])
    df_Q.sort_index(inplace=True)

    # 计算TTM
    df_TTM = df_Q.rolling(4).sum().dropna(how='all')
    return df_TTM


def load_feather_file(file_path):
    df = pd.read_feather(file_path)
    return df


def save_factor_to_pkl(factor_path, factor_tag, factor_name, factor_data):
    factor_path = os.path.join(factor_path, factor_tag)
    if not os.path.exists(factor_path):
        os.makedirs(factor_path)

    factor_data.to_pickle(os.path.join(
        factor_path, "{}.pkl".format(factor_name)))

    return None


def load_factor_from_pkl(factor_path, factor_tag, factor_name):
    factor_path = os.path.join(factor_path, factor_tag)

    try:
        factor = pd.read_pickle(os.path.join(
            factor_path, "{}.pkl".format(factor_name)))
        return factor
    except Exception as e:
        print(factor_name, e)
        return None

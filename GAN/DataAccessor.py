# encoding=utf-8
from enum import Enum
from sqlalchemy import create_engine
import pandas as pd
import pymysql, time
import stockstats
import subprocess as sub
from StockNorm import StockNorm


class DataType:
    index = 0       # 2676天
    stock = 1
    code = 2
    index_data = 3


class DataAccessor(object):
    def __init__(self, name, password, db_ip, db_port):
        self.name = name
        self.password = password
        self.db_ip = db_ip
        self.db_port = db_port
        self.datatype = DataType

    def getSqlDb(self, dataType, code=None):
        """
        Effect:
            根据dataType以及code代码选取对于的数据库和相应的sql语句
        """
        if dataType == self.datatype.index:
            table_name = 'index_data_399606'
            db_name = 'stock_db'
            sql = 'SELECT date,close,open,high,low,volume from ' + table_name + ' WHERE date BETWEEN \'2000-01-04\' AND \'2018-08-01\'' \
                                                          ' ORDER BY date '
        elif dataType == self.datatype.stock and code is not None:
            table_name = 'stockdata_' + code
            db_name = 'stock_db'
            sql = 'SELECT date,close,open,high,low,turn,volume from ' + table_name + ' WHERE date BETWEEN \'2010-12-18\' AND ' \
                                                                          '\'2018-12-31\' ORDER BY date '
        elif dataType == self.datatype.code:
            db_name = 'predict'
            sql = "select code,name from stock_code_name"
        elif dataType == self.datatype.index_data and code is not None:
            table_name = 'index_data_' + code
            db_name = 'stock_db'
            sql = 'SELECT date,high from ' + table_name + ' ORDER BY date '
        else:
            raise IndexError('参数错误,缺失code或者dataType参数错误')

        return db_name, sql

    def getData(self, dataType, code=None):
        """
        Effect:
            从数据库中获取数据
        Return：
            df: 对应股票或者指数的单日最高价的所有数据(type: pd.DataFrame)
        """
        db_name, sql = self.getSqlDb(dataType, code)

        engine = create_engine(
            'mysql://' + self.name + ':' + self.password + '@' + self.db_ip + ':'
            + self.db_port + '/' + db_name + '?charset=utf8')

        df = pd.read_sql(sql, engine)
        return df

    def mergeData(self, code, isStock=True):
        """
        按照随机生成的index的编号，取出对应的5组数据
        Param：
            code: <list> 包含所有候选股票的代码
        Return:
            df_merge: <pd.DataFrame> 合成的对应的数据表
        """
        index_df = self.getData(0)
        index_df = index_df.drop_duplicates()
        first = True

        if isStock:
            flag = 1
        else:
            flag = 3

        for code_ in code:
            stock_df = self.getData(flag, code_)
            stock_df = stock_df.drop_duplicates()

            if first:
                df_merge = pd.merge(index_df, stock_df, on='date')
                first = False
            else:
                df_merge = pd.merge(df_merge, stock_df, on='date')

        return df_merge

    def getSyntheticData(self, which, is4501):
        if is4501:
            dir1 = str(4501)
        else:
            dir1 = str(3917)

        path = './testDataSet/' + dir1 + '/' + str(which) + '.csv'

        return pd.read_csv(path)

    def getStockCode(self, data_path='./stock'):
        stock = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line = line.split('.')
                stock.append(line[0])

        length = len(stock)
        return stock, length

    def get_some_indictor(self, N, df):
        lis = list(['ma', 'ema', 'bias',
                    # 'closesub5', 'closesub10', 'closesub15',
                    # 'closeadd5', 'closeadd10', 'closeadd15'
                    ])
        df = pd.concat([df, pd.DataFrame(columns=lis)], axis=1)
        total_rows = df.shape[0] - N + 1
        inter = N
        for i in range(1, total_rows):
            close_inter = df.iloc[i:inter+i, 1].values
            stNorm = StockNorm(df.iloc[inter+i-1, 1], close_inter, inter)
            MA = stNorm.movingAverages()
            EMA = stNorm.exponentialMovingAverages()
            bias = stNorm.bias(MA)
            # close = df.iloc[inter+i-1::inter+i, 1].values
            # closesub5 = close[0] - 5
            # closesub10 = close[0] - 10
            # closesub15 = close[0] - 15
            # closeadd5 = close[0] + 5
            # closeadd10 = close[0] + 10
            # closeadd15 = close[0] + 15
            df.iloc[inter+i-1, [6, 7, 8]] = MA, EMA, bias
            # df.iloc[inter+i-1, [9, 10, 11, 12, 13, 14]] = \
            #     closesub5, closesub10, closesub15, closeadd5, closeadd10, closeadd15

        return df

    def get_libraries_indictor(self, df):
        stockStat = stockstats.StockDataFrame.retype(df)
        # print(stockStat)
        stock_index = ['open', 'high', 'low',
                       # 'volume',  #'turn', 'volume',
                       # 'ma',  # Moving averages
                       # 'ema',  # Exponential movement index
                       # 'bias',  # Bias
                       # 'boll', 'boll_ub', 'boll_lb',  # bolling band
                       # 'rsi_6', 'rsi_12',  # Relative Strength index
                       # 'pdi', 'mdi', 'dx', 'adx', 'adxr',  # Directional movement index
                       # 'macd', 'macds', 'macdh',  # MACD
                       # 'kdjk', 'kdjd', 'kdjj',  # Stochastic index
                       # 'closesub5', 'closesub10', 'closesub15',
                       # 'closeadd5', 'closeadd10', 'closeadd15',
                       'close'  # 预测指标
                       ]
        df = stockStat[stock_index]
        return df


if __name__ == '__main__':
    name = 'root'
    password = 'szU@654321'
    db_ip = '210.39.12.25'
    db_port = '50002'
    code = '399606'
    import numpy as np
    import matplotlib.pyplot as plt

    pymysql.install_as_MySQLdb()
    accessor = DataAccessor(name, password, db_ip, db_port)
    df = accessor.getData(0, code)
    df = accessor.get_some_indictor(5, df)
    df = accessor.get_libraries_indictor(df)
    df.drop(df.index[0:5], inplace=True)
    df.to_csv('./indexData/' + code + '.csv', index=True)

    # stock, length =accessor.getStockCode()
    # print(length)
    # while(length>0):
    #     length -= 1.2.1.2
    #     df = accessor.getData(1.2.1.2, stock[length])
    #     df = accessor.get_some_indictor(10, df)
    #     df = accessor.get_libraries_indictor(df)
    #     df.to_csv('./stockData/' + stock[length] + '.csv', index=True)
    # df.to_csv('./stockData/'+code+'.csv', index=False, header=False)

    # df1 = pd.concat([df, pd.DataFrame(columns=list(['MA', 'EMA', 'bias']))], axis=1.2.1.2)
    print(df)

    data = df.iloc[:, -1].values
    print(data)
    # print(len(data))
    # print(np.max(data))
    # print(np.min(data))

    plt.plot(data)
    plt.grid()
    plt.show()

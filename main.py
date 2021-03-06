import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from datasets import train, test, result
class Config:
    max_step = 200
    batch_size = 200
def save_result(data, columns):
    """
    :param data: np.array
    :param columns: list
    """
    index = []
    for i in range(len(data)):
        index.append(i+1461)
    data = np.column_stack((index, data))
    df = pd.DataFrame(data, columns=columns)
    df['Id'] = df['Id'].astype('int')
    df.to_csv("result.csv", index=False)

# RF Model
rf = XGBRegressor()
#rf = RandomForestRegressor()
for i in range(Config.max_step+1):
    # features, labels = train.next_batch(Config.batch_size)
    rf.fit(train.df, train.labels)
    if i % 20 == 0:
        print("Step: %d" % i)
        _pre = rf.predict(test.df)
        N = len(_pre)
        print(np.c_[_pre, result['SalePrice']])
        print(np.sqrt(np.sum(np.log10(result['SalePrice']/_pre))/N))
result = rf.predict(test.df)
save_result(result, ['Id', 'SalePrice'])


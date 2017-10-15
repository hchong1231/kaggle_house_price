import numpy as np
import pandas as pd
from sklearn import preprocessing
train_data = pd.read_csv("train.csv", header=0)
test_data = pd.read_csv("test.csv", header=0)
result = pd.read_csv("sample_submission.csv", header=0)
class Datasets:
    def __init__(self, data, train_flag):
        self.pos = 0
        self.num_examples = len(data)
        self.df = data
        if train_flag:
            self.labels = data['SalePrice']
            self.df = data.iloc[:, :-1]
        self.df = self.preprocess(self.df)
    def preprocess(self, data):
        # drop columns
        data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
                        'LotFrontage', 'GarageCond', 'GarageType', 'GarageYrBlt',
                        'GarageQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond',
                          'MasVnrArea'], axis=1)
        # transfer string -> interger
        for x in data.columns:
            if data[x].dtype == object:
                data = pd.concat((data, pd.get_dummies(data[x], prefix=x)), axis=1)
                del data[x]
            else:
                data[x] = data[x].fillna(data[x].median())
                data[x] = data[x].apply(lambda m: (m-data[x].mean())/data[x].std())
        return data
    def next_batch(self, batch_size):
        features = self.df.iloc[self.pos:self.pos+batch_size, :]
        labels = self.labels[self.pos:self.pos+batch_size]
        self.pos += batch_size
        if self.pos >= self.num_examples:
            self.pos = 0
        return features, labels
train = Datasets(train_data, True)
test = Datasets(test_data, False)
for x in train.df.columns:
    if x in test.df.columns:
        pass
    else:
        train.df = train.df.drop(x, axis=1)
print(len(train.df.columns), len(test.df.columns))
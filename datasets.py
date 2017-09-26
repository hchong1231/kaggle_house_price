# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
train_data = pd.read_csv("train.csv", header=0, delimiter=',')
test_data = pd.read_csv("test.csv", header=0, delimiter=',')
result = pd.read_csv("gender_submission.csv", header=0, delimiter=',')
result = result['Survived']
class Datasets:
    def __init__(self, data, is_training):
        self.df = self.preprocess(data)
        if is_training:
            #self.df = self.oversampling(self.df)
            self.labels = self.df['Survived']
            self.df = self.df.iloc[:, 1:]
        self.num_examples = len(self.df)
        self.pos = 0
    def oversampling(self, data):
        pos = data[data['Survived'] == 1]
        neg = data[data['Survived'] == 0]
        n_pos = len(pos)
        n_neg = len(neg)
        if n_pos > n_neg:
            neg = pd.concat([neg, neg.sample(n=n_pos-n_neg, replace=True)])
        else:
            pos = pd.concat([pos, pos.sample(n=n_neg - n_pos, replace=True)])
        data = pd.concat([neg, pos])
        return data
    def preprocess(self, data):
        data = data.drop(labels=['Cabin', 'PassengerId', 'Ticket', 'Name'], axis=1)
        data['Sex'][data['Sex'] == 'male'] = 0
        data['Sex'][data['Sex'] == 'female'] = 1
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        data['Fare'] = data['Fare'].apply(lambda x: (x-data['Fare'].min())/(data['Fare'].max()- data['Fare'].min()))
        # data format
        data = pd.concat([data, pd.get_dummies(data['Embarked'], prefix='Embarked')], axis=1)
        data = pd.concat([data, pd.get_dummies(data['Pclass'], prefix='Pclass')], axis=1)
        data = data.drop(labels=['Embarked', 'Pclass'], axis=1)
        # fill null nan
        for x in data.columns:
            data[x][data[x].isnull()] = data[x].median()
        data['Age'][data['Age'] < 15.0] = 0
        data['Age'][data['Age'] >= 15.0] = 1
        return data
    def next_batch(self, batch_size):
        features = self.df.iloc[self.pos:self.pos+batch_size, :]
        labels = self.labels[self.pos:self.pos+batch_size]
        self.pos += batch_size
        if self.pos >= self.num_examples:
            self.pos = 0
        return features, labels
# delete features of train dataset which is not in test dataset
def train_test_feature_equal(train, test):
    for x in train.columns:
        if x in test.columns:
            pass
        else:
            if x != 'Survived':
                train = train.drop(x, axis=1)
    return train, test
train_div = int(len(train_data)*0.2)
train = Datasets(train_data.iloc[:, :], True)
validation = Datasets(train_data.iloc[:train_div, :], True)
test = Datasets(test_data, False)
train.df, test.df = train_test_feature_equal(train.df, test.df)

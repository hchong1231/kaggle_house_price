# -*- coding: UTF-8 -*-
from sklearn.ensemble import RandomForestClassifier
from datasets import train, test, result, validation
import  numpy as np
import pandas as pd
class Config:
    max_step = 200
    batch_size = 150

def save_result(data, columns):
    """
    :param data: np.array 
    :param columns: list
    """
    index = []
    for i in range(len(data)):
        index.append(i+892)
    data = np.column_stack((index, data))
    df = pd.DataFrame(data, columns=columns)
    df = df.astype('int')
    df.to_csv("result.csv", index=False)


rf = RandomForestClassifier(n_estimators=200, max_features=6)
for i in range(Config.max_step+1):
    features, labels = train.next_batch(batch_size=Config.batch_size)
    rf.fit(features, labels)
    if i % 10 == 0:
        print("Step %d" %i)
        print("Train_set accuracy %.5lf" %rf.score(train.df, train.labels))
        print("Validation_set accuracy %.5lf" % rf.score(validation.df, validation.labels))
        print("Test_set accuracy %.5lf" %rf.score(test.df, result))
predict = rf.predict(test.df)
save_result(predict, ['PassengerId', 'Survived'])
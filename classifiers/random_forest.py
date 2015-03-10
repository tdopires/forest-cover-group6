# This is just sample code we used to get started on the use of ScikitLearn
# It was found here, at this topic: http://bit.ly/1FwEoOF
# To run it, use the command: python random_forest.py

import os, datetime
import pandas as pd
from sklearn import ensemble

data_version = 3

if __name__ == "__main__":
  #loc_submission = os.path.normpath("C:/Users/Maria Matthes/Documents/GitHub/forest-cover-group6/data/result.csv")
  loc_train = "../data/train-" + str(data_version) + ".csv"
  loc_test = "../data/test-" + str(data_version) + ".csv"
  loc_submission = "../data/result-data" + str(data_version) + "-" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + ".csv"

  df_train = pd.read_csv(loc_train)
  df_test = pd.read_csv(loc_test)

  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]
  y = df_train['Cover_Type']
  test_ids = df_test['Id']

  clf = ensemble.RandomForestClassifier()

  clf.fit(X_train, y)

  with open(loc_submission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))

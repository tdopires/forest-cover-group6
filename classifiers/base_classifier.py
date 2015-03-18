import pandas as pd
import datetime

class MLClassifier:

  def __init__(self, clf_id, train_file, test_file, output_folder):
    self.df_train =  pd.read_csv(train_file)
    self.df_test = pd.read_csv(test_file)
    self.output_file = output_folder + "/result-" + clf_id + "-" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + ".csv"

  def parse_data(self):
    feature_cols = [col for col in self.df_train.columns if col not in ['Cover_Type','Id']]

    train_instances = self.df_train[feature_cols]
    test_instances = self.df_test[feature_cols]
    train_classes = self.df_train['Cover_Type']
    test_ids = self.df_test['Id']

    return (train_instances, train_classes, test_instances, test_ids)

  def output_result_to_file(self, test_ids, result):
    with open(self.output_file, "wb") as outfile:
      outfile.write("Id,Cover_Type\n")
      for e, val in enumerate(list(result)):
        outfile.write("%s,%s\n"%(test_ids[e],val))

    print "Result written on: " + self.output_file

  
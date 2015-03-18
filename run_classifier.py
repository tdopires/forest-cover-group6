import sys, os, datetime

from classifiers import svm, random_forest

import argparse

def main():

  parser = argparse.ArgumentParser(description='Run a machine learning classifier (previously implemented and configured) on some dataset.')
  parser.add_argument('classifier', metavar='classifier_name', type=str,
                     help='the name of the python file to run')
  parser.add_argument('data_folder_path', metavar='data_folder_path', type=str,
                     help='the path to the folder containing the data to use on train/test')

  args = parser.parse_args()

  train_file = args.data_folder_path + "/train.csv"
  test_file = args.data_folder_path + "/test.csv"
  output_folder = args.data_folder_path #make a parameter for this later

  if args.classifier == "svm":
    clf = svm.SVM(train_file, test_file, output_folder)
  elif args.classifier == "random_forest":
    clf = random_forest.RandomForest(train_file, test_file, output_folder)
  else:
    print "Error: Unknown classifier"
    sys.exit(0)

  clf.run()

if __name__ == "__main__":
  main()

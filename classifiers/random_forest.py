from base_classifier import MLClassifier

from sklearn import ensemble

class RandomForest(MLClassifier):

  def __init__(self, train_file, test_file, output_folder):
    MLClassifier.__init__(self, "rf", train_file, test_file, output_folder)

  def run(self):
    (train_instances, train_classes, test_instances, test_ids) = self.parse_data()

    clf = ensemble.RandomForestClassifier()

    clf.fit(train_instances, train_classes)

    result_class_with_labels = clf.predict(test_instances)

    self.output_result_to_file(test_ids, result_class_with_labels)


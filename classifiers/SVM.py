from base_classifier import MLClassifier

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedKFold


class SVM(MLClassifier):

  def __init__(self, train_file, test_file, output_folder):
    MLClassifier.__init__(self, "svm", train_file, test_file, output_folder)

  def run(self):
   
    (train_instances, train_classes, test_instances, test_ids) = self.parse_data()

    (X_train, X_test, y_train) = (train_instances, test_instances, train_classes)

    svc = svm.SVC(class_weight='auto')
    param_grid = {'kernel': ['rbf'], #'poly', 'rbf', 'sigmoid'
                  'C': [1e0, 1e1, 1e2, 1e3, 1e4]} #
    strat_2fold = StratifiedKFold(y_train, n_folds=2)
    print "    Parameters to be chosen through cross validation:"
    for name, vals in param_grid.iteritems():
      print "        {0}: {1}".format(name, vals)
    clf = GridSearchCV(svc, param_grid, n_jobs=1, cv=strat_2fold)
    clf.fit(X_train, y_train)
    print "== Best Parameters:", clf.best_params_
    result_class_with_labels = clf.predict(X_test)
    
    self.output_result_to_file(test_ids, result_class_with_labels)

from base_classifier import MLClassifier

#from sklearn import multiclass
#from sklearn.svm import LinearSVC

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedKFold


class SVM(MLClassifier):

  def __init__(self, train_file, test_file, output_folder):
    MLClassifier.__init__(self, "svm", train_file, test_file, output_folder)

  def run(self):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=200, n_features=2, 
                      centers=((0,0), (4, 4)),
                      cluster_std=1.0)
    print 'X', len(X)
    print 'y', len(y)

    (train_instances, train_classes, test_instances, test_ids) = self.parse_data()

    #clf = multiclass.OneVsRestClassifier(LinearSVC())
    #clf.multilabel = True
    #clf.fit(train_instances, train_classes)
    #result_class_with_labels = clf.predict(test_instances)
    print 'train_instances' + str(len(train_instances))
    print 'test_instances' + str(len(test_instances))

    X_train, X_test, y_train, y_test = train_test_split(train_instances, test_instances, test_size=0.25)

    svc = svm.SVC(class_weight='auto')
    param_grid = {'kernel': ['poly'],
                  'C': [1e0, 1e1, 1e2, 1e3, 1e4]}
    strat_2fold = StratifiedKFold(y_train, n_folds=2)
    print "    Parameters to be chosen through cross validation:"
    for name, vals in param_grid.iteritems():
      print "        {0}: {1}".format(name, vals)
    clf = GridSearchCV(svc, param_grid, n_jobs=1, cv=strat_2fold)
    clf.fit(X_train, y_train)
    print "== Best Parameters:", clf.best_params_
    result_class_with_labels = clf.predict(X_test)
    acc = len(np.where(result_class_with_labels == y_test)[0]) / float(len(y_pred))
    print "== Accuracy:", acc
    
    self.output_result_to_file(test_ids, result_class_with_labels)

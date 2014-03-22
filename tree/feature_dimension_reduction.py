from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD
import numpy as np

max_number_features = 2000000
print 'loading traininig data'
X_train, Y_train = load_svmlight_file("train-fast.csv", n_features=max_number_features, multilabel=True)
print 'loading test data'
X_test, Y_test = load_svmlight_file("test-fast.csv", n_features=max_number_features, multilabel=True)

svd = TruncatedSVD(n_components=200)
X_test_reduced = svd.fit_transform(X_test)
np.savetxt('X_test_reduced.txt', X_test_reduced)

svd = TruncatedSVD(n_components=200)
X_train_reduced = svd.fit_transform(X_train)
np.savetxt('X_train_reduced.txt', X_train_reduced)
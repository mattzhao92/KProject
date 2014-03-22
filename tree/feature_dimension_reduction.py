from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD
import numpy as np

max_number_features = 2000000

# print 'loading test data'
# X_test, Y_test =   load_svmlight_file("test-remapped.csv", n_features=max_number_features, multilabel=True)

print 'loading traininig data'
X_train, Y_train = load_svmlight_file("train-quarter-remapped.csv", n_features=max_number_features, multilabel=True)


# print 'begins test data reduction'
# svd = TruncatedSVD(n_components=200)
# X_test_reduced = svd.fit_transform(X_test)
# np.savetxt('X_test_reduced.txt', X_test_reduced)

# print 'begins training data reduction'
# svd = TruncatedSVD(n_components=200)
# X_train_reduced = svd.fit_transform(X_train)
# np.savetxt('X_train_reduced.txt', X_train_reduced)

# Generating training data labels
# training_labels = open('train-labels', 'w')

# for i in range(len(Y_train)):
# 	y_row = [str(int(n)) for n in Y_train[i]]
# 	training_labels.write(' '.join(y_row)+"\n")
# training_labels.close()
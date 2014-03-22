import sys
import json
import random
import numpy as np
import heapq, random
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from scipy.sparse import vstack
from sklearn import preprocessing

max_number_features = 2000000
# Knn classifier benchmark

print 'reading testing data ... '
X_test,Y_test = load_svmlight_file("test-remapped.csv",n_features=max_number_features, multilabel=True)

print 'reading training data ... '
X_train,Y_train = load_svmlight_file("train-remapped.csv",n_features=max_number_features, multilabel=True)

print 'computing inverse_document_frequency from traninng data'
transformer = TfidfTransformer()
transformer.fit_transform(X_train)
idf_weights = csr_matrix(transformer.idf_)


def calculateDistance(doc1, doc2, idf_weights):
	assert doc1.shape == doc2.shape

	doc2_tdidf = doc2.multiply(idf_weights)
	doc2_nonzero_indices = doc2.nonzero()[1]
	top3Features = heapq.nlargest(3, doc2_nonzero_indices, key=lambda x: doc2_tdidf[0,x])
	
	featureSet1 = set(doc1.nonzero()[1])
	# check there is at least one feature in doc1 that appears in topThreeFeatures
	topFeaturePresent = False
	for topFeature in top3Features:
		if topFeature in featureSet1:
			topFeaturePresent = True

	if not topFeaturePresent :
		return 10000
	
	featureSet2 = set(doc2_nonzero_indices)
	commonFeatureSet = featureSet1.intersection(featureSet2)

	numFeature1 = len(featureSet1)
	numFeature2 = len(featureSet2)
	numCommonFeature = len(commonFeatureSet)

	return 1.0 * (numFeature1 + numFeature2 - 2 * numCommonFeature) / (numFeature1 + numFeature2 - numCommonFeature)

print 'start classification ... '
outputfile = open('output','w')
outputfile.write("Id,Predicted\n")

for i in range(X_test.shape[0]):
	doc_distance_pairs = []
	test_doc = X_test.getrow(i)
	
	
	# for each train_doc , calculate its distance to test_doc
	for j in range(X_train.shape[0]):
		train_doc = X_train.getrow(j)
		dist = calculateDistance(train_doc, test_doc, idf_weights)
		if dist < 10000:
			doc_distance_pairs.append((j, dist))
	n_closest = heapq.nsmallest(5, doc_distance_pairs, key=lambda pair: pair[1])

	new_n_closest = []
	foundPerfect = False
	for i in range(len(n_closest)):
		if not foundPerfect:
			if n_closest[i][1] == 0.0:
				new_n_closest.append(n_closest[i])
				foundPerfect = True
				break

	n_closest = new_n_closest

	n_label_indices = [x[0] for x in n_closest]	
	results = []
	for i in range(len(n_label_indices)):
		label_index = n_label_indices[i]
		labels = [str(int(label)) for label in Y_train[label_index]]
		results.extend(labels)
		if len(results) >= 3:
			break
	#print results

	outputfile.write(str(i+1) +",")
	if len(results) > 0:
		outputfile.write(' '.join(results)+"\n")
	else:
		outputfile.write("0\n")
outputfile.close()
#print 'start preprocessing tranining data ... '

# trans = TfidfTransformer()
# X_test = trans.fit_transform(X_test)

# print 'X_shape before transformation ', X_train.shape

# X_train_subset = set();
# for i in range(X_test.shape[0]):
# 	n_largest = heapq.nlargest(3, enumerate(X_test.getrow(i).data))

# 	top_indices = [x[0] for x in n_largest]

# 	# iterate all training examples, find docs that share at least one feature with the top 3 feature


# 	for j in range(X_train.shape[0]):
# 		for top_feature_indice in top_indices:
# 			if X_train[j,top_feature_indice] > 0.05: 
# 				X_train_subset.add(j)
# 				break


# matrix = None

# for doc_index in X_train_subset:
# 	if matrix == None:
# 		matrix = csr_matrix(X_train.getrow(doc_index))
# 	else:
# 		matrix = vstack([matrix, X_train.getrow(doc_index)])

# print 'X_shape after transformation ', matrix.shape



            
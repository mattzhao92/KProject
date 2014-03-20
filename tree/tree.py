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

class TrainingDataLoader:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.labelToDoc = {}

    def load(self, train_path, test_path,max_number_features):
        X_train,Y_train = load_svmlight_file(train_path,n_features=max_number_features, multilabel=True)
        X_test,Y_test = load_svmlight_file(test_path,n_features=max_number_features, multilabel=True)
        trans = TfidfTransformer()
        X_test_trans = trans.transform(X_train)
        X_test_trans_row_num = X_test_trans.shape[0]
        X_test_trans_col_num = X_test_trans.shape[1]
        
        for i in range(X_test_trans_row_num):
            n_largest = heapq.nlargest(3, X_test_trans.getrow(i) )
            print X_test_trans.getrow(i)
            print n_largest
        sys.exit(0)




        X_length = X.shape[0];
        Y_length = len(Y)
        assert X_length == Y_length
        self.X_train = X
        self.Y_train = Y

        # start building a dict from the class labels to rows in X_train
        for i in range(X_length):
            doc = self.X_train.getrow(i)
            lables = self.Y_train[i]

            for label in lables:
                if label not in self.labelToDoc:
                    self.labelToDoc[label] = []
                self.labelToDoc[label].append(i)

    # returns a sparse matrix, each row represents a document belong to a label
    def getDocIndicesByLabel(self, label):
    	#print 'getDocIndicesByLabel ',label
        if label in self.labelToDoc:
        	#print 'found something here', len(self.labelToDoc[label])
        	return set(self.labelToDoc[label])
        return set()
        # docs = csr_matrix()
        # if label in self.labelToDoc:
        #     doc_indexes = self.labelToDoc
        #     for doc_index in doc_indexes:
        #         docs = vstack([docs, self.Y_train.getrow(doc_index)])
        # return docs

    def translateIndicesToDocs(self, sets_indices):
        matrix = None
        #print 'translateIndicesToDocs ', sets_indices
        for set_doc_indices in sets_indices:
        	for doc_index in set_doc_indices:
	        	if matrix == None:
	        		matrix = csr_matrix(self.X_train.getrow(doc_index))
	        	else:
	           		matrix = vstack([matrix, self.X_train.getrow(doc_index)])
    	return matrix


class Node:
    def __init__(self,class_label, loader):
        self.parent = None
        self.children = set()
        self.class_label = int(class_label)
        self.classifier = None
        self.loader = loader
        self.X_train = None
        self.Y_train = None
        self.lb = None
        self.depth = 0;


    def getDepth(self):
        if len(self.children) == 0 :
            return 1
        else :
            max_dep = 0
            for node1 in self.children:
                depth = node1.getDepth()
                if depth > max_dep:
                    max_dep = depth
            return max_dep + 1

    def getTrainingDataIndices(self):
        #print 'getTrainingDataIndices', self.class_label
        if len(self.children) == 0:
            return self.loader.getDocIndicesByLabel(self.class_label)
        else:
            listOfIndices = set()
            for child in self.children:
                listOfIndices.update(child.getTrainingDataIndices())
            return listOfIndices

    # doc is an one-dimensional sparse matrix
    def classify(self, doc):
        print 'classify is called ', self.class_label
        if len(self.children) == 0:
            return self.class_label
        else:
            if self.classifier == None:
                self.classifier = Pipeline([
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC(class_weight='auto'))])

                X_train = []
                Y_train = []

                #print 'number of children',len(self.children)
                num_different_labels = 0
                for child in self.children:
                    indices = child.getTrainingDataIndices()
                    if len(indices) > 0:
                        num_different_labels += 1
                        X_train.append(indices)
                        Y_train.extend([[child.class_label] for i in range(len(indices))])

                #print len(X_train)

                # read in real data
                self.X_train = self.loader.translateIndicesToDocs(X_train)
                if num_different_labels > 1 and self.X_train != None:
                    self.Y_train = np.array(Y_train)
                    print self.X_train.shape
                    #print 'Y_train', self.Y_train
                    self.num_different_labels = num_different_labels
                    self.classifier.fit(self.X_train, self.Y_train)
                    self.pickRandomChild = False
                else:
                    self.pickRandomChild = True

             
            #binary_labels = self.classifier.predict(doc)
            #print 'decisionfunc', self.classifier.decision_function(doc)
            #print 'binary_labels', binary_labels
            #predicated_labels = self.lb.inverse_transform(binary_labels)
            print 'fuck pear'
            if self.pickRandomChild:
                return random.sample(self.children, 1)[0].classify(doc)
            else:
                predicated_labels = self.classifier.predict(doc)
                if len(predicated_labels) == 0:
                    raise Exception('No prediction has been made')

                #print 'predicated_labels',predicated_labels
                next_label = predicated_labels[0]
                #print 'next_label ', next_label
                for child in self.children:
                    if child.class_label == next_label:
                        return child.classify(doc)

                raise Exception('No children carries the predicated class label')



class ClassificationTree:
    def __init__(self, loader):
        self.loader = loader
        self.root = Node(0, loader)
        self.depth = 0;
        self.node_dict = {}

    def addNode(self,num1,num2):
        if num2 not in self.node_dict :
            self.node_dict[num2] = Node(num2,self.loader)

        if num1 not in self.node_dict :
            self.node_dict[num1] = Node(num1,self.loader)

        self.node_dict[num1].children.add(self.node_dict[num2])
        self.node_dict[num2].parent = self.node_dict[num1]


    def buildTree(self):
        for node in self.node_dict.keys() :
            if self.node_dict[node].parent == None :
                self.root.children.add(self.node_dict[node])
                self.node_dict[node].parent = self.root
        if 0 in self.node_dict:
            self.root = self.node_dict[0]

    def cacluateDepth(self):
        if self.root == None:
            return;
        currentLevel = []
        nextLevel = []
        currentLevel.append(self.root);

        while len(currentLevel) > 0:
            currNode = currentLevel.pop(0);
            if currNode != None:
                self.depth = currNode.depth
                for child in currNode.children:
                    if child.depth == 0:
                        child.depth = currNode.depth + 1
                        nextLevel.append(child)

            if len(currentLevel) == 0:
                temp = nextLevel
                nextLevel = currentLevel
                currentLevel = temp

    def serialize(self, saveToPath):
        with open(saveToPath, 'w') as f:
            currentLevel = []
            nextLevel = []
            currentLevel.append(self.root);

            while len(currentLevel) > 0:
                currNode = currentLevel.pop(0);
                if currNode != None:
                    for child in currNode.children:
                        f.write('%(parent)s %(child)s %(d1)d  %(d2)d\n' % {"parent": currNode.class_label, "child": child.class_label, "d1":currNode.depth, "d2":child.depth})
                        nextLevel.append(child)

                if len(currentLevel) == 0:
                    temp = nextLevel
                    nextLevel = currentLevel
                    currentLevel = temp

    def removeCycles(self):
        currentLevel = []
        nextLevel = []
        currentLevel.append(self.root);
        while len(currentLevel) > 0:
            currNode = currentLevel.pop(0);
            new_children = set()
            if currNode != None:
                for child in currNode.children:
                    if child.depth == currNode.depth + 1:
                        nextLevel.append(child)
                        new_children.add(child)
                currNode.children = new_children

            if len(currentLevel) == 0:
                temp = nextLevel
                nextLevel = currentLevel
                currentLevel = temp

    def getDepth(self):
        return self.depth

    def classify(self, doc):
        return self.root.classify(doc)


# specify the maximum number of features we consider
max_number_features = 10


# Read in training data
print 'building training data loader ... '
trainingDataloader = TrainingDataLoader()
trainingDataloader.load("train-fast.csv","test-fast.csv", max_number_features)

# Build tree hierarchy
print 'building tree hierarchy ... '
tree = ClassificationTree(trainingDataloader)
for line in open("new_hierarchy.txt") :
    arr = line.split(' ')
    tree.addNode(int(arr[0]), int(arr[1]))
tree.buildTree()

# print 'calculating the depth of the tree ... '
# tree.cacluateDepth()

# print 'removing cycles in the tree hierarchy ... '
# tree.removeCycles()

# print 'saving the tree to new_hierarchy.txt'
# tree.serialize('new_hierarchy.txt')



# Read in the testing data
print 'reading test data ... '
X_test,Y_test = load_svmlight_file("test-fast.csv",n_features=max_number_features, multilabel=True)

# classify
print 'start classifying ... '
num_rows = X_test.shape[0]

if num_rows == 0:
    print '%100 done'
else:
    print '%0 done'
for i in range(num_rows):
    doc = X_test.getrow(i)
    #sys.stdout.write("\033[F") # Cursor up one line
    print tree.classify(doc)
    print '%(number)d%(percent)s done' % {"number" : int((i+1) / float(num_rows) * 100), "percent": "%"}


# tree.root.data = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])

# with open('my_dict.json', 'w') as f:
#     json.dump(tree.node_dict, f)

# print len(tree.root.children)
# print tree.depth(tree.root)

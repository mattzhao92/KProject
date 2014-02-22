from scipy.sparse import coo_matrix
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors

# read file
ins = open( "train.txt", "r" )
line_array = []
for line in ins:
    line_array.append( line )
ins.close()
# split lines into pairs

max_word_code = 0
document = []
dictionary = {}
y_temp=[]
for index_out,line in enumerate(line_array):
    pair_array = line.split()
    y_temp.append([])
    document_dic = {}
    for index,pair in enumerate(pair_array):
        if(pair[-1:]==","):
             y_temp[index_out].append(int(pair[:-1]))
        elif (pair.find(":") == -1):
             y_temp[index_out].append(int(pair))
        else:
            temp_pair = pair.split(':')
            document_dic[temp_pair[0]] = temp_pair[1]
            dictionary[temp_pair[0]] = temp_pair[0]
            if(max_word_code < int(temp_pair[0])) :
                   max_word_code = int(temp_pair[0])
            document.append(document_dic)
y_temp.pop()
y = y_temp
#for document_dic in document:
#    print document_dic

dictionary_size = len(dictionary)
document_bag = []
row = []
col = []
data = []
for index,document_dic in enumerate(document):
    for key, value in document_dic.items():
       row.append(int(index)) 
       col.append(int(key))
       data.append(int(value))

#print document_bag
word_counts = coo_matrix((data,(row,col)), shape = (len(document)+1,max_word_code+1))
#print word_counts.todense()
tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_cou
nts)
X = tf_transformer.transform(word_counts)
print X
    
n_neighbors = 11
weights = 'uniform'
weights = 'distance'
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# test the classifier
print '\n\n'
#print colored('Testing classifier with train-test split', 'magenta', attrs=['bold'])
test_classifier(X, y, clf, test_size=0.2)
def test_classifier(X, y, clf, test_size=0.4):
        #train-test split
        print 'test size is: %2.0f%%' % (test_size*100)
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, t
est_size=test_size)
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)

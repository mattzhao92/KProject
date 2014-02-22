from scipy.sparse import coo_matrix
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors


def readTrainingData(fileName):
    # read file
    ins = open(fileName, "r" )
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
    word_counts = coo_matrix((data,(row,col)), shape = (len(document),max_word_code+1))
    #print word_counts.todense()
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)
    return X, y;


def readTestData(fileName):
    # read file
    ins = open(fileName, "r" )
    line_array = []
    for line in ins:
        line_array.append( line )
    ins.close()
    # split lines into pairs

    max_word_code = 0
    document = []
    dictionary = {}
    for index_out,line in enumerate(line_array):
        pair_array = line.split()
        document_dic = {}
        for index,pair in enumerate(pair_array):
            temp_pair = pair.split(':')
            document_dic[temp_pair[0]] = temp_pair[1]
            dictionary[temp_pair[0]] = temp_pair[0]
            if(max_word_code < int(temp_pair[0])) :
                max_word_code = int(temp_pair[0])
        document.append(document_dic)
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
    word_counts = coo_matrix((data,(row,col)), shape = (len(document),max_word_code+1))
    #print word_counts.todense()
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)
    return X;
   
X_train, y_train = readTrainingData("train.txt");
classifier = Pipeline([
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, y_train)

X_test = readTestData("test.txt");
predicted = classifier.predict(X_test);
for labels in predicted:
    print 'labels '.join(str(x) for x in labels)

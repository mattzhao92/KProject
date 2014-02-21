#!/usr/bin/python
# read file
ins = open( "train.txt", "r" )
line_array = []
for line in ins:
    line_array.append( line )
ins.close()
# split lines into pairs

document = []
document_bow = []
dictionary = {}

document = []
document_bow = []
dictionary = {}
for line in line_array:
    pair_array = line.split()
    document_dic = {}
    for pair in pair_array:
        temp_pair = pair.split(':')
        document_dic[temp_pair[0]] = temp_pair[1]
        dictionary[temp_pair[0]] = temp_pair[0]
    document.append(document_dic)
dictionary_size = len(dictionary)
document_bag = []
for document_dic in document:
    document_bag_array = [0] * dictionary_size
    for i in range(dictionary_size):
        document_bag_array[i]=document_dic.get(i,0)
    document_bag.append(document_bag_array)
result = coo_matrix(document_bag).todense()
print result

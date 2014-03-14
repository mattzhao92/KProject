import sys
import json
import numpy as np
from scipy.sparse import csr_matrix

class Node:
    def __init__(self,class_label1):
        self.parent = None
        self.children = []
        self.class_label = class_label1
        self.data = None

    def depth(self):
        if len(self.children) == 0 :
            return 1
        else :
            max_dep = 0
            for node1 in self.children:
                depth = node1.depth()
                if depth > max_dep:
                    max_dep = depth
            return max_dep + 1

class BinaryTree(Node):
    def __init__(self):
        self.root = Node(-1)
        self.node_dict = {}

    def addNode(self,num1,num2):
        if num2 not in self.node_dict :
            self.node_dict[num2] = Node(num2)           
        
        if num1 not in self.node_dict :
            self.node_dict[num1] = Node(num1)           
        
        self.node_dict[num1].children.append(self.node_dict[num2])
        self.node_dict[num2].parent = self.node_dict[num1]
        

    def buildTree(self):     
        for node in self.node_dict.keys() :
            if self.node_dict[node].parent == None :
                self.root.children.append(self.node_dict[node])
    def bfs(self):
        queue = []
        visited = []
        queue.append(self.root)
        visited.append(self.root)
        while len(queue) > 0:
            current_node = queue.pop(0)                 
            for node in current_node.children:
                 if node not in visited :
                       queue.append(node)
                       sys.stdout.write(node.class_label+" ")
            print " "
    def depth(self, node):
        return self.root.depth()


    
tree = BinaryTree()
for line in open("hierarchy.txt") :
    arr = line.split(' ')
    tree.addNode(arr[0], arr[1])

tree.buildTree()
tree.root.data = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])

with open('my_dict.json', 'w') as f:
    json.dump(tree.node_dict, f)

print len(tree.root.children)
print tree.depth(tree.root)
#tree.bfs()

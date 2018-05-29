'''
Python script for doing hierarchical clustering on word embeddings and presenting results as dendrogram.
Input data format: a word file with each line containing a word, and an embedding file with each line containing the corresponding embedding vector. Each dimension of embedding separated by a tab.
'''

import sys
import pdb

from collections import OrderedDict

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

def load_data(word_file, emb_file):
	words = list(map(lambda l: l.strip(), open(word_file, 'r').readlines()))
	embs = np.array(list( map(lambda l: l.split('\t'), open(emb_file, 'r').readlines()) ))
	return words, embs


class TreeNode(object):
	"""
	If this is a leaf node, then type(self.val) == str
	Else, type(self.val) == tuple(left_idx, right_idx)
	"""
	def __init__(self, val):
		super(TreeNode, self).__init__()
		self.val = val
		

class BinRepr(object):
	"""docstring for BinRepr"""
	def __init__(self, words, embs):
		super(BinRepr, self).__init__()
		self.words = words
		self.embs = embs

		# agglomerative hierarchical clustering
		print('Agglomerative hierarchical clustering ...')
		Z = linkage(self.embs, 'ward')

		# build binary tree
		print('Building Tree ...')
		nodeList = list(map(TreeNode, self.words)) # leaf nodes
		for i in range(Z.shape[0]):
			nodeList.append(TreeNode( (int(Z[i,0]), int(Z[i,1])) ))

		# build binary representation
		self.bin = dict()

		def visit(v, s):
			if type(v.val) == str:
				self.bin[s] = v.val
			else:
				visit(nodeList[v.val[0]], s+'0')
				visit(nodeList[v.val[1]], s+'1')

		print('Computing binary representation ...')
		visit(nodeList[-1], '')
		print('Initialization finished.')

	def save_txt(self, fname):
		od = OrderedDict(sorted(self.bin.items()))
		f = open(fname,'w+')
		for k, v in od.iteritems():
			f.write('{}\t{}\n'.format(k, v))
		f.close


def main():
	words, embs = load_data(sys.argv[1], sys.argv[2])
	binRepr = BinRepr(words, embs)
	binRepr.save_txt(sys.argv[3])

if __name__ == '__main__':
	main()

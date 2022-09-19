"""
Class for connected components analysis.

Changelog:
  20200508: split from wadwrapper_lib.py
"""
import numpy as np
import scipy.ndimage as scind

class connectedComponents():
    def __init__(self):
        self.maskIn = None
        self.cca = None
        self.nb_labels = 0
        self.cluster_sizes = []

    def run(self,pBool):
        self.maskIn = pBool
        self.cca,self.nb_labels = scind.label(self.maskIn)
        self.cluster_sizes = []
        return self.cca,self.nb_labels

    def removeSmallClusters(self,minsize):
        mask_size = self.clusterSizes()<minsize
        self.cca[mask_size[self.cca]] = 0

        labels = np.unique(self.cca)
        self.nb_labels = len(labels)
        self.cca = np.searchsorted(labels, self.cca)
        self.cluster_sizes = []

    def clusterSizes(self):
        if len(self.cluster_sizes) == 0:
            self.cluster_sizes = scind.sum(self.maskIn, self.cca, range(self.nb_labels + 1))
        return self.cluster_sizes

    def indicesOfCluster(self,val):
        clus = np.where(self.cca == val)
        clus = [(x,y) for x,y in zip(clus[0],clus[1])]
        return clus

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        self.synthetic = np.zeros((self.n_samples * self.N, self.n_attrs), dtype=self.samples.dtype)
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in xrange(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape((1,-1)),return_distance=False)[0]  #Finds the K-neighbors of a point.
            self._populate(self.N,i,nnarray)
        return self.synthetic


    # for each minority class sample i ,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in xrange(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i] + gap*dif
            self.newindex+=1

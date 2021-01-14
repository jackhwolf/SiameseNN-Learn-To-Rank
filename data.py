import numpy as np 
from torch import FloatTensor
from torch.autograd import Variable
from itertools import combinations

def getdata(**kw):
    d = kw.get('data_name', 'sample_data')
    dm = {'sample_data': SampleData, 'beer_data': BeerData}
    return dm[d](**kw)

'''
dataset operations for experiments
'''
class Data:

    # our dataset has points, x_star, maybe l_star, and some other args
    def __init__(self, points, x_star, l_star, training_iterations, batch_size, data_name, **kw):
        self.points = points
        self.N = points.shape[0]
        self.D = points.shape[1]
        self.x_star = x_star
        self.l_star = l_star
        self.L = None if self.l_star is None else self.l_star.shape[1]
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        self.data_name = data_name
        self.ranks = None # build_ranks()

    # yield training data and ranks
    def training_iterator(self):
        combos = [(i,j) for i,j in combinations(range(self.N), 2)]
        combos = np.array(combos)
        K = combos.shape[0]
        for ti in range(self.training_iterations):
            for step in range(int(K / self.batch_size) + 1):
                combos_slice = combos[step*self.batch_size:(step+1)*self.batch_size,:]
                if len(combos_slice) == 0:
                    break  
                i, j = combos_slice[:,0], combos_slice[:,1]
                i, j = self.points[i,:], self.points[j,:]
                ranks = np.array([self.ranks[ik,jk] for ik,jk in combos_slice])
                yield i, j, ranks
            np.random.shuffle(combos)
            
    # yield prediction data and ranks
    def prediction_iterator(self):
        combos = [(i,j) for i,j in combinations(range(self.N), 2)]
        np.random.shuffle(combos)
        for i, j in combos:
            pi, pj, rij = self.points[i,:], self.points[j,:], self.ranks[i,j]
            yield pi, pj, rij

    def build_ranks(self):
        mat = np.zeros((self.N, self.N))
        for i, j in combinations(range(self.N), 2):
            di, dj, x = self.points[i,:], self.points[j,:], self.x_star
            if self.l_star is not None:
                di, dj, x = di @ self.l_star, dj @ self.l_star, x @ self.l_star
            d = self.distance(di, dj, x)
            mat[i,j] = d
        self.ranks = mat

    def distance(self, i, j, x):
        dist_i, dist_j = 0, 0
        if isinstance(i, np.ndarray):
            dist_i = np.linalg.norm(i - x)
            dist_j = np.linalg.norm(j - x)
        rank_ij = np.sign(dist_j - dist_i)
        return rank_ij

'''
sample data for experiments generated randomly in (N,D) space
'''
class SampleData(Data):

    def __init__(self, N=10, D=2, L=None, training_iterations=1, batch_size=1, **kw):
        points = np.random.uniform(-1, 1, size=(N, D))
        x_star = np.random.uniform(-1, 1, size=(1, D))
        l_star = None if L is None else np.random.normal(size=(D, L))
        super().__init__(points, x_star, l_star, training_iterations, batch_size, 'sample_data')
        self.build_ranks()


'''
beer data for experiments
'''
class BeerData(Data):

    def __init__(self, N=10, L=None, training_iterations=1, batch_size=1, **kw):
        points = np.load('Files/beer-data/beers_processed.npy')
        D = points.shape[1]
        mask = np.array([False] * points.shape[0])
        mask[:N] = True
        np.random.shuffle(mask)
        points = points[mask,:]
        x_star = np.random.uniform(-1, 1, size=(1, D))
        l_star = None if L is None else np.random.normal(size=(D, L))
        super().__init__(points, x_star, l_star, training_iterations, batch_size, 'beer_data')
        self.build_ranks()

if __name__ == "__main__":
    print(getdata().describe)
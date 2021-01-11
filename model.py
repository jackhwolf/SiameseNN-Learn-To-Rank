import torch
import numpy as np
from torch.autograd import Variable
from hingeloss import hingeloss

def getmodel(**kw):
    return RankLearner(**kw)

'''
siamese neural network for learning-to-rank
'''
class RankLearner:

    def __init__(self, D=2, L=None, criterion='MSELoss', lr=0.01, \
                            weight_decay=1e-5, optimizer='SGD', \
                            epochs=1000, **kw):
        self.D = D
        self.L = L
        self.criterion_t = criterion
        self.optimizer_t = optimizer
        self.x_hat = self.to_var(np.random.uniform(-1, 1, (1,D)), True)
        self.l_hat = None if self.L is None else self.to_var(np.random.normal(size=(self.D, self.L)), True)
        try:
            self.criterion = getattr(torch.nn, criterion)()
        except:
            cmap = {'Hinge': hingeloss()}
            self.criterion = cmap[criterion]
        self.lr = float(lr) 
        self.weight_decay = float(weight_decay)
        opt = getattr(torch.optim, optimizer)
        params = [self.x_hat]
        if self.L is not None:
            params.append(self.l_hat)
        self.optimizer = opt(params, lr=self.lr, weight_decay=self.weight_decay)
        self.epochs = epochs
        self.const_inp_x = torch.FloatTensor([1])
        self.const_inp_l = None if self.L is None else torch.FloatTensor(np.identity(self.D))

    def learn_pairwise_rank(self, point_i, point_j, true_rank_ij):
        true_rank_ij = self.to_var(true_rank_ij)
        for i in range(self.epochs):
            pred_rank_ij = self.forward(point_i, point_j)
            loss = self.criterion(pred_rank_ij, true_rank_ij).sum()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    # perform two forward passes and return prediction
    def forward(self, point_i, point_j):
        point_i, point_j = self.to_var(point_i), self.to_var(point_j)
        dist_i = self.forward_one(point_i)
        dist_j = self.forward_one(point_j)
        pred_rank_ij = (dist_j - dist_i).reshape(-1)
        return pred_rank_ij

    # single forward pass
    def forward_one(self, point):
        if self.L is None:
            # no linear transform: just compare point to x_hat
            self.const_inp_x.matmul(self.x_hat)
            return (point-self.x_hat).pow(2).sum() 
        else:
            # linear transform: just compare l_hat(point) to l_hat(x_hat)
            self.const_inp_x.matmul(self.x_hat)
            self.const_inp_l.matmul(self.l_hat)
            point_transform = point.matmul(self.l_hat)
            return (point_transform-self.x_hat.matmul(self.l_hat)).pow(2).sum() 

    # return sign of prediction for points without tracking gradients
    def predict_pairwise_rank(self, point_i, point_j):
        with torch.no_grad():
            pred = self.forward(point_i, point_j).sign()
        pred = pred.detach().numpy().item()
        return pred

    @property
    def current_x_hat(self):
        return self.x_hat.detach().numpy()

    @property
    def current_l_hat(self):
        if self.L is not None:
            return self.l_hat.detach().numpy()
        return None

    def to_var(self, foo, rg=False):
        return Variable(torch.FloatTensor(foo), requires_grad=rg)

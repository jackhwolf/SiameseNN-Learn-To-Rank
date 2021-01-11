import warnings
warnings.filterwarnings('ignore')
from data import getdata
from model import getmodel
import os
import json
import yaml
import numpy as np
from collections import OrderedDict
from distributed import worker_client
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

'''
manage running multiple experiments and saving results
'''
class ExperimentManager:
    
    def __init__(self, input_filename):
        with open(input_filename) as f:
            p = yaml.load(f, Loader=yaml.FullLoader)
            p = {k: [v] if not isinstance(v, list) else v for k, v in p.items()}
            p = list(ParameterGrid(p))
        self.parameters = p
        self.results_file = 'Results/results.json'

    # run all experiments defined in input_filename
    def __call__(self):
        with worker_client() as wc:
            futures = [wc.submit(Experiment(p)) for p in self.parameters]
            futures = wc.gather(futures)
        self.save_new_results(futures)
        return futures

    def save_new_results(self, new_res):
        if not os.path.exists(self.results_file):
            res = []
        else:
            with open(self.results_file, 'r') as fp:
                res = json.loads(fp.read())
        res.extend(new_res)
        with open(self.results_file, 'w') as fp:
            fp.write(json.dumps(res))
  
'''
manage running a single experiment given parameters
'''
class Experiment:

    def __init__(self, parameters):
        for k in parameters:
            if parameters[k] == 'None':
                parameters[k] = None
        self.params_ = parameters
        self.data = getdata(**parameters)
        if not parameters.get('D'):
            self.model = getmodel(D=self.data.D, **parameters)
        else:
            self.model = getmodel(**parameters)
        self.prediction_accuracy = -1.0

    # run experiment and return description
    def __call__(self):
        for (pi, pj, rij) in self.data.training_iterator():
            self.model.learn_pairwise_rank(pi, pj, rij)
        preds = []
        true = []
        for (pi, pj, rij) in self.data.prediction_iterator():
            preds.append(self.model.predict_pairwise_rank(pi, pj))
            true.append(rij)
        preds, true = np.array(preds), np.array(true)
        self.prediction_accuracy = accuracy_score(true, preds)
        return self.describe

    @property
    def describe(self):
        out = {}
        out.update(vars(self.data))
        out.update(vars(self.model))
        for key in ['points', 'x_star', 'l_star', 'ranks']:
            if out.get(key) is not None:
                out[key] = out[key].tolist()
        for key in ['x_hat_fc', 'l_hat_fc']:
            if out.get(key) is not None:
                out[key] = out[key].detach().numpy().tolist()
        for key in ['criterion', 'optimizer', 'const_inp_x', 'const_inp_l']:
            del out[key]
        for key in ['x_hat', 'l_hat']:
            if out.get(key) is not None:
                out[key] = out[key].detach().numpy().tolist()
        out['prediction_accuracy'] = self.prediction_accuracy
        out['experiment_name'] = self.params_['experiment_name']
        return OrderedDict(out) 
        
if __name__ == "__main__":
    import sys
    import time
    from distributed import Client

    input_filename = sys.argv[1]
    dask_scheduler_address = sys.argv[2]
    
    client = Client('tcp://' + dask_scheduler_address)
    client.upload_file('hingeloss.py')
    client.upload_file('model.py')
    client.upload_file('data.py')
    client.upload_file('experiment.py')

    future = client.submit(ExperimentManager(input_filename))
    future = client.gather(future)
    print(future)
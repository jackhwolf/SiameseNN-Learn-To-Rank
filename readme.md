## implementation of siamese neural network for Learning-To-Rank

Blake Mason, Jack Wolf

### How to run:
- make an input file similar to `Files/input_files/beer_complete_sample.yaml`
- create a virtual environment with the packages from `requirements.txt`
- make sure the dask scheduler address of the server you are working on is 
in the `dask_addr_map` in `dask_deployment.py`
- run the command `(venv) $ python3 dask_deployment.py input_filename n_workers`
- results are stored in `Results/results.json`
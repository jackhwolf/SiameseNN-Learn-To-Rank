import libtmux 
import socket
import os

# map server hostnames to dask scheduler addresses
dask_addr_map = {
    "SkippyElvis": "192.168.86.83:8786",
    "opt-a003.discovery.wisc.edu": "144.92.142.183:8786",
    "opt-a004.discovery.wisc.edu": "144.92.142.184:8786"
}

# command to activate virtual environment
venv_cmd = 'source venv/bin/activate'

'''
handles setting up tmux workspace to run experiments in
this involves three tmux windows, one for each
    - dask scheduler
    - dask workers
    - python script
'''
class DaskDeployer:

    def __init__(self, input_filename, n_workers, cloud=False):
        self.input_filename = input_filename
        self.n_workers = int(n_workers)
        self.hostname = socket.gethostname()
        self.addr = dask_addr_map[self.hostname]
        assert self.n_workers > 1, "need at least 1 worker"
        self.session_name = 'beerspace-experiments'
        self.server = libtmux.Server()
        self.cloud = cloud

    # turn on scheduler, make workers, submit script
    def __call__(self):
        os.system('tmux new -s blank -d')
        scheduler, workers, script = self._make_windows()
        scheduler.attached_pane.send_keys(self._scheduler_command)
        worker_pane = None
        for nw in range(self.n_workers):
            if nw == 0:
                worker_pane = workers[nw//4].attached_pane
            else:
                worker_pane = workers[nw//4].split_window(vertical=False, attach=False)
            worker_pane.send_keys(self._worker_command)
        script.attached_pane.send_keys(self._script_command)
        for i, _ in enumerate(workers):
            workers[i].select_layout('tiled')
        os.system('tmux kill-session -t blank')

    def _make_windows(self):
        session = self.server.new_session(self.session_name, kill_session=True)
        scheduler = session.attached_window
        scheduler.rename_window('scheduler')
        worker_windows = (self.n_workers // 4) + 1
        workers = []
        for i in range(worker_windows):
            worker = session.new_window(f'workers{i}', attach=False)
            workers.append(worker)
        script = session.new_window('script', attach=False)
        return (scheduler, workers, script)

    @property
    def _scheduler_command(self):
        return f'{venv_cmd}; dask-scheduler'

    @property
    def _worker_command(self):
        return f'{venv_cmd}; export OMP_NUM_THREADS=3; dask-worker {self.addr}'

    @property
    def _script_command(self):
        return f'{venv_cmd}; python3 experiment.py {self.input_filename} {self.addr} --cloud={self.cloud}'

if __name__ == "__main__":
    import sys
    from deployment_argparser import deployment_parser
    args = deployment_parser()
    dd = DaskDeployer(**args)
    dd()
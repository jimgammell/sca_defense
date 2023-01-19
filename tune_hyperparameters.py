import os
from copy import deepcopy
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import air
from run_trial import run_trial
from ray.tune import CLIReporter
from ray.tune.search.bayesopt import BayesOptSearch

def flatten_config(config):
    is_flat = True
    for skey, sitem in deepcopy(config).items():
        if type(sitem) == dict:
            is_flat = False
            for nkey, nitem in sitem.items():
                config['&'.join((skey, nkey))] = nitem
            del config[skey]
    if is_flat:
        return config
    else:
        return flatten_config(config)

def unflatten_config(config):
    is_unflat = True
    for fkey, fitem in deepcopy(config).items():
        if '&' in fkey:
            is_unflat = False
            skey = '&'.join(fkey.split('&')[:-1])
            nkey = fkey.split('&')[-1]
            if skey in config.keys():
                config[skey].update({nkey: fitem})
            else:
                config[skey] = {nkey: fitem}
            del config[fkey]
    if is_unflat:
        return config
    else:
        return unflatten_config(config)

    # Todo: 25 samples, 20 epochs/sample
def tune_hyperparameters(*args, params_to_tune={}, **kwargs):
    params_to_tune = flatten_config(params_to_tune)
    tune_config = {}
    for key, item in params_to_tune.items():
        [search_space_name, search_space_args] = item
        search_space_constructor = getattr(tune, search_space_name)
        search_space = search_space_constructor(*search_space_args)
        tune_config[key] = search_space
    scheduler = ASHAScheduler(
        metric='mean_rank',
        mode='min',
        max_t = 50,
        grace_period=5,
        reduction_factor=2)
    search_algorithm = BayesOptSearch(
        points_to_evaluate=[{
            'disc_opt_kwargs&lr': 1e-3,
            'disc_opt_kwargs&beta1': 0.9,
            'disc_opt_kwargs&weight_decay': 0.0,
            'disc_kwargs&dense_dropout': 0.1}])
    
    class ExperimentTerminationReporter(CLIReporter):
        def should_report(self, trials, done=False):
            return done
        
    trials_per_gpu = 3 #maximum number of trials which can fit simultaneously in GPU memory
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run_trial_with_raytune, args=args, kwargs=kwargs, working_dir=os.getcwd()),
            resources={'gpu': 1.0/trials_per_gpu}
        ),
        tune_config=tune.TuneConfig(
            metric='mean_rank',
            mode='min',
            scheduler=scheduler,
            #search_alg=search_algorithm,
            num_samples=96,
            max_concurrent_trials=torch.cuda.device_count()*trials_per_gpu
        ),
        param_space=tune_config,
        run_config=air.RunConfig(local_dir=None if 'save_dir' not in kwargs.keys()
                                 else os.path.join('.', 'results', kwargs['save_dir']),
                                 progress_reporter=ExperimentTerminationReporter())
    )
    results = tuner.fit()
    best_results = results.get_best_result('mean_rank', mode='min')
    print('Best trial config: {}'.format(best_results.config))
    print('Best trial metrics: {}'.format(best_results.metrics))
    
def run_trial_with_raytune(hparam_config, args=[], kwargs={}, working_dir=None):
    os.chdir(working_dir)
    kwargs = flatten_config(kwargs)
    kwargs.update(hparam_config)
    kwargs = unflatten_config(kwargs)
    def repackage_betas(d):
        if 'beta1' in d.keys() or 'beta2' in d.keys():
            betas = [
                d['beta1'] if 'beta1' in d.keys() else 0.9,
                d['beta2'] if 'beta2' in d.keys() else 0.999
            ]
            d['betas'] = betas
        if 'beta1' in d.keys():
            del d['beta1']
        if 'beta2' in d.keys():
            del d['beta2']
        return d
    if 'disc_opt_kwargs' in kwargs.keys():
        kwargs['disc_opt_kwargs'] = repackage_betas(kwargs['disc_opt_kwargs'])
    if 'gen_opt_kwargs' in kwargs.keys():
        kwargs['gen_opt_kwargs'] = repackage_betas(kwargs['gen_opt_kwargs'])
    run_trial(*args, using_raytune=True, **kwargs)
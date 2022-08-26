import os
import argparse
import datetime

from parse_config import parse_config
from utils import get_print_to_log, specify_log_path
print = get_print_to_log(__file__)
import trial

def get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir',
                        dest='config_dir',
                        default=os.path.join('.', 'config'),
                        help='Specify the directory in which to look for configuration files. If unspecified, it will default to \'./config\'.')
    parser.add_argument('--config',
                        dest='config_files',
                        nargs='+',
                        default=None,
                        help='Specify which configuration files to use. If unspecified, the program will run all files in the configuration directory.')
    parser.add_argument('--results-dir',
                        dest='results_dir',
                        default=os.path.join('.', 'results'),
                        help='Specify the directory in which to store results. If unspecified, it will default to \'./results\'.')
    parser.add_argument('--datasets-dir',
                        dest='datasets_dir',
                        default=os.path.join('.', 'saved_datasets'),
                        help='Specify the directory in which to look for datasets. If unspecified, it will default to \'./saved_datasets\'.')
    parser.add_argument('--debug',
                        dest='debug',
                        default=False,
                        action='store_true',
                        help='Run each trial for the minimum amount of time that is reasonable (i.e. train for 1 epoch), for debugging purposes.')
    args = parser.parse_args()
    if args.config == None:
        args.config = [f for f in os.listdir(args.config_dir) if 'json' in f.split('.')]
    return args

def main():
    dt = datetime.datetime.now()
    cl_args = get_command_line_arguments()
    
    print('Beginning trials...')
    print('\tConfiguration directory: {}'.format(cl_args.config_dir))
    print('\tResults directory: {}'.format(cl_args.results_dir))
    print('\tDatasets directory: {}'.format(cl_args.datasets_dir))
    print('\tConfig files to run: {}'.format(', '.join(cl_args.config_files)))
    print('\tDebug mode: {}'.format(cl_args.debug))
    if not os.path.exists(os.path.join(cl_args.results_dir, 'execution_logs')):
        os.mkdir(os.path.join(cl_args.results_dir, 'execution_logs'))
    specify_log_path(os.mkdir(os.path.join(cl_args.results_dir, 'execution_logs', 'log__%d_%d_%d_%d_%d_%d.txt'%(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))), save_buffer=True)
    
    for config_file in cl_args.config_files:
        print()
        print()
        if config_file.split('.')[-1] != 'json':
            config_file = '.'.join((config_file, 'json'))
        print('Running trial specified in {}...'.format(os.path.join(cl_args.config_dir, config_file)))
        results_dir = os.path.join(cl_args.results_dir, config_file.split('.')[0])
        specify_log_path(os.path.join(results_dir, 'log.txt'), save_buffer=True)
        print('\tResults directory: {}'.format(results_dir))
        if os.path.exists(results_dir):
            os.rename(results_dir, results_dir + '__%d_%d_%d_%d_%d_%d'%(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
        os.mkdir(results_dir)
        expanded_config_files, expanded_keys = parse_config(os.path.join(cl_args.config_fir, config_file))
        print('\tTrial description: {}'.format(expanded_config_files[0]['trial_description']))
        print('\tConfig file expanded into %d trials'%(len(expanded_config_files)))
        print('\t\tExpanded keys: {}'.format(', '.join(expanded_keys)))
        for trial_idx, expanded_config_file in enumerate(expanded_config_files):
            specify_log_path(None)
            print('Running trial %d'%(trial_idx+1))
            print('Expanded elements:')
            for key in expanded_keys:
                print('\t{}: {}'.format(key, expanded_config_file[key]))
            trial_dir = os.path.join(results_dir, 'trial_%d'%(trial_idx))
            print('Saving results in {}'.format(trial_dir))
            specify_log_path(os.path.join(trial_dir, 'log.txt'), save_buffer=True)
            trial = getattr(trials, expanded_config_file['trial'])
            results = trial.main(debug=cl_args.debug, **expanded_config_file['trial_kwargs'])
            trial.save_results(results, trial_dir)

if __name__ == '__main__':
    main()
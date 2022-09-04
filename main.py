import os
import argparse
import datetime
import json

from parse_config import parse_config
from utils import get_print_to_log, specify_log_path, get_package_module_names, get_package_modules, get_filename
print = get_print_to_log(get_filename(__file__))
import trials
import generate_figures

def get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir',
                        dest='config_dir',
                        default=os.path.join('.', 'config'),
                        help='Specify the directory in which to look for configuration files. If unspecified, it will default to \'./config\'.')
    parser.add_argument('--config',
                        dest='config_files',
                        nargs='*',
                        default=None,
                        help='Specify which configuration files to use. If the argument is not passed, the program will not run any trials. If the argument is passed but not followed by any configuration files, all configuration files in the configuration directory will be used.')
    parser.add_argument('--results-dir',
                        dest='results_dir',
                        default=os.path.join('.', 'results'),
                        help='Specify the directory in which to store results. If unspecified, it will default to \'./results\'.')
    parser.add_argument('--datasets-dir',
                        dest='datasets_dir',
                        default=os.path.join('.', 'saved_datasets'),
                        help='Specify the directory in which to look for datasets. If unspecified, it will default to \'./saved_datasets\'.')
    parser.add_argument('--generate-figures',
                        dest='generate_figures',
                        nargs='*',
                        default=False,
                        help='Generate figures for the results in the specified directories. If the argument is passed but no directories are specified, figures will be generated for the newest folder in the results directory, including results generated during the current execution of this program if applicable.')
    parser.add_argument('--debug',
                        dest='debug',
                        default=False,
                        action='store_true',
                        help='Run each trial for the minimum amount of time that is reasonable (i.e. train for 1 epoch), for debugging purposes.')
    parser.add_argument('--cpu',
                        dest='cpu',
                        default=False,
                        action='store_true',
                        help='Set the device to \'cpu\' for all trials, ignoring settings in their config files.')
    args = parser.parse_args()
    if args.config_files == None:
        args.config_files = []
    elif args.config_files == []:
        args.config_files = [f for f in os.listdir(args.config_dir) if 'json' in f.split('.')]
    if args.generate_figures == []:
        most_recent_f = max([f for f in os.listdir(args.results_dir)
                             if os.path.isdir(os.path.join(args.results_dir, f)) and f != 'execution_logs'],
                            key=lambda f: os.path.getmtime(os.path.join(args.results_dir, f)))
        args.generate_figures = [os.path.join(args.results_dir, most_recent_f)]
    return args

def main():
    dt = datetime.datetime.now()
    cl_args = get_command_line_arguments()
    if not os.path.exists(cl_args.results_dir):
        os.mkdir(cl_args.results_dir)
    
    print('Beginning trials...')
    print('\tConfiguration directory: {}'.format(cl_args.config_dir))
    print('\tResults directory: {}'.format(cl_args.results_dir))
    print('\tDatasets directory: {}'.format(cl_args.datasets_dir))
    print('\tConfig files to run: {}'.format(', '.join(cl_args.config_files)))
    print('\tDebug mode: {}'.format(cl_args.debug))
    if not os.path.exists(os.path.join('.', 'results', 'execution_logs')):
        os.mkdir(os.path.join('.', 'results', 'execution_logs'))
    specify_log_path(os.path.join('.', 'results', 'execution_logs', 'log__%d_%d_%d_%d_%d_%d.txt'%(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)), save_buffer=True)
    
    for config_file in cl_args.config_files:
        specify_log_path(None)
        print()
        print()
        if config_file.split('.')[-1] != 'json':
            config_file = '.'.join((config_file, 'json'))
        print('Running trial specified in {}...'.format(os.path.join(cl_args.config_dir, config_file)))
        results_dir = os.path.join(cl_args.results_dir, config_file.split('.')[0])
        print('\tResults directory: {}'.format(results_dir))
        if os.path.exists(results_dir):
            os.rename(results_dir, results_dir + '__%d_%d_%d_%d_%d_%d'%(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
        os.mkdir(results_dir)
        specify_log_path(os.path.join(results_dir, 'log.txt'), save_buffer=True)
        expanded_config_files, expanded_keys = parse_config(os.path.join(cl_args.config_dir, config_file))
        print('\tTrial description: {}'.format(expanded_config_files[0]['trial_description']))
        print('\tConfig file expanded into %d trials'%(len(expanded_config_files)))
        print('\t\tExpanded keys: {}'.format(', '.join(expanded_keys)))
        for trial_idx, expanded_config_file in enumerate(expanded_config_files):
            print()
            specify_log_path(None)
            print('Running trial %d'%(trial_idx))
            trial_dir = os.path.join(results_dir, 'trial_%d'%(trial_idx))
            os.mkdir(trial_dir)
            print('Saving results in {}'.format(trial_dir))
            specify_log_path(os.path.join(trial_dir, 'log.txt'), save_buffer=True)
            trial = get_package_modules(trials)[
                get_package_module_names(trials)[0].index(expanded_config_file['trial'])]
            expanded_config_file = trial.preprocess_config(expanded_config_file)
            if cl_args.cpu:
                expanded_config_file['device'] = 'cpu'
            results = trial.main(debug=cl_args.debug, **expanded_config_file)
            trial.save_results(results, trial_dir)
            with open(os.path.join(trial_dir, 'config.json'), 'w') as F:
                json.dump(expanded_config_file, F, indent=2)
     
    if cl_args.generate_figures is not False:
        specify_log_path(os.path.join('.', 'results', 'execution_logs', 'log__%d_%d_%d_%d_%d_%d.txt'%(
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)), save_buffer=True)
        for folder in cl_args.generate_figures:
            subfolders = [os.path.join(folder, f) for f in os.listdir(folder)
                          if os.path.isdir(os.path.join(folder, f))]
            print()
            print('Generating figures for results in {}...'.format(folder))
            for subfolder in subfolders:
                print('\tGenerating figures for results in {}...'.format(subfolder))
                generate_figures.main(subfolder)

if __name__ == '__main__':
    main()
import concurrent.futures
import itertools
import logging
import logging.handlers
import glob
import os
import time

import click
from tqdm import tqdm
import yaml
import psutil


@click.group()
def cli():
    pass



@cli.command(help="Run decode signal a certain number of times.")
@click.argument('config_file', type=click.File('r'))
@click.argument('num', type=click.INT)
def run(config_file, num):
    import decode_signal

    config = yaml.load(config_file)
    now = int(time.time())
    test_path = os.path.join('tests', str(now))

    # Make directory for experiment
    os.mkdir(test_path)

    # Save parameters in folder
    with open(os.path.join(test_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Run experiments
    with tqdm(total=num) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            ids = range(1, num + 1)
            folders = itertools.repeat(os.path.join('tests', str(now)))
            params = itertools.repeat(config)

            for id_, result in zip(ids, executor.map(decode_signal.main, ids, folders, params)):
                pbar.update()


def successful_experiment(file_name):
    file = open(file_name)
    lines = (line for line in file)
    lines = ('Result: True' in line for line in lines)

    return any(lines)


def analyze_experiment(experiment_folder):
    files = glob.glob(os.path.join(experiment_folder, '*.log'))

    with open(os.path.join(experiment_folder, 'config.yaml')) as f:
        config = yaml.load(f)

    successful = sum(successful_experiment(f) for f in files)
    total = len(files)

    print("{} ({}): {}/{} = {:.2%}".format(experiment_folder, config, successful, total, successful / total))


@cli.command(help="Analyze results.")
def analyze():
    # Get all experiments in test directory
    directory = 'tests'
    experiments = [o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]

    for e in sorted(experiments):
        analyze_experiment(os.path.join(directory, e))


if __name__ == '__main__':
    cli()

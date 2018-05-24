import concurrent.futures
import copy
import itertools
import logging
import logging.handlers
import glob
from operator import itemgetter
import os
import time

import click
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
import psutil



def _run(config, num, executor):
    import decode_signal

    now = int(time.time())
    test_path = os.path.join('tests', str(now))

    # Make directory for experiment
    os.mkdir(test_path)

    # Save parameters in folder
    with open(os.path.join(test_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Run experiments
    with tqdm(total=num) as pbar:
        ids = range(1, num + 1)
        folders = itertools.repeat(os.path.join('tests', str(now)))
        params = itertools.repeat(config)

        results = {}
        for id_, result in zip(ids, executor.map(decode_signal.main, ids, folders, params)):
            results[id_] = result
            pbar.update()

    with open(os.path.join(test_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def split_num_list(ctx, param, value):
    if value is None or value == '':
        return [None]
    try:
        try:
            return [int(x) for x in value.split(',') if x]
        except ValueError:
            return [float(x) for x in value.split(',') if x]
    except ValueError:
        raise click.BadParameter('List must only contain numbers')


def analyze_experiment(experiment_folder):
    files = glob.glob(os.path.join(experiment_folder, '*.log'))

    with open(os.path.join(experiment_folder, 'config.yaml')) as f:
        config = yaml.load(f)

    with open(os.path.join(experiment_folder, 'results.json')) as f:
        results = json.load(f)
    successful = sum(1 for value in results.values() if value)
    total = len(results.values())

    return {"folder": experiment_folder,
            "config": config,
            "successful": successful,
            "total": total}


def check_config(config):
    if 'seed' in config and not click.confirm('Seed is set in the configuration file. Are you sure about that?'):
        exit()


@click.group()
def cli():
    pass


@cli.command(help="Run decode signal a certain number of times.")
@click.argument('config_file', type=click.File('r'))
@click.argument('num', type=click.INT)
def run_simulate(config_file, num):
    config = yaml.load(config_file)
    check_config(config)
    with concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
        _run(config, num, executor)


@cli.command(help="Run a batch of decode signal a certain number of times.")
@click.argument('config_file', type=click.File('r'))
@click.argument('num', type=click.INT)
@click.option('--max_len_seq', callback=split_num_list)
@click.option('--signal', callback=split_num_list)
def batch(config_file, num, max_len_seq, signal):
    config = yaml.load(config_file)
    check_config(config)

    with concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
        for mls, sig in itertools.product(max_len_seq, signal):
            if mls is not None:
                config['max_len_seq'] = mls

            if sig is not None:
                config['signal'] = sig

            print("Running with {} and {}".format(mls, sig))
            _run(config, num, executor)


@cli.command(help="Analyze results.")
@click.option('--graph/--no-graph', default=True)
def analyze(graph):
    # Get all experiments in test directory
    directory = 'tests'
    experiments = [o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]

    results = []
    for e in sorted(experiments):
        result = analyze_experiment(os.path.join(directory, e))
        results.append(result)
        print(f"{result['folder']} ({result['config']}): {result['successful']}/{result['total']} = {result['successful']/result['total']:.2%}")

    if not graph:
        return

    data = ((r['config']['signal'], r['config']['max_len_seq'], (r['successful'] / r['total']) * 100)
            for r in results)

    data = sorted(data)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    for z, rest in itertools.groupby(data, lambda x: x[0]):
        _, xs, ys = zip(*rest)

        ax.plot(xs, ys, label=z, marker='.')

    ax.set_xlabel("Symbol Size ($2^x-1$)")
    plt.xticks(xs, xs)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Noise ($\mu={results[0]['config']['noise']['mu']}$, $\sigma={results[0]['config']['noise']['sigma']}$)")

    plt.legend()
    plt.tight_layout()
    plt.savefig('batch.pdf')


def get_consecutive_number_groups(lst):
    for k, g in itertools.groupby(enumerate(lst), lambda x: x[0]-x[1]):
        yield list(map(itemgetter(1), g))

def create_graph(result, location):
    samples = result.samples
    sample_period = result.sample_period
    detected_signal = result.detected_signal
    correlation = result.correlation
    correlation_threshold_high = result.correlation_threshold_high

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8,4))

    # Plot the raw samples
    ax1.plot(np.arange(len(samples)) * sample_period, samples)
    ax1.set_xlabel('Time (s)')

    scatter_data = [(x, y) for x, y, event in detected_signal if event == 'detected_peak']
    if scatter_data:
        x, y = zip(*scatter_data)
        ax3.scatter(x * sample_period, y,
                    marker='x',
                    color='yellow')


    ax3.plot(np.arange(len(correlation_threshold_high)) * sample_period, correlation_threshold_high, color='green', label='upper threshold')
    # ax3.plot(correlation_threshold_low, color='orange', label='lower threshold')
    ax3.plot(np.arange(len(correlation)) * sample_period, correlation, label='correlation')

    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xlabel('Time (s)')

    # plt.legend()
    plt.tight_layout()

    name = f"{result.metadata['distance']}-{result.metadata['location']}-{result.metadata['experiment_number']}"
    # plt.savefig(os.path.join(location, f'{name}.pdf'))
    plt.savefig(os.path.join(location, f'{name}.png'), dpi=600)


def get_metadata(file):
    with open(file) as f:
        return json.load(f)


@cli.command(help="Run ONPC on collected data different parameters")
@click.argument('config_file', type=click.File('r'))
@click.option('-d', '--data', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
@click.option('--low_pass_filter_size', callback=split_num_list)
@click.option('--correlation_buffer_size', callback=split_num_list)
@click.option('--correlation_std_threshold', callback=split_num_list)
@click.option('--graph/--no-graph')
def run_data(config_file, data, folder, low_pass_filter_size, correlation_buffer_size,
             correlation_std_threshold, graph):
    import decode_signal

    data_files = itertools.chain(data, *[glob.glob(os.path.join(f, '*.json')) for f in folder])
    data_files = sorted(set(data_files))

    params = (data_files, low_pass_filter_size, correlation_buffer_size, correlation_std_threshold)
    # params = [x for x in params if x is not None and x[0] is not None]
    param_combinations = list(itertools.product(*params))

    config = yaml.load(config_file)
    results = []

    with tqdm(total=len(param_combinations)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            future_to_param = {}
            for index, param in enumerate(param_combinations):
                current_config = copy.deepcopy(config)

                if param[0] is not None:
                    current_config['sample_file']['name'] = param[0]
                    current_config['sample_file']['type'] = 'wl'

                if param[1] is not None:
                    current_config['low_pass_filter_size'] = param[1]

                if param[2] is not None:
                    current_config['correlation_buffer_size'] = param[2]

                if param[3] is not None:
                    current_config['correlation_std_threshold'] = param[3]

                f = executor.submit(decode_signal.main, index, None, current_config)
                future_to_param[f] = param

            for future in concurrent.futures.as_completed(future_to_param):
                file_name = param[0]
                metadata = get_metadata(file_name)

                param = future_to_param[future]
                result = future.result()
                result.param = param
                result.metadata = metadata

                results.append(result)
                pbar.update()

    print()

    # # Group by location, description, experiment number
    # sorted_location_results = sorted(results, lambda x: (x.config['location'],
    #                                                      x.config['description'],
    #                                                      x.config['experiment_number'])

    # for result in sorted_location_results:
    #     print(result.config['location'])


    for result in results:
        # print(result)

        if graph:
            if not os.path.exists('graphs'):
                os.makedirs('graphs')
            create_graph(result, 'graphs')

        detected_signal_index = [i for i, _, _ in result.detected_signal]
        groups = list(get_consecutive_number_groups(detected_signal_index))

        groups_first_value = np.array([g[0] for g in groups])
        diffs_between_groups = np.diff(groups_first_value)

        print('-' * 80)
        print(f"Distance: {result.metadata['distance']}")
        print(f"Location: {result.metadata['location']}")
        print(f"Experiment #: {result.metadata['experiment_number']}")
        print(f"Description: {result.metadata['description']}")
        print()

        for i, group in enumerate(groups):
            print(group)
            if i < len(diffs_between_groups):
                print(f'  |')
                print(f'  | {diffs_between_groups[i]}')
                print(f'  |')
                print(f'  v')

        print()
        print('-' * 80)

if __name__ == '__main__':
    cli()

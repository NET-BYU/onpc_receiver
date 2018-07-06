import binascii
import concurrent.futures
import copy
import functools
import hashlib
import io
import itertools
import logging
import logging.handlers
import glob
from operator import itemgetter
import os
import pickle
import time

import click
from glom import glom
from jinja2 import FileSystemLoader, Environment
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from pint import UnitRegistry
from tqdm import tqdm
import yaml
import psutil


ureg = UnitRegistry()


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


def get_consecutive_number_groups(lst, tolerence=20):
    groups = itertools.groupby(enumerate(lst), lambda x: x[0]-x[1])
    prev = list(map(itemgetter(1), next(groups)[1]))

    for k, g in groups:
        current = list(map(itemgetter(1), g))

        if current[0] - prev[-1] < tolerence:
            prev.extend(current)
            continue

        yield prev
        prev = current

    yield prev

def create_graph(result, location):
    original_samples = result.original_result.samples
    samples = result.limited_result.samples
    sample_period = result.sample_period
    detected_signal = result.limited_result.detected_signal
    correlation = result.limited_result.correlation
    correlation_threshold_high = result.limited_result.threshold

    fig, (ax0, ax1, ax3) = plt.subplots(3, 1, figsize=(8,4), sharex=True)

    ax0.plot(np.arange(len(original_samples)) * sample_period, original_samples, '.', markersize=.7)
    ax0.set_xlabel('Time (s)')

    # Plot the raw samples
    ax1.plot(np.arange(len(samples)) * sample_period, samples, '.', markersize=.7)

    if detected_signal:
        x, y = zip(*detected_signal)
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

    if isinstance(location, str):
        name = f"{result.metadata['distance']}-{result.metadata['location']}-{result.metadata['experiment_number']}"
        # plt.savefig(os.path.join(location, f'{name}.pdf'))
        plt.savefig(os.path.join(location, f'{name}.png'), dpi=300)
    else:
        # Treat location as file
        plt.savefig(location, format='png', dpi=300)

    plt.close(fig)


def generate_graph(result):
    graph_data = io.BytesIO()
    graph = create_graph(result, graph_data)

    data = binascii.b2a_base64(graph_data.getvalue(), newline=False)
    return f"data:image/png;base64,{data.decode()}"


def get_unique_id(result):
    return result.name


def get_details(result):
    return {'Run #': result.metadata['experiment_number'],
            'Transmitting': result.metadata.get('transmitting', True),
            'Run time': result.metadata['run_time'],
            'Filename': result.name}


def get_symbol_groups(result):
    detected_signal_index = [x for x, y in result.limited_result.detected_signal]
    groups = list(get_consecutive_number_groups(detected_signal_index))

    new_groups = []


    groups_first_value = np.array([g[0] for g in groups])
    diffs_between_groups = np.diff(groups_first_value)

    return groups, diffs_between_groups


def get_symbol_summary(result):
    groups, diffs_between_groups = get_symbol_groups(result)
    str_out = io.StringIO()

    for i, group in enumerate(groups):
        str_out.write(f'{group}\n')
        if i < len(diffs_between_groups):
            str_out.write(f'  |\n')
            str_out.write(f'  | {diffs_between_groups[i]}\n')
            str_out.write(f'  |\n')
            str_out.write(f'  v\n')

    return str_out.getvalue()


def get_result_score(result):
    groups, diffs_between_groups = get_symbol_groups(result)

    run_time = result.metadata['run_time']
    chip_time = ureg(result.metadata['chip_time']).magnitude / 1e3
    symbol_size = result.metadata.get('symbol_size', 1023)
    symbol_time = chip_time * symbol_size

    expected_received_symbols = int((run_time // symbol_time) - 1)
    # print(run_time, symbol_time, expected_received_symbols)
    # print(groups)
    # print(diffs_between_groups)
    # print(input.sample_period)
    # print(symbol_time)
    # print(symbol_time / input.sample_period.magnitude)
    # exit()

    if not result.metadata.get('transmitting', True):
        return {"Total": 0,
                "Correct": 0,
                "False positive": len(groups)}

    false_positive = 0
    correct = 0

    if len(groups) > 0:
        # Assume that the first symbol is correct
        correct += 1

    for diff in diffs_between_groups:
        time_diff = diff * result.sample_period.magnitude  # Convert from sample number diffs to time diffs
        num_symbols = time_diff / symbol_time  # Number of symbols between groups
        offset = abs(round(num_symbols) - num_symbols)
        # print(f"Number of symbols: {num_symbols} ({offset})")

        # print(diff, time_diff, offset)

        # If the offset is far enough away, then it is a false positive
        if time_diff < 10 or offset > .2:
            false_positive += 1
        else:
            correct += 1

    return {"Total": expected_received_symbols,
            "Correct": correct,
            "False positive": false_positive}


def get_all_results_score(results):
    result_scores = [get_result_score(result) for result in results]

    return glom(result_scores, {'Total': (['Total'], sum),
                                'Correct': (['Correct'], sum),
                                'False positive': (['False positive'], sum)})


@cli.command(help="Run ONPC on collected data different parameters")
@click.option('-d', '--data-file', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
@click.option('--lpf-size', default=30)
@click.option('--threshold-size', default=600)
@click.option('--threshold-lag', default=100)
@click.option('--threshold-std', default=4.5)
@click.option('--webpage/--no-webpage')
def run_onpc(data_file, folder, lpf_size, threshold_size, threshold_lag,
             threshold_std, webpage):
    import onpc_v2

    data_files = itertools.chain(data_file,
        *[glob.glob(os.path.join(f, '*.json')) for f in folder])

    # Remove possible duplicates
    data_files = sorted(set(data_files))

    results = []

    with tqdm(total=len(data_files)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            futures = []
            for file in data_files:
                f = executor.submit(onpc_v2.run,
                                    file,
                                    lpf_size=lpf_size,
                                    threshold_size=threshold_size,
                                    threshold_lag=threshold_lag,
                                    threshold_std=threshold_std,
                                    graph=False,
                                    interactive=False)
                futures.append(f)

            results = []
            for future in concurrent.futures.as_completed(futures):
                pbar.update()
                results.append(future.result())

    sorted_location_results = sorted(results, key=lambda x: (x.metadata['location'],
                                                             x.metadata['description'],
                                                             x.metadata['experiment_number']))

    if webpage:
        env = Environment(loader=FileSystemLoader('templates'))
        env.filters['generate_graph'] = generate_graph
        env.filters['get_details'] = get_details
        env.filters['get_symbol_summary'] = get_symbol_summary
        env.filters['get_unique_id'] = get_unique_id
        env.filters['get_result_score'] = get_result_score
        env.filters['get_all_results_score'] = get_all_results_score
        template = env.get_template('results.html')

        with open('onpc_results.html', 'w') as f:
            f.write(template.render(results=sorted_location_results))

    print(get_all_results_score(sorted_location_results))


if __name__ == '__main__':
    cli()

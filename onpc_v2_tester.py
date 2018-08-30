import binascii
import concurrent.futures
import copy
import functools
import hashlib
import io
import itertools
import json
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from pint import UnitRegistry
import psutil
from tqdm import tqdm
import yaml


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

def create_graph(experiment_result, location):
    sample_period = experiment_result.sample_period
    fig, axs = plt.subplots(len(experiment_result.results) * 2, 1, figsize=(8,6), sharex=True)

    for i, result in enumerate(experiment_result.results.values()):
        axs[i * 2].plot(np.arange(len(result.samples)) * sample_period,
             result.samples, '.', markersize=.7)

        axs[i * 2 + 1].plot(np.arange(len(result.threshold)) * sample_period, result.threshold, color='green', linewidth=1)
        axs[i * 2 + 1].plot(np.arange(len(result.correlation)) * sample_period, result.correlation, linewidth=1)
        axs[i * 2 + 1].set_ylabel(result.name.title())

        if result.detected_signal:
            xs, ys = zip(*result.detected_signal)
            axs[i * 2 + 1].scatter(xs * sample_period, ys,
                                   marker='x',
                                   color='yellow')

    axs[-1].set_xlabel('Time (s)')

    plt.tight_layout()

    if isinstance(location, str):
        name = f"{experiment_result.metadata['distance']}-{experiment_result.metadata['location']}-{experiment_result.metadata['experiment_number']}"
        # plt.savefig(os.path.join(location, f'{name}.pdf'))
        plt.savefig(os.path.join(location, f'{name}.png'), dpi=300)
    else:
        # Treat location as file
        plt.savefig(location, format='png', dpi=300)

    plt.close(fig)


# def create_graph(result, location):
#     original_samples = result.original_result.samples
#     samples = result.limited_result.samples
#     sample_period = result.sample_period
#     detected_signal = result.limited_result.detected_signal
#     correlation = result.limited_result.correlation
#     correlation_threshold_high = result.limited_result.threshold

#     fig, (ax0, ax1, ax3) = plt.subplots(3, 1, figsize=(8,4), sharex=True)

#     ax0.plot(np.arange(len(original_samples)) * sample_period, original_samples, '.', markersize=.7)
#     ax0.set_xlabel('Time (s)')

#     # Plot the raw samples
#     ax1.plot(np.arange(len(samples)) * sample_period, samples, '.', markersize=.7)

#     if detected_signal:
#         x, y = zip(*detected_signal)
#         ax3.scatter(x * sample_period, y,
#                     marker='x',
#                     color='yellow')


#     ax3.plot(np.arange(len(correlation_threshold_high)) * sample_period, correlation_threshold_high, color='green', label='upper threshold')
#     # ax3.plot(correlation_threshold_low, color='orange', label='lower threshold')
#     ax3.plot(np.arange(len(correlation)) * sample_period, correlation, label='correlation')

#     ax3.set_xlim(ax1.get_xlim())
#     ax3.set_xlabel('Time (s)')

#     # plt.legend()
#     plt.tight_layout()

#     if isinstance(location, str):
#         name = f"{result.metadata['distance']}-{result.metadata['location']}-{result.metadata['experiment_number']}"
#         # plt.savefig(os.path.join(location, f'{name}.pdf'))
#         plt.savefig(os.path.join(location, f'{name}.png'), dpi=300)
#     else:
#         # Treat location as file
#         plt.savefig(location, format='png', dpi=300)

#     plt.close(fig)


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


def get_symbol_groups(result, tolerence, result_name='main'):
    if result_name == 'main':
        detected_signal_index = [x for x, y in result.main_result.detected_signal]
    else:
        detected_signal_index = [x for x, y in result.results[result_name].detected_signal]
    groups = list(get_consecutive_number_groups(detected_signal_index, tolerence=tolerence))

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


def get_symbols(result, offset_limit=.3, tolerence=20, result_name='main'):
    groups, diffs_between_groups = get_symbol_groups(result, tolerence=tolerence, result_name=result_name)
    # print(groups)

    run_time = result.metadata['run_time']
    chip_time = ureg(result.metadata['chip_time']).magnitude / 1e3
    symbol_size = result.metadata.get('symbol_size', 1023)
    symbol_time = chip_time * symbol_size

    all_found_peaks = []
    for start_index in range(len(groups)):
        # print("-------", start_index, "-------")
        good_groups = []
        bad_groups = []
        prev_diff = 0

        peak = np.array(groups[start_index]).mean()
        # print(peak)

        peaks = np.array([np.array(group).mean() for group in groups[start_index:]])
        peaks -= peak  # Put peak at zero
        peaks_times = peaks * result.sample_period.magnitude  # Convert from sample number to time
        # print(peaks_times + peak * result.sample_period.magnitude)
        peaks_symbol_times = peaks_times / symbol_time  # Convert to number of symbols away
        offsets = peaks_symbol_times % 1  # Remove whole number (we only care about offset)

        # print(np.diff(peaks_times))
        enough_space = list(np.diff(peaks_times) > 10)
        enough_space = np.array([True] + enough_space)  # Need to pad the first value
        # print(peaks_symbol_times)
        # print(offsets)

        detected_symbol = np.logical_and(enough_space,
                                         np.logical_or(offsets < offset_limit,
                                                       offsets > 1-offset_limit))

        found_peaks = np.where(detected_symbol)[0]
        found_peaks += start_index
        # print(found_peaks)

        all_found_peaks.append(found_peaks)

    most_peaks = max(all_found_peaks, key=lambda x: len(x))

    good_groups = []
    bad_groups = []
    for i, group in enumerate(groups):
        if i in most_peaks:
            good_groups.append(group)
        else:
            bad_groups.append(group)

    return good_groups, bad_groups


def get_result_score(result):
    good, bad = get_symbols(result)

    run_time = result.metadata['run_time']
    chip_time = ureg(result.metadata['chip_time']).magnitude / 1e3
    symbol_size = result.metadata.get('symbol_size', 1023)
    symbol_time = chip_time * symbol_size
    expected_received_symbols = round((run_time - symbol_time) / symbol_time)

    if not result.metadata.get('transmitting', True):
        return {"Total": 0,
                "Correct": 0,
                "False positive": len(good) + len(bad)}

    return {"Total": expected_received_symbols,
            "Correct": len(good),
            "False positive": len(bad)}


def get_all_results_score(results):
    result_scores = [get_result_score(result) for result in results]

    return glom(result_scores, {'Total': (['Total'], sum),
                                'Correct': (['Correct'], sum),
                                'False positive': (['False positive'], sum)})


def onpc(data_file, folder, lpf_size=30, threshold_size=600, threshold_lag=100,
         threshold_std=4.0, rank_method='min', antenna_select=None,
         antenna_method='average'):
    import onpc_v2

    data_files = itertools.chain(data_file,
        *[glob.glob(os.path.join(f, '*.json')) for f in folder])

    # Remove possible duplicates
    data_files = sorted(set(data_files))

    results = []

    with tqdm(total=len(data_files)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            for file in data_files:
                result = onpc_v2.run(file,
                                     lpf_size=lpf_size,
                                     threshold_size=threshold_size,
                                     threshold_lag=threshold_lag,
                                     threshold_std=threshold_std,
                                     rank_method=rank_method,
                                     antenna_select=antenna_select,
                                     graph=False,
                                     interactive=False,
                                     executor=executor)
                results.append(result)
                pbar.update()

    return results


@cli.command(help="Run ONPC on collected data different parameters")
@click.option('-d', '--data-file', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
@click.option('--lpf-size', default=30)
@click.option('--threshold-size', default=600)
@click.option('--threshold-lag', default=100)
@click.option('--threshold-std', default=4.0)
@click.option('--webpage/--no-webpage', default=False)
def run_onpc(data_file, folder, lpf_size, threshold_size, threshold_lag,
             threshold_std, webpage):

    results = onpc(data_file, folder, lpf_size, threshold_size, threshold_lag,
                   threshold_std)

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

    print(json.dumps(get_all_results_score(sorted_location_results), indent=2))


@cli.command(help="Test ONPC with different threshold factors")
@click.option('-d', '--data-file', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
@click.option('-t', '--threshold_factor', 'threshold_factors', type=float,
              multiple=True, help='Threshold factor')
@click.option('-s', '--start', type=float, default=None)
@click.option('-e', '--end', type=float, default=None)
@click.option('-n', '--number', type=float, default=None)
def run_threshold_factor_test(data_file, folder, threshold_factors, start, end, number):
    scores = []
    interval = 13.7
    actual_peaks = {'1b288878c81d4885accaf7eb8d6bf2e3': np.arange(16.9, 413, interval),
                    '3c52f85e46f74cb29b8d4bbc3ae961e1': np.arange(27.495, 413, interval),
                    '5bec18c060a3493db900007fdb77ea60': np.arange(16.886, 413, interval),
                    '66ec9013e9504104ba7cf6c2aef93d68': np.arange(16.483, 410, interval),
                    '7b20a1adccb348ff9a18ad934e08b3e9': np.arange(22.671, 413, interval),
                    'a026cb5b821b4df8ace9360231ec236b': np.arange(30.803, 413, interval),
                    'aaecd371c93d4d4da81858d83bd8981c': np.arange(20.803, 413, interval),
                    'e7ef6b71022f4151a377760da6be6ba5': np.arange(25.821, 413, interval),
                    'fc79d1bd0caa4975b0fc86db8634b4a7': np.arange(21.673, 413, interval)}
    tolerence = 2

    if start and end and number:
        threshold_factors = list(threshold_factors) + list(np.linspace(start, end, number))

    for factor in threshold_factors:
        results = onpc(data_file, folder, threshold_std=factor)
        false_positives = 0
        false_positives_total = 0
        num_detected = 0
        total_symbols = 0

        for result in results:
            correlation = result.main_result.correlation
            threshold = result.main_result.threshold
            sample_period = result.sample_period.magnitude

            correlation = np.nan_to_num(correlation)
            threshold = np.nan_to_num(threshold)

            # Find all false positives
            prev_start = 0
            for g in actual_peaks[result.name]:
                start = int(round(prev_start / sample_period))
                end = int(round((g - tolerence) / sample_period))
                end = end if end < len(correlation) else len(correlation) - 1

                above_threshold = correlation[start:end] > threshold[start:end]
                false_positives += above_threshold.sum()
                false_positives_total += (correlation[start:end] > 0).sum()

                prev_start = g + tolerence

            # Find all detected symbols
            for g in actual_peaks[result.name]:
                start = int(round((g - tolerence) / sample_period))
                end = int(round((g + tolerence) / sample_period))
                above_threshold = any(correlation[start:end] > threshold[start:end])

                if above_threshold:
                    num_detected += 1

            total_symbols += len(actual_peaks[result.name])

        scores.append({'threshold_factor': factor,
                       'correct': num_detected,
                       'total_correct': total_symbols,
                       'false_positives': int(false_positives),
                       'false_positives_total': int(false_positives_total)})



    print(json.dumps(scores, indent=2))


@cli.command(help="Test ONPC with different low pass filter sizes")
@click.option('-d', '--data-file', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
@click.option('-l', '--lpf-size', 'lpf_sizes', type=int,
              multiple=True, help='Threshold factor')
def run_lpf_size_test(data_file, folder, lpf_sizes):
    scores = []
    for size in lpf_sizes:
        results = onpc(data_file, folder, lpf_size=size)
        score = get_all_results_score(results)
        score['Low pass filter size'] = size
        scores.append(score)

    print(json.dumps(scores, indent=2))


@cli.command(help="Test ONPC with different rank methods")
@click.option('-d', '--data-file', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
def run_rank_method_test(data_file, folder):
    scores = []
    methods = ['average', 'min', 'max', 'dense', 'ordinal']
    for method in methods:
        results = onpc(data_file, folder, rank_method=method)
        score = get_all_results_score(results)
        score['Rank method'] = method
        scores.append(score)

    print(json.dumps(scores, indent=2))


@cli.command(help="Test ONPC with different antenna select configurations")
@click.option('-d', '--data-file', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
def run_antenna_select_test(data_file, folder):
    scores = []
    antennas = [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    for antenna in antennas:
        results = onpc(data_file, folder, antenna_select=antenna)
        score = get_all_results_score(results)
        score['Antenna select'] = antenna
        scores.append(score)

    print(json.dumps(scores, indent=2))


@cli.command(help="Test ONPC with different antenna methods")
@click.option('-d', '--data-file', multiple=True, help='Data file')
@click.option('-f', '--folder', multiple=True, help='Data folder')
def run_antenna_method_test(data_file, folder):
    scores = []
    methods = ['average', 'max', 'min']
    for method in methods:
        results = onpc(data_file, folder, antenna_method=method)
        score = get_all_results_score(results)
        score['Antenna method'] = method
        scores.append(score)

    print(json.dumps(scores, indent=2))



if __name__ == '__main__':
    cli()

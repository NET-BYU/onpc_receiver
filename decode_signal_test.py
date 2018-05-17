import concurrent.futures
import itertools
import logging
import logging.handlers
import glob
import os
import time

import click
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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


@cli.command(help="Run ONPC on collected data different parameters")
@click.argument('config_file', type=click.File('r'))
@click.option('-d', '--data', multiple=True, help='Data file')
@click.option('--low_pass_filter_size', callback=split_num_list)
@click.option('--correlation_buffer_size', callback=split_num_list)
@click.option('--correlation_std_threshold', callback=split_num_list)
def run_data(config_file, data, low_pass_filter_size, correlation_buffer_size,
             correlation_std_threshold):
    import decode_signal

    params = [data, low_pass_filter_size, correlation_buffer_size, correlation_std_threshold]
    # params = [x for x in params if x is not None and x[0] is not None]
    param_combinations = list(itertools.product(*params))

    config = yaml.load(config_file)
    results = []

    with tqdm(total=len(param_combinations)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            future_to_param = {}
            for index, param in enumerate(param_combinations):
                current_config = dict(config)

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
                param = future_to_param[future]
                config, result = future.result()

                results.append((result, param))
                pbar.update()

    for result, param in results:
        print(param, len(result))


if __name__ == '__main__':
    cli()

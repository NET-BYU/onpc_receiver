from collections import deque
import itertools
import multiprocessing
from pathlib import Path

import click
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import stats


def load_data(file_name):
    if file_name[-3:] == 'npy':
        with open(file_name, 'rb') as f:
           return np.load(f)

    try:
        with open('{}.npy'.format(file_name[:-3]), 'rb') as f:
           return np.load(f)
    except FileNotFoundError:
        data = sio.loadmat(file_name)
        new_data = []

        keys = (key[1:] for key in data if key[0] == 'Y')
        keys = (0 if key == '' else int(key) for key in keys)
        keys = sorted(keys)

        for key in keys:
            if key == 0:
                key = ''
            new_data.append(np.array([d[0] for d in data['Y{}'.format(key)]]))

        with open('{}.npy'.format(file_name[:-3]), 'wb') as f:
           np.save(f, new_data)

        return new_data


def get_samples_from_file(file_name):
    for data in load_data(file_name):
        power_data = np.abs(data) ** 2  # Get magnitude and convert to power
        yield power_data


def find_transmissions(samples):
    transmissions = []

    median = np.median(samples)
    std = np.std(samples)
    threshold = median + std

    above = samples > threshold
    above_iter = iter(above)

    WINDOW = 100
    index = 0

    # Find all transmissions
    while True:
        try:
            # Progress to True value (start of signal)
            value = False
            while not value:
                value = next(above_iter)
                index += 1

            start = index

            # Progress until there are no True values for WINDOW samples (end of signal)
            buffer = deque([value], maxlen=WINDOW)
            while any(buffer) is not False:
                buffer.append(next(above_iter))
                index += 1

            end = index - WINDOW

            transmissions.append((start, end))
        except StopIteration:
            break

    # Remove first transmission in case we caught a middle of a transmission
    transmissions.pop(0)

    return transmissions


def find_transmit_pause_times(transmissions, sample_file, index):
    # Find difference between start and end of transmission
    # (transmission duration)
    tx_durations = np.array([end - start for start, end in transmissions])

    # Find difference between end of transmission and start of another transmission
    # (pause duration)
    starts, ends = zip(*transmissions)
    pause_durations = np.array([end - start for start, end in zip(ends, starts[1:])])

    # The data coming in is in terms of samples
    # The spectrum analyzer samples at 50 MHz which is
    # once every 20 ns.
    tx_durations = tx_durations * .02        # Convert to μs
    pause_durations = pause_durations * .02  # Convert to μs

    # Get metadata
    sample_file = Path(sample_file)
    device = sample_file.stem.split('-')[-2]
    pause_duration = int(sample_file.stem.split('-')[-1])

    return device, pause_duration, tx_durations.std(), pause_durations.std()


def run(samples, sample_file, index, cb):
    print("Processing {} - {}".format(sample_file, index))
    transmissions = find_transmissions(samples)
    return cb(transmissions, sample_file, index)


def process_data(sample_files, cb):
    def inputs():
        for sample_file in sample_files:
            for i, samples in enumerate(get_samples_from_file(sample_file)):
                yield samples, sample_file, i, cb

    processes = 8
    with multiprocessing.Pool(processes=processes) as pool:
        return pool.starmap(run, inputs(), chunksize=10)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('sample_files', nargs=-1)
@click.option('--graph/--no-graph', default=False)
def transmit_pause(sample_files, graph):
    data = process_data(sample_files, find_transmit_pause_times)

    for device_type, d in itertools.groupby(data, lambda x: x[0]):
        _, pause_times, tx_std, pause_std = map(np.array, zip(*d))

        print(device_type)
        print(pause_times)
        print(tx_std)
        print(pause_std)

        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(211)
        ax1.scatter(pause_times, tx_std, marker='x')
        ax1.set_ylabel('TX Time Std (μs)')

        ax2 = fig.add_subplot(212)
        ax2.scatter(pause_times, pause_std, marker='x')
        ax2.set_ylabel('Pause Time Std (μs)')

        ax2.set_xlabel('Pause Time (μs)')

        plt.tight_layout()
        plt.savefig('timing-results-{}.pdf'.format(device_type))


@cli.command()
@click.argument('sample_files', nargs=-1)
@click.argument('period_time')
@click.option('--graph/--no-graph', default=False)
def transmit_period(sample_files, period_time, graph):
    all_data = []
    for sample_file in sample_files:
        for i, samples in enumerate(get_samples_from_file(sample_file)):
            print("Processing {} - {}".format(sample_file, i))
            transmissions = find_transmissions(samples)
            all_data.append(find_transmit_pause_times(transmissions))

            if graph:
                # TODO: Do all graphing here!!!!
                pass


    print(list(all_data))



if __name__ == '__main__':
    cli()

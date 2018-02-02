from collections import deque
import itertools
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import stats


def load_data(file_name):
    try:
        with open('{}.npy'.format(file_name), 'rb') as f:
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

        with open('{}.npy'.format(file_name), 'wb') as f:
           np.save(f, new_data)

        return new_data


def get_samples_from_file(file_name):
    for data in load_data(file_name):
        power_data = np.abs(data) ** 2  # Get magnitude and convert to power
        yield power_data


def test_timing(sample_file, graph):
    all_data = []
    for file_index, data in enumerate(get_samples_from_file(sample_file)):
        print("Processing {} - {}".format(sample_file, file_index))

        median = np.median(data)
        std = np.std(data)
        threshold = median + std

        above = data > threshold
        above_iter = iter(above)

        WINDOW = 100
        index = 0
        peaks = []

        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(411)
        ax1.plot(data, '.')
        ax1.axhline(y=threshold, color='red')

        ax2 = fig.add_subplot(412)
        ax2.plot(above, '.')


        # Find all peaks
        while True:
            try:
                # Progress to True value (start of signal)
                value = False
                while not value:
                    value = next(above_iter)
                    index += 1

                start = index
                ax2.axvline(x=index, color='red')

                # Progress until there are no True values for WINDOW samples (end of signal)

                buffer = deque([value], maxlen=WINDOW)
                while any(buffer) is not False:
                    buffer.append(next(above_iter))
                    index += 1

                # TODO: What about the WINDOW values we jumped ahead?
                end = index - WINDOW
                ax2.axvline(x=index - WINDOW, color='red')

                peaks.append((start, end))
            except StopIteration:
                break

        # Analyze peaks
        tx_durations = []
        noise_durations = []

        for start, end in peaks:
            tx_durations.append(end - start)

        starts, ends = zip(*peaks)
        for start, end in zip(ends, starts[1:]):
            noise_durations.append(end - start)

        # Remove first TX in case we caught a middle of a TX
        tx_durations.pop(0)

        ax3 = fig.add_subplot(413)
        ax3.plot(tx_durations, '.')

        ax3 = fig.add_subplot(414)
        ax3.plot(noise_durations, '.')

        if graph:
            plt.tight_layout()
            plt.savefig(sample_file + '-{}.png'.format(file_index))

        all_data.append({"tx_durations": np.array(tx_durations),
                         "noise_durations": np.array(noise_durations),
                         "file": sample_file,
                         "index": file_index})

    return all_data


def process_result(data):
    return (int(Path(data['file']).stem.split('-')[-1]),
            data['tx_durations'].std(),
            data['noise_durations'].std())



def main(sample_files, graph):
    processes = min(len(sample_files), 4)
    with multiprocessing.Pool(processes=processes) as pool:
        data = pool.starmap(test_timing,
                            [(sample_file, graph)
                             for sample_file in sample_files])
    data = itertools.chain(*data)
    data = (process_result(d) for d in data)

    pause_times, tx_std, noise_std = map(np.array, zip(*data))

    print(pause_times)
    print(tx_std)
    print(noise_std)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(211)
    ax1.scatter(pause_times, tx_std, marker='x')
    ax1.set_ylabel('TX Time Std')

    ax2 = fig.add_subplot(212)
    ax2.scatter(pause_times, noise_std, marker='x')
    ax2.set_ylabel('Noise Time Std')

    ax2.set_xlabel('Pause Time (μs)')

    plt.tight_layout()
    plt.savefig('result.pdf')




if __name__ == '__main__':
    import click
    import yaml

    @click.command()
    @click.argument('sample_files', nargs=-1)
    @click.option('--graph/--no-graph', default=False)
    def cli(sample_files, graph):
        main(sample_files, graph)


    cli()

from collections import deque
from itertools import groupby

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def load_data(file_name):
    try:
        with open('{}.npy'.format(file_name), 'rb') as f:
           return np.load(f)
    except FileNotFoundError:
        data = sio.loadmat(file_name)
        data = data['Y']
        data = np.array([d[0] for d in data])

        with open('{}.npy'.format(file_name), 'wb') as f:
           np.save(f, data)

        return data


def get_samples_from_file(file_name):
    data = load_data(file_name)
    chunk_size = 100_000

    data = data[:len(data) - (len(data) % chunk_size)]
    power_data = np.abs(data) ** 2  # Get magnitude and convert to power

    return power_data


def main(sample_file, start, end):
    data = get_samples_from_file(sample_file)
    data = data[start:end]

    median = np.median(data)
    std = np.std(data)
    threshold = median + std

    above = data > threshold
    above_iter = iter(above)

    WINDOW = 1000
    index = 0
    peaks = []

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(211)
    ax1.plot(data, '.')
    ax1.axhline(y=threshold, color='red')

    ax2 = fig.add_subplot(212)
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


    print(tx_durations)
    print(noise_durations)

    plt.tight_layout()
    plt.savefig('test_timing.png')


if __name__ == '__main__':
    import click
    import yaml

    @click.command()
    @click.argument('sample_file')
    @click.argument('start', type=click.INT)
    @click.argument('end', type=click.INT)
    def cli(sample_file, start, end):
        main(sample_file, start, end)


    cli()

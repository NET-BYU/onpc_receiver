from collections import deque
import functools
import itertools
import multiprocessing
from pathlib import Path
import time

import arrow
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
        return pool.starmap(run, inputs())


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


def find_transmit_period(transmissions, sample_file, index, period):
    starts = np.array([start for start, _ in transmissions])
    starts -= starts[0]  # Make zero relative to first element

    # The data coming in is in terms of samples
    # The spectrum analyzer samples at 50 MHz which is
    # once every 20 ns.
    starts = starts * .02  # Convert to μs
    # print(starts)

    # Convert from μs to periods
    periods = starts / period
    # print(periods)

    # Calculate how many missed periods there are
    missing = sorted(set(np.arange(len(periods))) - set(np.around(periods)))

    # Calculate offset
    offsets = periods - np.around(periods)
    offsets *= period  # Convert from periods into μs
    offsets = offsets[1:]  # Remove first value because it is always zero
    # print(offsets)

    # Get metadata
    sample_file = Path(sample_file)
    device = sample_file.stem.split('-')[-2]
    pause_duration = int(sample_file.stem.split('-')[-1])

    return sample_file, device, pause_duration, missing, offsets


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
@click.argument('period_time', type=click.INT)
@click.option('--graph/--no-graph', default=False)
def transmit_period(sample_files, period_time, graph):
    data = process_data(sample_files,
                        functools.partial(find_transmit_period,
                                          period=period_time))

    all_missing = [len(d[3]) for d in data]
    all_offsets = [d[4] for d in data]
    std_dev = np.concatenate(all_offsets).std()
    mean = np.concatenate(all_offsets).mean()

    print(f"Total missing: {sum(all_missing)} ({all_missing})")
    print(f"Std: {std_dev}")

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    ax.plot(np.concatenate(all_offsets), '.')
    ax.axhline(y=mean + std_dev, color='r')
    ax.axhline(y=mean - std_dev, color='r')

    ax.set_ylabel('Offset from Transmission Period (μs)')
    ax.set_xlabel('Transmission Number')

    plt.tight_layout()
    plt.savefig(f'offsets_transmission-{period_time}.pdf')
    plt.close(fig)

    #####################################################

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.boxplot(all_offsets)

    ax.set_ylabel('Offset from Transmission Period (μs)')
    ax.set_xlabel('Experiment Number')

    plt.tight_layout()
    plt.savefig(f'offsets_experiments-{period_time}.pdf')
    plt.close(fig)

    #####################################################

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(np.concatenate(all_offsets),
                               bins=100,
                               normed=1,
                               histtype='step',
                               cumulative=True)
    ax.axvline(x=mean + std_dev, color='r')
    ax.axvline(x=mean - std_dev, color='r')

    ax.set_ylabel('Probability')
    ax.set_xlabel('Offset from Transmission Period (μs)')

    plt.tight_layout()
    plt.savefig(f'offsets_cdf-{period_time}.pdf')
    plt.close(fig)

    # for name, device_type, pause_time, missing, offsets in data:
    #     print(name)
    #     print(offsets)


def process_wl_samples(samples, name, total_time):
    def process_data(data):
        if '-' not in data:
            return [np.nan, np.nan, np.nan]

        return [float(d[:-3]) for d in data.split()]

    data = (line.strip() for line in samples)  # Take off new lines
    data = (process_data(d) for d in data)

    ys1, ys2, ys3 = zip(*data)

    samples_collected = len(ys1)
    xs = np.linspace(0, total_time, num=samples_collected)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.plot(xs, ys1, '.-')
    ax2.plot(xs, ys2)
    ax3.plot(xs, ys3)

    # ax1.set_xlim(1, 2)

    ax1.set_ylabel('Noise (dBm)')
    ax2.set_ylabel('Noise (dBm)')
    ax3.set_ylabel('Noise (dBm)')
    ax3.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(f'wl_timing-{name}.png')
    plt.savefig(f'wl_timing-{name}.pdf')
    plt.close(fig)


@cli.command()
@click.argument('sample_file', nargs=1, type=click.File())
@click.option('--total_time', type=click.FLOAT, default=None)
def test_ap_wl(sample_file, total_time):
    process_wl_samples(sample_file, Path(sample_file.name).stem, total_time)


@cli.command()
@click.argument('remote', nargs=1)
@click.argument('name', nargs=1)
def test_ap_wl_remote_timing(remote, name):
    import paramiko
    import re
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote, username='root')

    # wl_command = "wl phy_rxiqest -r 1 -s 15"
    wl_command = "date -Iseconds >> time.out; wl phy_rxiqest -r 1 -s 15"
    command = f"time sh -c 'for i in `seq 1 1000`; do {wl_command} >> data.out; done'"

    # Make sure old samples are deleted
    print('Removing old data file...')
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('rm data.out; rm time.out')
    ssh_stdout.read()

    print('Collect samples...')
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
    time_results = ssh_stderr.read().decode().split('\n')[0]
    re_result = re.search("(\\d+)m (\\d+.\\d+)s", time_results)
    run_time = int(re_result.group(1)) * 60 + float(re_result.group(2))

    # Get samples
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('cat data.out')
    raw_samples = ssh_stdout.read()
    samples = raw_samples.decode().split('\n')

    print(f'Run time: {run_time} ({run_time / len(samples)})')

    # Save samples for later
    with open(f'{name}-data.log', 'wb') as f:
        f.write(raw_samples)

    # Get times
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('cat time.out')
    raw_times = ssh_stdout.read()
    times = raw_times.decode().strip().split('\n')

    # Save times for later
    with open(f'{name}-times.log', 'wb') as f:
        f.write(raw_times)

    # Delete samples
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('rm data.out; rm time.out')
    ssh_stdout.read()

    print('Processing samples and times...')
    def process_data(data):
        if '-' not in data:
            return [np.nan, np.nan, np.nan]

        return [float(d[:-3]) for d in data.split()]

    times = [arrow.get(t) for t in times]
    print(f"Date run time: {times[-1] - times[0]}")

    data = (line.strip() for line in samples)  # Take off new lines
    data = (process_data(d) for d in data)

    ys1, ys2, ys3 = zip(*data)

    samples_collected = len(ys1)
    xs = np.linspace(0, run_time, num=samples_collected)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(xs, ys1, '.-')
    ax2.plot([t.datetime for t in times])

    # ax1.set_xlim(1, 2)

    ax1.set_ylabel('Noise (dBm)')
    ax2.set_ylabel('Time')

    plt.tight_layout()
    plt.savefig(f'wl_timing-{name}.png')
    plt.savefig(f'wl_timing-{name}.pdf')
    plt.close(fig)


@cli.command()
@click.argument('remote', nargs=1)
@click.option('--num_samples', default=1000)
@click.option('--num_runs', default=5)
def test_ap_wl_sample_time(remote, num_samples, num_runs):
    import paramiko
    import re
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote, username='root')

    wl_command = "wl phy_rxiqest -r 1 -s 15"
    command = f"time sh -c 'for i in `seq 1 {num_samples}`; do {wl_command} >> data.out; done'"

    run_results = []
    for i in range(num_runs):
        print(f"\n############ RUN {i+1}")
        # print('Removing old data file...')
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('rm data.out')
        ssh_stdout.read()

        # print('Collect samples...')
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
        time_results = ssh_stderr.read().decode().split('\n')[0]
        re_result = re.search("(\\d+)m (\\d+.\\d+)s", time_results)
        run_time = int(re_result.group(1)) * 60 + float(re_result.group(2))

        run_results.append(run_time)
        print(f'Run time: {run_time} s')
        print(f'Time per sample: {run_time / num_samples} s ({run_time}/{num_samples})')

        # print('Removing data file...')
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('rm data.out')
        ssh_stdout.read()

        # Give some time to let things settle
        time.sleep(1)

    run_results = np.array(run_results)
    run_results = run_results / num_samples  # Convert from total time to time per sample
    print("\n\n################ Results (Time per sample)")
    print(f"Mean: {run_results.mean() * 1000} ms")
    print(f"Median: {np.median(run_results) * 1000} ms")
    print(f"Std: {run_results.std() * 1000} ms")



@cli.command()
@click.argument('sample_file', nargs=1, type=click.File())
def test_ap_proc(sample_file):
    data = [float(line.strip()) for line in sample_file]


    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    ax.plot(data)

    ax.set_ylabel('Noise (dBm)')
    ax.set_xlabel('Time (??)')

    plt.tight_layout()
    plt.savefig(f'temp.pdf')
    plt.close(fig)



if __name__ == '__main__':
    cli()

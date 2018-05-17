import os
import time
import json
import uuid

import click
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def process_wl_samples(samples):
    def process_data(data):
        if '-' not in data:
            return [np.nan, np.nan, np.nan]

        return [float(d[:-3]) for d in data.split()]

    data = (line.strip() for line in samples)  # Take off new lines
    data = (process_data(d) for d in data)

    return list(data)


def graph(data, name, total_time):
    ys1, ys2, ys3 = zip(*data)
    samples_collected = len(ys1)
    xs = np.linspace(0, total_time, num=samples_collected)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.plot(xs, ys1, '.-')
    ax2.plot(xs, ys2, '.-')
    ax3.plot(xs, ys3, '.-')

    # ax1.set_xlim(1, 2)
    ax2.set_ylim(ax1.get_ylim())
    ax3.set_ylim(ax1.get_ylim())

    ax1.set_ylabel('Noise (dBm)')
    ax2.set_ylabel('Noise (dBm)')
    ax3.set_ylabel('Noise (dBm)')
    ax3.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(f'wl_timing-{name}.png')
    plt.savefig(f'wl_timing-{name}.pdf')
    plt.close(fig)


@click.command()
@click.argument('remote', nargs=1)
@click.argument('name', nargs=1)
@click.option('--folder', type=click.Path())
@click.option('--num_samples', default=7000)
@click.option('--tap', default=None)
@click.option('--distance', type=float, prompt=True)
@click.option('--location', prompt=True)
@click.option('--description', prompt=True)
@click.option('--experiment_number', type=int, prompt=True)
@click.option('--symbol-size', type=int, prompt=True)
@click.option('--transmitting', type=bool, prompt=True)
def get_samples(remote, name, folder, num_samples, tap, distance, location, description, experiment_number, symbol_size, transmitting):
    import paramiko
    import re

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote, username='root', timeout=10)

    wl_command = "wl phy_rxiqest -r 1 -s 15"
    command = f"time sh -c 'for i in `seq 1 {num_samples}`; do {wl_command} >> data.out; done'"

    # Make sure old samples are deleted
    print('Removing old data file...')
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('rm data.out')
    ssh_stdout.read()

    print('Collect samples...')
    now = time.time()
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
    time_results = ssh_stderr.read().decode().split('\n')[0]
    re_result = re.search("(\\d+)m (\\d+.\\d+)s", time_results)
    run_time = int(re_result.group(1)) * 60 + float(re_result.group(2))

    # Get samples
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('cat data.out')
    raw_samples = ssh_stdout.read()
    samples = raw_samples.decode().strip().split('\n')

    print(f'Run time: {run_time} ({run_time}/{len(samples)} = {run_time / len(samples)})')

    # Delete samples
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('rm data.out')
    ssh_stdout.read()

    print('Processing samples...')
    samples = process_wl_samples(samples)

    # Save samples for later
    file_name = str(uuid.uuid4()).replace('-', '') + '.json'
    with open(os.path.join(folder, file_name), 'w') as f:
        json.dump({'name': name,
                   'capture_time': now,
                   'run_time': run_time,
                   'command': wl_command,
                   'count': num_samples,
                   'distance': distance,
                   'tap': tap,
                   'location': location,
                   'experiment_number': experiment_number,
                   'description': description,
                   'symbol_size': symbol_size,
                   'transmitting': transmitting,
                   'samples': samples}, f)

    graph(samples, name, run_time)


if __name__ == '__main__':
    get_samples()

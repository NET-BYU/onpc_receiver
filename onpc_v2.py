import concurrent.futures
from functools import partial
import json
import logging
import pathlib
import time

import attr
import click
import numpy as np
import pandas as pd
import pint
import psutil
from scipy import signal, stats
import yaml


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
ureg = pint.UnitRegistry()

@attr.s
class Result(object):
    samples = attr.ib()
    correlation = attr.ib()
    threshold = attr.ib()
    detected_signal = attr.ib()
    name = attr.ib()

@attr.s
class ExperimentResult(object):
    main_result = attr.ib()
    results = attr.ib()
    sample_period = attr.ib()
    metadata = attr.ib()
    name = attr.ib()


def split_num_list(ctx, param, value):
    if value is None or value == '':
        return None
    try:
        try:
            return [int(x) for x in value.split(',') if x]
        except ValueError:
            return [float(x) for x in value.split(',') if x]
    except ValueError:
        raise click.BadParameter('List must only contain numbers')


@click.command()
@click.argument('data_file', type=click.File('r'))
@click.option('--lpf-size', default=30)
@click.option('--threshold-size', default=600)
@click.option('--threshold-lag', default=100)
@click.option('--threshold-std', default=4.0)
@click.option('--sample-factor', default=3)
@click.option('--limiting-threshold-percentile', default=10)
@click.option('--limiting-std-factor', default=.7)
@click.option('--limiting-threshold', default=None, type=float)
@click.option('--limiting-std', default=None, type=float)
@click.option('--rank-method', default='min',
              type=click.Choice(['average', 'min', 'max', 'dense', 'ordinal']))
@click.option('--run-raw/--no-run-raw', default=False)
@click.option('--run-limited/--no-run-limited', default=False)
@click.option('--run-ranked/--no-run-ranked', default=True)
@click.option('--antenna-select', default=None, callback=split_num_list)
@click.option('--graph/--no-graph', default=True)
@click.option('--interactive/-no-interactive', default=False)
def main(*args, **kwargs):
    return run(*args, **kwargs, logging_level=logging.INFO)


def run(data_file, lpf_size=30, threshold_size=600, threshold_lag=100,
         threshold_std=4.0, sample_factor=3, limiting_threshold_percentile=10,
         limiting_std_factor=.7, limiting_threshold=None, limiting_std=None,
         rank_method='min', run_raw=False, run_limited=False, run_ranked=True,
         antenna_select=None, graph=False, interactive=False,
         logging_level=logging.ERROR):

    if logging_level:
        LOGGER.setLevel(logging_level)

    if isinstance(data_file, str):
        with open(data_file) as f:
            experiment_name = pathlib.Path(data_file).stem
            experiment_data = json.load(f)
    else:
        experiment_name = pathlib.Path(data_file.name).stem
        experiment_data = json.load(data_file)

    samples, sample_period = prepare_samples(experiment_data,
                                             sample_factor,
                                             antenna_select=antenna_select)

    symbol = get_symbol(experiment_data, sample_factor)

    LOGGER.info("Location: %s", experiment_data['location'])
    LOGGER.info("Description: %s", experiment_data['description'])

    results = []

    if run_raw:
        start = time.time()
        raw_result = decode_symbols(samples, symbol,
                                    name='raw',
                                    limiting_func=lambda x: x,
                                    correlation_func=partial(regular_correlation,
                                                             lpf_size=lpf_size),
                                    threshold_func=partial(rolling_std_factor_threshold,
                                                           size=threshold_size,
                                                           factor=threshold_std,
                                                           lag=threshold_lag))
        end = time.time()
        LOGGER.info("Raw run time: %s", end - start)
        results.append(raw_result)

    if run_limited:
        start = time.time()
        limited_result = decode_symbols(samples, symbol,
                                        name='soft limited',
                                        limiting_func=partial(norm_limit_samples,
                                                              threshold_percentile=limiting_threshold_percentile,
                                                              std_factor=limiting_std_factor,
                                                              threshold=limiting_threshold,
                                                              std=limiting_std),
                                        correlation_func=partial(regular_correlation,
                                                                 lpf_size=lpf_size),
                                        threshold_func=partial(rolling_std_factor_threshold,
                                                               size=threshold_size,
                                                               factor=threshold_std,
                                                               lag=threshold_lag))
        end = time.time()
        LOGGER.info("Norm limiting run time: %s", end - start)
        results.append(limited_result)

    # start = time.time()
    # slow_rank_result = decode_symbols(samples, symbol,
    #                                   limiting_func=lambda x: x,
    #                                   correlation_func=partial(slow_rank_correlation),
    #                                   threshold_func=partial(rolling_std_factor_threshold,
    #                                                          size=threshold_size,
    #                                                          factor=threshold_std,
    #                                                          lag=threshold_lag))
    # end = time.time()
    # LOGGER.info("Slow rank limiting run time: %s", end - start)
    # results.append(slow_rank_result)

    if run_ranked:
        start = time.time()
        rank_result = decode_symbols(samples, symbol,
                                     name='ranked',
                                     limiting_func=lambda x: x,
                                     correlation_func=partial(rank_correlation,
                                                              method=rank_method),
                                     threshold_func=partial(std_factor_threshold,
                                                            factor=threshold_std))
        end = time.time()
        LOGGER.info("Rank limiting run time: %s", end - start)
        results.append(rank_result)

    # Combine all results together
    results = {result.name: result for result in results}

    if graph:
        graph_data(experiment_name, sample_period,
                   results=results,
                   interactive=interactive)

    return ExperimentResult(main_result=results['ranked'],
                            results=results,
                            sample_period=sample_period,
                            metadata=experiment_data,
                            name=experiment_name)


def decode_symbols(samples, symbol, limiting_func, correlation_func, threshold_func, name=None):
    samples = limiting_func(samples)
    correlation = correlation_func(samples, symbol)
    threshold = threshold_func(correlation)
    peak_xs, peak_ys = find_peaks(correlation, threshold)


    empty = np.empty(len(symbol))
    empty[:] = np.nan

    if isinstance(threshold, (float, int)):
        threshold = np.concatenate((empty, np.ones(len(correlation)) * threshold))
    else:
        threshold = np.concatenate((empty, np.array(threshold)))

    correlation = np.concatenate((empty, np.array(correlation)))

    peak_xs += len(empty)

    peaks = list(zip(peak_xs, peak_ys))

    return Result(samples=samples,
                  correlation=correlation,
                  threshold=threshold,
                  detected_signal=peaks,
                  name=name)


def find_peaks(correlation, threshold):
    # Calculate threshold crossings
    peak_xs = np.where(correlation > threshold)[0]
    peak_ys = correlation[peak_xs]

    return peak_xs, peak_ys


def slow_rank_correlation(samples, symbol):
    def calc(samples):
        rank = stats.rankdata(samples)
        rank = rank - (len(rank) / 2)  # Make it zero mean
        rank = rank / (len(rank) / 2)  # Make values between -1 and 1
        return (rank * symbol).sum()

    correlation = pd.Series(samples).rolling(window=len(symbol)).apply(calc, raw=True)
    correlation = correlation[len(symbol):]

    return correlation


def run_rank_correlation(samples, symbol, method='average'):
    def calc(data):
        rank = stats.rankdata(data, method)
        rank = rank - (len(rank) / 2)  # Make it zero mean
        rank = rank / (len(rank) / 2)  # Make values between -1 and 1
        return (rank * symbol).sum()

    correlation = np.array(pd.Series(samples).rolling(window=len(symbol)).apply(calc, raw=True))
    return correlation[len(symbol):]


def rank_correlation(samples, symbol, num_splits=8, method='average'):
    split_index = round(len(samples) / num_splits)

    # Split up samples into parts
    samples = np.array(samples)
    split_samples = []
    for split in range(num_splits):
        start = split_index * split
        end = split_index * (split + 1)

        if split > 0:
            start -= len(symbol)

        split_samples.append(samples[start:end])

    # Process the different parts in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_splits, psutil.cpu_count())) as executor:
        futures = {}
        for i, d in enumerate(split_samples):
            f = executor.submit(run_rank_correlation, d, symbol, method=method)
            futures[f] = i

        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append((futures[future], future.result()))

    # Make sure results are in the right order
    results = sorted(results)
    i, correlation_parts = zip(*results)

    # Combine parts back together
    correlation = np.concatenate(correlation_parts)
    correlation = pd.Series(correlation)

    return correlation


def regular_correlation(samples, symbol, lpf_size):
    correlation = np.correlate(symbol, samples)
    correlation = np.flip(correlation, 0)
    correlation = pd.Series(correlation)
    correlation = correlation / np.sum(symbol == 1)

    # Low pass filter
    if lpf_size:
        correlation = correlation.rolling(window=lpf_size).mean()

    return correlation


def std_factor_threshold(correlation, factor):
    threshold = correlation.std() * factor
    return threshold


def rolling_std_factor_threshold(correlation, size, factor, lag):
    def calc_threshold(data):
        data = data[:-lag]
        return data.std() * factor + data.mean()

    threshold = correlation.rolling(window=size).apply(calc_threshold,
                                                       raw=True)

    return threshold


def norm_limit_samples(raw_samples, threshold_percentile=10, std_factor=.7,
                       threshold=None, std=None):

    threshold = threshold or np.percentile(raw_samples, threshold_percentile)
    std = std or raw_samples.std()

    LOGGER.info("Std: %s", std)
    LOGGER.info("Threshold: %s", threshold)

    samples = stats.norm.cdf(raw_samples, loc=threshold, scale=std * std_factor)
    samples = (samples * 4) - 3  # Set values between -1 and 1

    return samples


def get_symbol(experiment_data, sample_factor, symbols_file='symbols.yaml'):
    with open(symbols_file) as f:
        symbols = yaml.load(f)

    if 'symbol_number' not in experiment_data:
        LOGGER.warning("symbol_number is not specified in experiment data. Using symbol 1.")
        symbol = symbols[1]
    else:
        # experiment_data['symbol_number'] = 2
        # experiment_data['symbol_number'] = 3
        LOGGER.info("Using symbol number: %s", experiment_data['symbol_number'])
        symbol = symbols[experiment_data['symbol_number']]

    symbol = np.array(symbol) * 2 - 1
    symbol = np.repeat(symbol, sample_factor)
    return symbol


def prepare_samples(data, sample_factor=3, antenna_select=None):
    chip_time = ureg(data['chip_time'])
    antennas = list(map(pd.Series, zip(*data['samples'])))
    antenna_select = np.array(antenna_select or [1, 3])
    LOGGER.info("Using antennas: %s", antenna_select)
    antenna_select -= 1

    if len(antenna_select) == 0:
        LOGGER.error("antenna_select must have at least one value")
        exit()


    total = antennas[antenna_select[0]].interpolate()
    for a in antenna_select[1:]:
        total += antennas[a].interpolate()

    samples = total / len(antenna_select)

    LOGGER.info("Reading sample file:")
    LOGGER.info("\tNumber of samples collected: %s", len(samples))
    LOGGER.info("\tRun time: %s s (%s ms)", data['run_time'], 1000 * data['run_time'] / len(samples))
    LOGGER.info("\tChip time: %s", chip_time)
    LOGGER.info("\tSample factor: %s", sample_factor)

    new_samples, sample_period = resample(samples,
                                          data['run_time'],
                                          chip_time,
                                          sample_factor)

    LOGGER.info("\tNew sample period: %s", sample_period.to(ureg.ms))

    return pd.Series(new_samples), sample_period


def resample(samples, sample_time, chip_time, sample_factor):
    sample_time = sample_time * ureg.s  # Convert to seconds
    # See how many samples should have been collected during sample time
    num_samples = (sample_time / (chip_time / sample_factor)).to_base_units()
    num_samples = round(num_samples.magnitude)

    new_samples = signal.resample(samples, num_samples)
    new_samples[new_samples < samples.min()] = samples.min()

    sample_period = sample_time / len(new_samples)
    return new_samples, sample_period


def graph_data(name, sample_period, results, interactive=False):
    if not interactive:
        import matplotlib
        matplotlib.use('agg')

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(results) * 2, 1, figsize=(8,6), sharex=True)

    for i, result in enumerate(results.values()):
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
    plt.savefig(f'graphs/decoded-{name}.png', dpi=300)

    if interactive:
        plt.show()


if __name__ == '__main__':
    result = main()

    if result.detected_signal:
        print(f"Success!!! ({result.detected_signal})")
    else:
        print("Didn't find data")

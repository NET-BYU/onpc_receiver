import json
import logging
import pathlib

import attr
import click
import numpy as np
import pandas as pd
import pint
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

@attr.s
class ExperimentResult(object):
    original_result = attr.ib()
    limited_result = attr.ib()
    sample_period = attr.ib()
    metadata = attr.ib()
    name = attr.ib()


@click.command()
@click.argument('data_file', type=click.File('r'))
@click.option('--lpf-size', default=30)
@click.option('--threshold-size', default=600)
@click.option('--threshold-lag', default=100)
@click.option('--threshold-std', default=4.5)
@click.option('--sample-factor', default=3)
@click.option('--limiting-threshold-percentile', default=10)
@click.option('--limiting-std-factor', default=.7)
@click.option('--limiting-threshold', default=None, type=float)
@click.option('--limiting-std', default=None, type=float)
@click.option('--graph/--no-graph', default=True)
@click.option('--interactive/-no-interactive', default=False)
def main(*args, **kwargs):
    return run(*args, **kwargs, logging_level=logging.INFO)


def run(data_file, lpf_size=30, threshold_size=600, threshold_lag=100,
         threshold_std=4.5, sample_factor=3, limiting_threshold_percentile=10,
         limiting_std_factor=.7, limiting_threshold=None, limiting_std=None,
         graph=False, interactive=False, logging_level=logging.ERROR):

    if logging_level:
        LOGGER.setLevel(logging_level)

    if isinstance(data_file, str):
        with open(data_file) as f:
            experiment_name = pathlib.Path(data_file).stem
            experiment_data = json.load(f)
    else:
        experiment_name = pathlib.Path(data_file.name).stem
        experiment_data = json.load(data_file)

    raw_samples, sample_period = prepare_samples(experiment_data,
                                                 sample_factor)

    # raw_samples = raw_samples[int(150 / sample_period.magnitude):int(-15 / sample_period.magnitude):]

    symbol = get_symbol(experiment_data)

    LOGGER.info("Location: %s", experiment_data['location'])
    LOGGER.info("Description: %s", experiment_data['description'])

    samples = limit_samples(raw_samples,
                            threshold_percentile=limiting_threshold_percentile,
                            std_factor=limiting_std_factor,
                            threshold=limiting_threshold,
                            std=limiting_std)

    raw_result = decode_symbols(raw_samples, symbol, sample_factor, lpf_size,
                                threshold_size, threshold_std, threshold_lag)

    limited_result = decode_symbols(samples, symbol, sample_factor, lpf_size,
                                    threshold_size, threshold_std, threshold_lag)

    if graph:
        graph_data(experiment_name, raw_result, limited_result, sample_period,
                   interactive)

    return ExperimentResult(original_result=raw_result,
                            limited_result=limited_result,
                            sample_period=sample_period,
                            metadata=experiment_data,
                            name=experiment_name)


def decode_symbols(samples, symbol, sample_factor, lpf_size, threshold_size,
                   threshold_std, threshold_lag):
    correlation = calculate_correlation(samples,
                                        symbol,
                                        sample_factor,
                                        lpf_size)

    threshold = calculate_threshold(correlation,
                                    threshold_size,
                                    threshold_std,
                                    threshold_lag)

    peak_xs, peak_ys = find_peaks(correlation, threshold)

    # Make all data the same size (good for graphing)
    empty = np.empty(len(symbol) * sample_factor - 1)
    empty[:] = np.nan
    correlation = np.concatenate((empty, np.array(correlation)))
    threshold = np.concatenate((empty, np.array(threshold)))
    peak_xs += len(empty)

    peaks = list(zip(peak_xs, peak_ys))

    return Result(samples=samples,
                  correlation=correlation,
                  threshold=threshold,
                  detected_signal=peaks)


def find_peaks(correlation, threshold):
    # Calculate threshold crossings
    peak_xs = np.where(correlation > threshold)[0]
    peak_ys = correlation[peak_xs]

    return peak_xs, peak_ys


def calculate_correlation(samples, symbol, sample_factor, lpf_size):

    # Calculate correlation
    correlation = np.correlate(np.repeat(symbol, sample_factor), samples)
    correlation = np.flip(correlation, 0)
    correlation = pd.Series(correlation)
    correlation = correlation / np.sum(symbol == 1)

    # Low pass filter
    correlation = correlation.rolling(window=lpf_size).mean()

    # correlation = signal.medfilt(correlation, 101)
    # correlation = pd.Series(correlation)

    return correlation


def calculate_threshold(correlation, threshold_size, threshold_std, threshold_lag):
    def calc_threshold(data):
        data = data[:-threshold_lag]
        return data.std() * threshold_std + data.mean()

    threshold = correlation.rolling(window=threshold_size).apply(
        calc_threshold)

    return threshold


def limit_samples(raw_samples, threshold_percentile=10, std_factor=.7, threshold=None, std=None):

    threshold = threshold or np.percentile(raw_samples, threshold_percentile)
    std = std or raw_samples.std()

    # Overwrite calculated values
    # std = 0.768194724106
    # threshold = -98.8879846223

    LOGGER.info("Std: %s", std)
    LOGGER.info("Threshold: %s", threshold)

    samples = stats.norm.cdf(raw_samples, loc=threshold, scale=std * std_factor)
    samples = (samples * 4) - 3  # Set values between -1 and 1

    return samples


def get_symbol(experiment_data, symbols_file='symbols.yaml'):
    with open(symbols_file) as f:
        symbols = yaml.load(f)

    if 'symbol_number' not in experiment_data:
        LOGGER.warning("symbol_number is not specified in experiment data. Using symbol 1.")
        symbol = symbols[1]
    else:
        symbol = symbols[experiment_data['symbol_number']]

    symbol = np.array(symbol) * 2 - 1
    return symbol


def prepare_samples(data, sample_factor=3):
    chip_time = ureg(data['chip_time'])
    antenna1, antenna2, antenna3 = map(pd.Series, zip(*data['samples']))

    antenna1 = antenna1.interpolate()
    antenna2 = antenna2.interpolate()
    antenna3 = antenna3.interpolate()

    samples = (antenna1 + antenna2 + antenna3) / 3
    # samples = antenna1

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

    return new_samples, sample_period


def resample(samples, sample_time, chip_time, sample_factor):
    sample_time = sample_time * ureg.s  # Convert to seconds
    # See how many samples should have been collected during sample time
    num_samples = (sample_time / (chip_time / sample_factor)).to_base_units()
    num_samples = round(num_samples.magnitude)

    new_samples = signal.resample(samples, num_samples)
    new_samples[new_samples < samples.min()] = samples.min()

    sample_period = sample_time / len(new_samples)
    return new_samples, sample_period


def graph_data(name, original_result, result, sample_period, interactive=False):
    if not interactive:
        import matplotlib
        matplotlib.use('agg')

    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(8,6), sharex=True)

    ax0.plot(np.arange(len(original_result.samples)) * sample_period,
             original_result.samples, '.', markersize=.7)

    ax1.plot(np.arange(len(original_result.threshold)) * sample_period, original_result.threshold, color='green', linewidth=1)
    ax1.plot(np.arange(len(original_result.correlation)) * sample_period, original_result.correlation, linewidth=1)

    if original_result.detected_signal:
        xs, ys = zip(*original_result.detected_signal)
        ax1.scatter(xs * sample_period, ys,
                    marker='x',
                    color='yellow')

    ax2.plot(np.arange(len(result.samples)) * sample_period,
             result.samples, '.', markersize=.7)

    ax3.plot(np.arange(len(result.threshold)) * sample_period, result.threshold, color='green', linewidth=1)
    ax3.plot(np.arange(len(result.correlation)) * sample_period, result.correlation, linewidth=1)

    if result.detected_signal:
        xs, ys = zip(*result.detected_signal)
        ax3.scatter(xs * sample_period, ys,
                    marker='x',
                    color='yellow')

    ax3.set_xlabel('Time (s)')

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

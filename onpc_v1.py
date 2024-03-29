from collections import deque
import copy
from functools import lru_cache
import itertools
import json
import logging
from math import log10
import os
from pathlib import Path
import pickle
import random
import sys
import time

import attr
import numpy as np
import pandas as pd
from pint import UnitRegistry
from scipy import signal
import scipy.io as sio
from scipy.stats import norm

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOGGER = None
DATA_SIZE = 0

index = 0
correlation = []
correlation_threshold = []
correlation_threshold_high = []
correlation_threshold_low = []
events = []

ureg = UnitRegistry()

# Time:
# old                now
# <-------------------|

@attr.s
class Result(object):
    original_samples = attr.ib()
    samples = attr.ib()
    sample_period = attr.ib()
    detected_signal = attr.ib()
    correlation = attr.ib()
    correlation_threshold_high = attr.ib()
    config = attr.ib()
    param = None
    metadata = None

def filter_nearby_transmitters(samples):
    mean = samples.mean()
    std = samples.std()
    threshold = np.percentile(samples, 10)

    # threshold =  -99.25
    # std = 13.7438368521

    # threshold = -99.4166666667
    # std = 20.0348352445

    print("threshold", threshold)
    print("std", std)

    filtered_samples = norm.cdf(samples, loc=threshold, scale=std * .7)
    filtered_samples_shifted = (filtered_samples * 4) - 3

    return filtered_samples_shifted


def correlate_samples(samples, symbol, sample_factor):
    """Multiples a buffer of samples by a symbol"""
    global index

    # Create buffer
    sample_buffer = np.zeros(len(symbol))

    LOGGER.debug("COR Sample buffer length: %s", len(sample_buffer))

    # Build up buffer with samples until it is not empty
    for i in range(len(sample_buffer) - 1):
        sample_buffer = np.roll(sample_buffer, -1)
        sample_buffer[-1] = next(samples)

        index += 1
        correlation.append(np.nan)
        correlation_threshold_high.append(np.nan)
        correlation_threshold_low.append(np.nan)

    # Correlate samples with symbol
    for sample in samples:
        sample_buffer = np.roll(sample_buffer, -1)
        sample_buffer[-1] = sample

        LOGGER.debug("COR Sample buffer: %s", sample_buffer)
        LOGGER.debug("COR Symbol: %s", symbol)

        result = (sample_buffer * symbol).sum()
        LOGGER.debug("COR Correlation: %s", result)

        symbol_size = len(symbol) / sample_factor
        num_ones = symbol_size // 2 * sample_factor

        # Divide by the number of ones
        result = result / np.sum(symbol == 1)

        yield result


def low_pass_filter(samples, filter_size=20):
    global index

    # Create buffer
    sample_buffer = np.zeros(filter_size)

    # Build up buffer with samples until it is not empty
    for i in range(len(sample_buffer) - 1):
        sample_buffer = np.roll(sample_buffer, -1)
        sample_buffer[-1] = next(samples)

        index += 1
        correlation.append(np.nan)
        correlation_threshold_high.append(np.nan)
        correlation_threshold_low.append(np.nan)

    # Run low pass filter
    for sample in samples:
        sample_buffer = np.roll(sample_buffer, -1)
        sample_buffer[-1] = sample
        yield sample_buffer.mean()
        # yield np.median(sample_buffer)


def detect_symbols(correlations, corr_buffer_size, symbol_size, sync_word_size, corr_std_factor):
    """Detects if a correlation is greater than some threshold"""
    global index

    # Create buffer
    corr_buffer = np.zeros(corr_buffer_size)

    # Build up buffer with correlations until it is not empty
    for i in range(len(corr_buffer) - 1):
        corr_buffer = np.roll(corr_buffer, -1)
        corr_buffer[-1] = next(correlations)
        correlation.append(corr_buffer[-1])
        correlation_threshold_high.append(np.nan)
        correlation_threshold_low.append(np.nan)
        index += 1

    for corr in correlations:
        index += 1
        corr_buffer = np.roll(corr_buffer, -1)
        corr_buffer[-1] = corr

        corr_threshold_high = corr_buffer[:-100].mean() + corr_std_factor * corr_buffer[:-100].std()
        corr_threshold_low = corr_buffer[:-100].mean() - corr_std_factor * corr_buffer[:-100].std()

        LOGGER.debug("DETECT Corr Buffer: \n%s", corr_buffer)
        LOGGER.debug("DETECT Mean: %s", corr_buffer.mean())
        LOGGER.debug("DETECT Std: %s", corr_buffer.std())
        LOGGER.debug("DETECT High threshold: %s", corr_threshold_high)
        LOGGER.debug("DETECT Low threshold: %s", corr_threshold_low)
        LOGGER.debug("DETECT value: %s", corr_buffer[-1])
        LOGGER.debug("DETECT index: %s", index)

        correlation.append(corr_buffer[-1])
        correlation_threshold_high.append(corr_threshold_high)
        correlation_threshold_low.append(corr_threshold_low)
        # bits.append(None)

        # If the correlation value is higher than one of the thresholds
        # We don't need to check the low threshold because we are only sending a 1
        if corr_buffer[-1] > corr_threshold_high:
            events.append((index, corr_buffer[-1], 'detected_peak'))
            LOGGER.debug("DETECT Found a symbol!!!!!!")
            yield 1


def bit_decision(symbols):
    """Converts a correlation value into a bit"""
    for symbol in symbols:
        try:
            if symbol > 0:
                LOGGER.debug("BIT DECISION: 1")
                yield 1
            else:
                LOGGER.debug("BIT DECISION: 0")
                yield 0
        except Exception as e:
            # Nothing we can do with an error, so we just pass it up
            symbols.throw(e)
            yield None  # Needed to restart generator that threw exception


def get_packet(bits, sync_word, fuzz):
    bit_buffer = np.ones(len(sync_word), dtype=np.int64) * -1
    bits_since_sync = 0

    for bit in bits:
        bit_buffer = np.roll(bit_buffer, -1)
        bit_buffer[-1] = bit

        LOGGER.debug("SYNC_WORD new bit: %s\n\n", bit_buffer)

        LOGGER.debug("SYNC_WORD %s =? %s", sync_word[fuzz:], bit_buffer[fuzz:])
        if (sync_word[fuzz:] == bit_buffer[fuzz:]).all():
            LOGGER.debug("SYNC_WORD Found sync word in bit buffer!!!!!")
            yield [next(bits) for _ in range(DATA_SIZE)]
            bits_since_sync = 0
        else:
            LOGGER.debug("SYNC_WORD Sync word is not in bit buffer")
            bits_since_sync += 1

        LOGGER.debug("%s >= %s", bits_since_sync, len(sync_word))
        if bits_since_sync >= len(sync_word):
            # We are receiving bits, but we haven't received the sync word yet.
            # Give up and go back to looking for symbols.
            LOGGER.debug("SYNC_WORD Giving up trying to find sync word")
            bits_since_sync = 0
            bits.throw(ValueError)


def decode_signal(samples, symbol, sample_factor, sync_word, sync_word_fuzz,
                  corr_buffer_size, corr_std_factor, low_pass_filter_size):
    # samples = list(samples)
    # LOGGER.debug("DECODE Samples: %s\n", samples)

    corr = low_pass_filter(correlate_samples(iter(samples), symbol, sample_factor), low_pass_filter_size)
    symbols = detect_symbols(corr, corr_buffer_size, len(symbol), len(sync_word), corr_std_factor)
    bits = bit_decision(symbols)
    packets = get_packet(bits, sync_word, sync_word_fuzz)

    for packet in packets:
        yield True


def downsample(samples, amount=1):
    i = 0
    while True:
        item = next(samples)

        if i % amount == 0:
            yield item


def bits(number):
    number = bin(number)
    for digit in number[2:]:
        yield int(digit)



def mw_to_dbm(mW):
    """This function converts a power given in mW to a power given in dBm."""
    return 10.*log10(mW)

def dbm_to_mw(dBm):
    """This function converts a power given in dBm to a power given in mW."""
    return 10**((dBm)/10.)


def create_samples(symbol, sync_word, data, rng, sample_factor, samples_per_transmission,
                   signal_params, noise_params, quantization, cs_params):
    """
    In this function, there is no concept of time. Everything is relative to
    the transmit rate / sample rate.
    """
    LOGGER.debug("Signal parameters: %s", signal_params)
    LOGGER.debug("Noise parameters: %s", noise_params)

    def signal_sample():
        noise = dbm_to_mw(rng.gauss(**noise_params))
        signal = dbm_to_mw(signal_params)
        total = noise + signal
        return quantize(mw_to_dbm(total))

    def noise_sample():
        return quantize(rng.gauss(**noise_params))

    def noise(num):
        yield from (noise_sample() for _ in range(num))

    def quantize(sample):
        return round(sample, quantization)

    slot_size = sample_factor
    transmit_slots = samples_per_transmission

    yield from noise(5 * len(symbol) * sample_factor)  # Add some noise padding
    slots_left = 0
    for bit in itertools.chain(sync_word, data):
        for chip in symbol:
            slots_left += slot_size
            if bit == chip:
                if slots_left < slot_size:
                    # The previous transmission crossed slots -- we can't transmit!
                    LOGGER.warning("Unable to transmit because previous transmission!")
                    yield from (noise_sample() for _ in range(slots_left))
                    slots_left = 0
                    continue

                # Carrier sensing
                tx_slots_delay = int(round(abs(rng.gauss(**cs_params))))  # TODO: Fix this! Is CS Gaussian?
                yield from (noise_sample() for _ in range(tx_slots_delay))

                # Transmit
                yield from (signal_sample() for _ in range(transmit_slots))

                # Figure out how much time is left in slot
                slots_left = slot_size - tx_slots_delay - transmit_slots

                if slots_left > 0:
                    yield from (noise_sample() for _ in range(slots_left))
                    slots_left = 0

            else:
                yield from (noise_sample() for _ in range(slot_size))
                slots_left = 0
    yield from noise(5 * len(symbol) * sample_factor)  # Add some noise padding


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

        with open('{}npy'.format(file_name[:-3]), 'wb') as f:
           np.save(f, new_data)

        return new_data

def resample(samples, sample_time, chip_time, sample_factor):
    sample_time = sample_time * ureg.s  # Convert to seconds
    # See how many samples should have been collected during sample time
    num_samples = (sample_time / (chip_time / sample_factor)).to_base_units()
    num_samples = round(num_samples.magnitude)

    new_samples = signal.resample(samples, num_samples)
    new_samples[new_samples < samples.min()] = samples.min()

    sample_period = sample_time / len(new_samples)
    return new_samples, sample_period


def get_samples_from_wl_file(sample_file, sample_factor):
    if sample_file['type'] != 'wl':
        LOGGER.error("Unknown sample file type: %s", sample_file['type'])
        exit()

    with open(sample_file['name']) as f:
        data = json.load(f)

    print(data['location'])
    print(data['description'])

    chip_time = ureg(data['chip_time'])
    antenna1, antenna2, antenna3 = map(pd.Series, zip(*data['samples']))
    # print("Number of NaN values:", np.isnan(antenna1).sum())

    antenna1 = antenna1.interpolate()
    antenna2 = antenna2.interpolate()
    antenna3 = antenna3.interpolate()

    samples = (antenna1 + antenna2 + antenna3) / 3
    # samples = antenna1

    LOGGER.warn("Reading sample file:")
    LOGGER.warn("\tRun time: %s s (%s ms)", data['run_time'], 1000 * data['run_time'] / len(samples))
    LOGGER.warn("\tChip time: %s", chip_time)
    LOGGER.warn("\tSample factor: %s", sample_factor)

    new_samples, sample_period = resample(samples,
                                          data['run_time'],
                                          chip_time,
                                          sample_factor)

    LOGGER.warn("\tNew sample period: %s", sample_period.to(ureg.ms))

    return new_samples, sample_period


def get_samples_from_spectrum_file(sample_file, src_period, dst_period):
    if sample_file['type'] != 'spectrum analyzer':
        LOGGER.error("Unknown sample file type: %s", sample_file['type'])
        exit()

    data = load_data(sample_file['name'])

    if len(data) > 1:
        LOGGER.warning("File contains multiple captures. Selecting the first one.")
    data = data[0]

    factor = (dst_period / src_period).to_base_units()
    if factor % 1 != 0:
        LOGGER.error("Source and destination sample period must be evenly divisible: %s / %s = %s",
                     dst_period,
                     src_period,
                     factor)
        exit()
    factor = int(factor)

    # Get magnitude and convert to power
    power_data = np.abs(data) ** 2

    # Downsample source to destination
    power_data =  np.array([power_data[i:i+factor].mean()
                            for i in range(0, len(power_data), factor)])

    # Convert to dBm
    power_data = 10.*np.log10(power_data)

    return power_data


def unfreeze(s):
    if isinstance(s, frozenset):
        return {key: unfreeze(value) for key, value in s}
    return s

def freeze(d):
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d

def get_hash(params):
    return hash(freeze(params))


def main(id_, folder, params):
    # hash_file = os.path.join('cache', f'{get_hash(params)}.pkl')

    # try:
    #     with open(hash_file, 'rb') as f:
    #         return pickle.load(f)
    # except FileNotFoundError:
    #     print("Unable to find in cache. Running...")

    # Set up logging
    global LOGGER
    LOGGER = logging.getLogger("{}".format(id_))
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    if folder is not None:
        folder_name = '{}/{:04}.log'.format(folder, id_)
    else:
        folder_name = 'out.log'

    if params.get('logging_output', True):  # Give the full log to a file
        handler = logging.FileHandler(folder_name, mode='w')
        LOGGER.setLevel(logging.DEBUG)
    elif params.get('command_line', False):  # Give only important stuff to the terminal
        handler = logging.StreamHandler()
        LOGGER.setLevel(logging.WARN)
    else:  # Give only important stuff to a file
        handler = logging.FileHandler(folder_name, mode='w')
        LOGGER.setLevel(logging.WARN)

    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    # Take care of symbol generation
    if 'max_len_seq' in params:
        LOGGER.debug("Max Length Sequence: %s", params['max_len_seq'])
        symbol, state = signal.max_len_seq(params['max_len_seq'])
    elif 'symbol' in params:
        symbol = np.array(params['symbol'])
    LOGGER.debug("Symbol: %s (%s)", symbol, len(symbol))

    symbol = symbol * 2 - 1

    # Sync word parameters
    sync_word = np.array(params['sync_word'])
    sync_word_fuzz = params['sync_word_fuzz']

    # Data
    # data = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])
    data = np.array([])

    if 'sample_file' in params and params['sample_file']['type'] == 'wl':
        # Don't bother with dealing with the sample period.
        # We will get that from the file.

        sample_factor = params['sample_factor']

        samples, sample_period = get_samples_from_wl_file(params['sample_file'],
                                                          sample_factor)
    else:
        # Sampling timing parameters
        sample_period = ureg(params['destination_sample_period'])
        chip_time = ureg(params['chip_time'])
        if chip_time < sample_period:
            LOGGER.error("Sample period must be smaller than chip time")
            exit()

        sample_factor = chip_time / sample_period
        if sample_factor % 1 != 0:
            LOGGER.error("Sample period and on/off period must be evenly divisible: %s / %s = %s",
                         chip_time,
                         sample_period,
                         sample_factor)
            exit()
        sample_factor = int(sample_factor)

        # Generate the samples (simulated or through a file)
        if 'sample_file' in params:
            # Resolve source and destination sample rates
            source_sample_period = ureg(params['source_sample_period'])

            samples = get_samples_from_spectrum_file(params['sample_file'],
                                            source_sample_period,
                                            sample_period)
        else:
            seed = params.get('seed', random.randrange(sys.maxsize))
            LOGGER.warning("Seed is: %s", seed)
            rng = random.Random(seed)

            # Transmission timing parameters
            transmission_time = ureg(params['transmit_time'])
            samples_per_transmission =  transmission_time / sample_period
            if samples_per_transmission % 1 != 0:
                LOGGER.error("Sample period and transmission time must be evenly divisible: %s / %s = %s",
                             transmission_time,
                             sample_period,
                             samples_per_transmission)
                exit()
            samples_per_transmission = int(samples_per_transmission)

            # Take care of carrier sensing
            if params.get('carrier_sensing'):
                cs = params['carrier_sensing']
                cs['sigma'] = (ureg(cs['sigma']) / sample_period).magnitude
                cs['mu'] = (ureg(cs['mu']) / sample_period).magnitude

            samples = create_samples(symbol, sync_word, data, rng, sample_factor, samples_per_transmission,
                                     signal_params=params['signal'],
                                     noise_params=params['noise'],
                                     quantization=params.get('quantization'),
                                     cs_params=params.get('carrier_sensing', {'mu': 0, 'sigma': 0}))
            samples = list(samples)

    LOGGER.debug("Starting...")
    ### TODO: Temporary!!!
    # samples = samples[int(0 / sample_period.magnitude):int(150 / sample_period.magnitude)]

    original_samples = samples.copy()

    if params.get('filter', True):
        # print("Filtering nearby transmitters")
        samples = filter_nearby_transmitters(samples)

    result = decode_signal(samples,
                           np.repeat(symbol, sample_factor),
                           sample_factor,
                           sync_word,
                           sync_word_fuzz,
                           params['correlation_buffer_size'],
                           params['correlation_std_threshold'],
                           params['low_pass_filter_size'])
    result = list(result)
    LOGGER.debug("Done...")


    def calc_threshold(data):
        ignore_values = 100

        data = data[:-ignore_values]
        return data.std() * params['correlation_std_threshold'] + data.mean()

    def onpc(samples):
        corr_samples = np.correlate(np.repeat(symbol, sample_factor), samples)
        corr_samples = np.flip(corr_samples, 0)
        corr_samples = pd.Series(corr_samples)

        corr_samples = corr_samples / np.sum(symbol == 1)

        # Low pass filter
        corr_samples = corr_samples.rolling(window=30).mean()

        threshold = corr_samples.rolling(window=600).apply(calc_threshold)

        empty = np.empty(len(symbol) * sample_factor - 1)
        empty[:] = np.nan

        corr_samples = np.concatenate((empty, np.array(corr_samples)))
        threshold = np.concatenate((empty, np.array(threshold)))

        peak_indexes = np.where(corr_samples > threshold)[0]
        ys = corr_samples[peak_indexes]
        xs = peak_indexes

        return corr_samples, threshold, zip(xs, ys)

    test_samples, test_threshold, peaks  = onpc(samples)
    test_samples_original, test_threshold_original, peaks_original  = onpc(original_samples)

    global index
    global correlation
    global correlation_threshold_high
    global events

    if params.get('graph', False) or params.get('interactive', False):
        expected = []
        for bit in sync_word:
            if bit == 0:
                expected.append(np.repeat(symbol, sample_factor) ^ 1)
            else:
                expected.append(np.repeat(symbol, sample_factor))
        expected = np.concatenate(expected)

        if not params.get('interactive', False):
            import matplotlib
            matplotlib.use('agg')

        import matplotlib.pyplot as plt

        fig, (ax0, ax1, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8,6), sharex=True)

        ax0.plot(np.arange(len(original_samples)) * sample_period, original_samples, '.', markersize=.7)

        # Plot the raw samples
        ax1.plot(np.arange(len(samples)) * sample_period, samples, '.', markersize=.7)

        # ax1.axhline(median, color='red')
        # ax1.axhline(mean, color='green')
        # ax1.axhline(mean + 2*std, color='green')

        # # Plot the expected symbol sequence
        # ax2.plot(np.arange(len(expected)) * sample_period, expected)
        # ax2.set_xlim(ax1.get_xlim())
        # ax2.set_xlabel('Time (ms)')

        scatter_data = [(x, y) for x, y, event in events if event == 'detected_peak_2']
        if scatter_data:
            x, y = zip(*scatter_data)
            ax3.scatter(x * sample_period, y,
                        marker='x',
                        color='grey')

        scatter_data = [(x, y) for x, y, event in events if event == 'detected_peak']
        if scatter_data:
            x, y = zip(*scatter_data)
            ax3.scatter(x * sample_period, y,
                        marker='x',
                        color='yellow')

        scatter_data = [(x, y) for x, y, event in events if event == 'detected_bit']
        if scatter_data:
            x, y = zip(*scatter_data)
            ax3.scatter(x * sample_period, y,
                        marker='x',
                        color='red')

        ax3.plot(np.arange(len(correlation_threshold_high)) * sample_period, correlation_threshold_high, color='green', linewidth=1)
        # ax3.plot(correlation_threshold_low, color='orange', label='lower threshold')
        ax3.plot(np.arange(len(correlation)) * sample_period, correlation, label='correlation', linewidth=1)

        ax3.set_xlim(ax1.get_xlim())
        ax3.set_xlabel('Time (s)')

        ax4.plot(np.arange(len(test_threshold)) * sample_period, test_threshold, color='green', linewidth=1)
        ax4.plot(np.arange(len(test_samples)) * sample_period, test_samples, linewidth=1)
        xs, ys = zip(*peaks)
        ax4.scatter(xs * sample_period, ys,
                    marker='x',
                    color='yellow')

        ax5.plot(np.arange(len(test_threshold_original)) * sample_period, test_threshold_original, color='green', linewidth=1)
        ax5.plot(np.arange(len(test_samples_original)) * sample_period, test_samples_original, linewidth=1)
        xs, ys = zip(*peaks_original)
        ax5.scatter(xs * sample_period, ys,
                    marker='x',
                    color='yellow')

        # plt.legend()
        plt.tight_layout()

        if params.get('graph', False):
            if 'sample_file' not in params:
                name = 'simulated'
            else:
                name = Path(params['sample_file']['name']).stem

            plt.savefig(f'decoded_signal-{id_}-{name}.pdf')
            plt.savefig(f'decoded_signal-{id_}-{name}.png', dpi=600)

        if params.get('interactive', False):
            plt.show()

    result = Result(original_samples=original_samples,
                    samples=samples,
                    sample_period=sample_period,
                    correlation=copy.deepcopy(correlation),
                    correlation_threshold_high=copy.deepcopy(correlation_threshold_high),
                    detected_signal=copy.deepcopy(events),
                    config=params)

    # Clear out global state
    index = 0
    correlation = []
    correlation_threshold_high = []
    events = []


    # with open(hash_file, 'wb') as f:
    #     pickle.dump(result, f)


    return result

if __name__ == '__main__':
    import click
    import yaml

    @click.command()
    @click.argument('config_file', type=click.File('r'))
    @click.option('--log/--no-log', default=None)
    @click.option('--graph/--no-graph', default=False)
    @click.option('--interactive/--no-interactive', default=False)
    @click.option('--wl-sample-file')
    def cli(config_file, log, graph, interactive, wl_sample_file):
        config = yaml.load(config_file)
        config['command_line'] = True
        config['graph'] = graph
        config['interactive'] = interactive

        if wl_sample_file:
            config['sample_file']['name'] = wl_sample_file
            config['sample_file']['type'] = 'wl'

        # If log has been set, then overwrite config
        if log is not None:
            config['logging_output'] = log

        result = main(id_=__name__,
                      folder=None,
                      params=config)

        if result.detected_signal:
            print(f"Success!!! ({result.detected_signal})")
        else:
            print("Didn't find data")


    cli()





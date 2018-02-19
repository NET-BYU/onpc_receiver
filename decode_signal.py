from collections import deque
from itertools import chain
import logging
from math import log10
import random
import sys
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from pint import UnitRegistry
from scipy import signal
import scipy.io as sio

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOGGER = None
CORR_BUFFER_SIZE = 75
CORR_STD_FACTOR = 5
DATA_SIZE = 0

signals = []
correlation = []
correlation_threshold = []
correlation_threshold_high = []
correlation_threshold_low = []
events = []

ureg = UnitRegistry()

# Time:
# old                now
# <-------------------|


def correlate_samples(samples, symbol):
    """Multiples a buffer of samples by a symbol"""

    # Create buffer
    sample_buffer = np.zeros(len(symbol))

    # Build up buffer with samples until it is not empty
    for i in range(len(sample_buffer) - 1):
        sample_buffer = np.roll(sample_buffer, -1)
        sample_buffer[-1] = next(samples)

    # Correlate samples with symbol
    for sample in samples:
        sample_buffer = np.roll(sample_buffer, -1)
        sample_buffer[-1] = sample

        LOGGER.debug("COR Sample buffer: %s", sample_buffer)
        LOGGER.debug("COR Symbol: %s", symbol)

        result = (sample_buffer * symbol).sum()
        LOGGER.debug("COR Correlation: %s", result)

        yield result


def detect_symbols(correlations, symbol_size, sync_word_size):
    """Detects if a correlation is greater than some threshold"""

    index = 0
    peak = None
    peak_index = None
    peak_mean = None

    # Create buffer
    corr_buffer = np.zeros(CORR_BUFFER_SIZE)

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

        corr_threshold_high = corr_buffer.mean() + CORR_STD_FACTOR * corr_buffer.std()
        corr_threshold_low = corr_buffer.mean() - CORR_STD_FACTOR * corr_buffer.std()

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

        # Only if the peak is maintained for a whole symbol size will it be considered a symbol
        if peak_index is not None and index >= (peak_index + symbol_size - 1):
            LOGGER.debug("DETECT Found a bit: %s !!!!!!!!!!", peak)
            events.append((peak_index, peak, 'detected_bit'))
            yield peak
            peak = None
            peak_index = None
            peak_mean = None

            LOGGER.debug("DETECT Looking for 2 bit of sync word")

            # Since we have already looked ahead a whole symbol_size, the next
            # correlation should be our symbol.
            corr = next(correlations)
            index += 1
            correlation.append(corr)
            correlation_threshold_high.append(np.nan)
            correlation_threshold_low.append(np.nan)
            events.append((index, corr, 'detected_bit'))
            LOGGER.debug("DETECT Bit 2: %s", corr)
            yield corr

            try:
                for i in range(sync_word_size - 2):
                    LOGGER.debug("DETECT Looking for %s bit of sync word", i + 2)
                    for _ in range(symbol_size):
                        corr = next(correlations)
                        index += 1
                        correlation.append(corr)
                        correlation_threshold_high.append(np.nan)
                        correlation_threshold_low.append(np.nan)
                    events.append((index, corr, 'detected_bit'))
                    LOGGER.debug("DETECT Bit %s: %s", i + 3, corr)
                    yield corr

                # Found the sync word, keep going!
                for i in range(DATA_SIZE):
                    LOGGER.debug("DETECT Looking for %s bit of data", i + 1)
                    for _ in range(symbol_size):
                        corr = next(correlations)
                        index += 1
                        correlation.append(corr)
                        correlation_threshold_high.append(np.nan)
                        correlation_threshold_low.append(np.nan)
                    events.append((index, corr, 'detected_bit'))
                    LOGGER.debug("DETECT Bit %s: %s", i + 1, corr)
                    yield corr
            except ValueError:
                # Our consumer is letting us know that it can't find the sync word.
                # Must have been a false alarm, go back to looking for symbols.
                LOGGER.debug("DETECT Give up trying to find bits")
                yield None  # Needed to restart generator that threw exception
                continue

        # If the correlation value is higher than one of the thresholds
        if corr_threshold_low > corr_buffer[-1] or corr_buffer[-1] > corr_threshold_high:
            corr_mean = corr_buffer.mean()

            # If we don't have a peak or if the peak is greater than the current peak
            if peak is None or abs(corr_buffer[-1] - corr_mean) > abs(peak - corr_mean):
                peak = corr_buffer[-1]
                peak_index = index
                peak_mean = corr_mean
                events.append((peak_index, peak, 'detected_peak'))
                LOGGER.debug("DETECT New peak %s !!!!!!", peak)

        # If we already have a peak and the current correlation value is greater than that peak
        if peak is not None and abs(corr_buffer[-1] - peak_mean) > abs(peak - peak_mean):
            peak = corr_buffer[-1]
            peak_index = index
            events.append((peak_index, peak, 'detected_peak_2'))
            LOGGER.debug("DETECT New peak %s !!!!!!", peak)


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


def decode_signal(samples, symbol, sync_word, sync_word_fuzz):
    # samples = list(samples)
    # LOGGER.debug("DECODE Samples: %s\n", samples)

    corr = correlate_samples(iter(samples), symbol)
    symbols = detect_symbols(corr, len(symbol), len(sync_word))
    bits = bit_decision(symbols)
    packets = get_packet(bits, sync_word, sync_word_fuzz)

    for packet in packets:
        return True


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


def create_samples(symbol, sync_word, data, rng, signal_params,
                   noise_params, quantization_params):
    LOGGER.debug("Signal parameters: %s", signal_params)
    LOGGER.debug("Noise parameters: %s", noise_params)

    def signal_sample():
        signals.append(1)
        noise = dbm_to_mw(rng.gauss(**noise_params))
        signal = dbm_to_mw(signal_params)
        total = noise + signal
        return quantize(mw_to_dbm(total))

    def noise_sample():
        signals.append(0)
        return quantize(rng.gauss(**noise_params))

    def noise(num):
        yield from (noise_sample() for _ in range(num))

    def quantize(sample):
        return round(sample, quantization_params)

    def one():
        yield from (signal_sample() if i == 1 else noise_sample() for i in symbol)

    def zero():
        yield from (noise_sample() if i == 1 else signal_sample() for i in symbol)

    yield from noise(5 * len(symbol))
    for bit in chain(sync_word, data):
        if bit:
            yield from one()
        else:
            yield from zero()
    yield from noise(5 * len(symbol))


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


def get_samples_from_file(file_name, src_period, dst_period):
    data = load_data(file_name)

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

    power_data = np.abs(data) ** 2  # Get magnitude and convert to power


    # Downsample source to destination
    power_data =  np.array([power_data[i:i+factor].mean()
                            for i in range(0, len(power_data), factor)])

    # Convert to dBm
    power_data = 10.*np.log10(power_data)

    return power_data


def main(id_, folder, params, sample_file=None):
    # Set up logging
    global LOGGER
    LOGGER = logging.getLogger("{}".format(id_))
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    if folder is not None:
        folder_name = '{}/{:04}.log'.format(folder, id_)
    else:
        folder_name = 'out.log'

    if params.get('logging_output', True):
        handler = logging.FileHandler(folder_name, mode='w')
        LOGGER.setLevel(logging.DEBUG)
    else:
        handler = logging.StreamHandler()
        LOGGER.setLevel(logging.WARN)

    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    # Take care of symbol generation
    if 'max_len_seq' in params:
        LOGGER.debug("Max Length Sequence: %s", params['max_len_seq'])
        symbol, state = signal.max_len_seq(params['max_len_seq'])
        symbol = symbol * 2 - 1
    elif 'symbol' in params:
        symbol = np.array(params['symbol'])
    LOGGER.debug("Symbol: %s", symbol)

    # Sync word parameters
    sync_word = np.array(params['sync_word'])
    sync_word_fuzz = params['sync_word_fuzz']

    # Data
    # data = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])
    data = np.array([])

    # Sampling and transmission timing parameters
    sample_period = ureg(params['destination_sample_period'])
    transmission_period = ureg(params['transmission_period'])
    if transmission_period < sample_period:
        LOGGER.error("Sample period must be smaller than transmission period")
        exit()

    factor = transmission_period / sample_period
    if factor % 1 != 0:
        LOGGER.error("Sample period and transmission period must be evenly divisible: %s / %s = %s",
                     transmission_period,
                     sample_period,
                     factor)
        exit()
    factor = int(factor)

    # Adjust symbol to match transmission and sample periods
    symbol = np.repeat(symbol, factor)

    # Generate the samples (simulated or through a file)
    if sample_file is None:
        seed = params.get('seed', random.randrange(sys.maxsize))
        LOGGER.warning("Seed is: %s", seed)
        rng = random.Random(seed)

        samples = create_samples(symbol, sync_word, data, rng,
                                 signal_params=params['signal'],
                                 noise_params=params['noise'],
                                 quantization_params=params.get('quantization', None))
        samples = list(samples)
    else:
        # Resolve source and destination sample rates
        source_sample_period = ureg(params['source_sample_period'])

        samples = get_samples_from_file(sample_file,
                                        source_sample_period,
                                        sample_period)


    LOGGER.debug("Starting...")
    result = decode_signal(samples, symbol, sync_word, sync_word_fuzz)
    LOGGER.debug("Done...")

    if params.get('graph', False):
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(211)
        ax1.plot(samples)

        ax2 = fig.add_subplot(212)


        ax2.scatter(*zip(*[(x, y) for x, y, event in events if event == 'detected_peak_2']),
                    marker='x',
                    color='grey')

        ax2.scatter(*zip(*[(x, y) for x, y, event in events if event == 'detected_peak']),
                    marker='x',
                    color='yellow')

        ax2.scatter(*zip(*[(x, y) for x, y, event in events if event == 'detected_bit']),
                    marker='x',
                    color='red')

        ax2.plot(correlation_threshold_high, color='green', label='upper threshold')
        ax2.plot(correlation_threshold_low, color='orange', label='lower threshold')
        ax2.plot(correlation, label='correlation')

        ax2.set_xlim(0, len(samples))

        plt.legend()
        plt.tight_layout()
        plt.savefig('decoded_signal.pdf')

    #     fig = plt.figure(figsize=(8,3))
    #     ax3 = fig.add_subplot(111)
    #     ax3.plot(np.array(correlation))
    #     plt.tight_layout()
    #     plt.savefig('corr.pdf')

    return result


if __name__ == '__main__':
    import click
    import yaml

    @click.command()
    @click.argument('config_file', type=click.File('r'))
    @click.option('--sample_file', default=None)
    @click.option('--log/--no-log', default=None)
    @click.option('--graph/--no-graph', default=False)
    def cli(config_file, sample_file, log, graph):
        config = yaml.load(config_file)
        config['command_line'] = True
        config['graph'] = graph

        # If log has been set, then overwrite config
        if log is not None:
            config['logging_output'] = log

        result = main(id_=__name__,
                      folder=None,
                      params=config,
                      sample_file=sample_file)

        if result:
            print("Success!!!")
        else:
            print("Didn't find data")


    cli()





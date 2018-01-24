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
from scipy import signal

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOGGER = None
SYMBOL_SIZE = 15
CORR_BUFFER_SIZE = 75
CORR_STD_FACTOR = 3
SYNC_WORD = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0])
SYNC_WORD_FUZZ = 3
DATA_SIZE = 0

signals = []
samps = []
correlation = []
correlation_threshold = []
correlation_threshold_high = []
correlation_threshold_low = []
bits = []

# Time:
# old                now
# <------------------->


def correlate_samples(samples, symbol):
    """Multiples a buffer of samples by a symbol"""

    # Create buffer
    sample_buffer = np.zeros(SYMBOL_SIZE)

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


def detect_symbols(correlations):
    """Detects if a correlation is greater than some threshold"""

    index = 0
    peak = None
    peak_index = None

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
        if peak_index is not None and index >= (peak_index + SYMBOL_SIZE - 1):
            LOGGER.debug("DETECT Found a bit: %s !!!!!!!!!!", peak)
            yield peak
            peak = None
            peak_index = None

            LOGGER.debug("DETECT Looking for 2 bit of sync word")

            # Since we have already looked ahead a whole SYMBOL_SIZE, the next
            # correlation should be our symbol.
            corr = next(correlations)
            index += 1
            correlation.append(corr)
            correlation_threshold_high.append(np.nan)
            correlation_threshold_low.append(np.nan)
            LOGGER.debug("DETECT Bit 2: %s", corr)
            yield corr

            try:
                for i in range(len(SYNC_WORD) - 2):
                    LOGGER.debug("DETECT Looking for %s bit of sync word", i + 2)
                    for _ in range(SYMBOL_SIZE):
                        corr = next(correlations)
                        index += 1
                        correlation.append(corr)
                        correlation_threshold_high.append(np.nan)
                        correlation_threshold_low.append(np.nan)
                    LOGGER.debug("DETECT Bit %s: %s", i + 3, corr)
                    yield corr

                # Found the sync word, keep going!
                for i in range(DATA_SIZE):
                    LOGGER.debug("DETECT Looking for %s bit of data", i + 1)
                    for _ in range(SYMBOL_SIZE):
                        corr = next(correlations)
                        index += 1
                        correlation.append(corr)
                        correlation_threshold_high.append(np.nan)
                        correlation_threshold_low.append(np.nan)
                    LOGGER.debug("DETECT Bit %s: %s", i + 1, corr)
                    yield corr
            except ValueError:
                # Our consumer is letting us know that it can't find the sync word.
                # Must have been a false alarm, go back to looking for symbols.
                LOGGER.debug("DETECT Give up trying to find bits")
                continue

        if corr_threshold_low > corr_buffer[-1] or corr_buffer[-1] > corr_threshold_high:
            corr_mean = corr_buffer.mean()

            if peak is None or abs(corr_buffer[-1] - corr_mean) > abs(peak - corr_mean):
                peak = corr_buffer[-1]
                peak_index = index
                LOGGER.debug("DETECT New peak %s !!!!!!", peak)


def bit_decision(symbols):
    """Converts a correlation value into a bit"""
    for symbol in symbols:
        try:
            if symbol > 0:
                yield 1
            else:
                yield 0
        except Exception as e:
            # Nothing we can do with an error, so we just pass it up
            symbols.throw(e)


def get_packet(bits):
    bit_buffer = np.ones(len(SYNC_WORD), dtype=np.int64) * -1
    bits_since_sync = 0

    for bit in bits:
        bit_buffer = np.roll(bit_buffer, -1)
        bit_buffer[-1] = bit

        LOGGER.debug("SYNC_WORD new bit: %s\n\n", bit_buffer)

        LOGGER.debug("SYNC_WORD %s =? %s", SYNC_WORD[SYNC_WORD_FUZZ:], bit_buffer[SYNC_WORD_FUZZ:])
        if (SYNC_WORD[SYNC_WORD_FUZZ:] == bit_buffer[SYNC_WORD_FUZZ:]).all():
            LOGGER.debug("SYNC_WORD Found sync word in bit buffer!!!!!")
            yield [next(bits) for _ in range(DATA_SIZE)]
            bits_since_sync = 0
        else:
            LOGGER.debug("SYNC_WORD Sync word is not in bit buffer")
            bits_since_sync += 1

        LOGGER.debug("%s >= %s", bits_since_sync, len(SYNC_WORD))
        if bits_since_sync >= len(SYNC_WORD):
            # We are receiving bits, but we haven't received the sync word yet.
            # Give up and go back to looking for symbols.
            LOGGER.debug("SYNC_WORD Giving up trying to find sync word")
            bits_since_sync = 0
            bits.throw(ValueError)


def decode_signal(samples, symbol):
    if len(symbol) != SYMBOL_SIZE:
        print("Symbol is wrong size! {} != {}".format(len(symbol), SYMBOL_SIZE))
        exit()

    samples = list(samples)
    LOGGER.debug("DECODE Samples: %s\n", samples)

    samples = graph_samples(samples)
    corr = correlate_samples(iter(samples), symbol)
    symbols = detect_symbols(corr)
    bits = bit_decision(symbols)
    packets = get_packet(bits)

    for packet in packets:
        LOGGER.debug("PACKET packet: %s", packet)
        LOGGER.warning("Result: True")


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


def create_samples(symbol, data, rng, signal_params, noise_params,
                   quantization_params):
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


    yield from noise(100)
    for bit in chain(SYNC_WORD, data):
        if bit:
            yield from one()
        else:
            yield from zero()
    yield from noise(100)


def graph_samples(samples):
    for sample in samples:
        samps.append(sample)
        yield sample


def main(id_, folder, params):
    # Set up logging
    global LOGGER
    LOGGER = logging.getLogger("{}".format(id_))
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    if folder is not None:
        folder_name = '{}/{:04}.log'.format(folder, id_)
    else:
        folder_name = 'out.log'


    handler = logging.FileHandler(folder_name, mode='w')
    handler.setFormatter(formatter)

    LOGGER.addHandler(handler)

    if params.get('logging_output', True):
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.WARN)

    if 'max_len_seq' in params:
        global SYMBOL_SIZE
        # Set up symbol
        LOGGER.debug("Max Length Sequence: %s", params['max_len_seq'])
        SYMBOL_SIZE = 2 ** params['max_len_seq'] - 1
        symbol, state = signal.max_len_seq(params['max_len_seq'])
        symbol = symbol * 2 - 1
        LOGGER.debug("Symbol: %s", symbol)

    # 0xDEAD
    # data = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])
    data = np.array([])

    seed = random.randrange(sys.maxsize)
    LOGGER.debug("Seed is: %s", seed)
    rng = random.Random(seed)

    samples = create_samples(symbol, data, rng,
                             signal_params=params.get('signal', [1, 2]),
                             noise_params=params.get('noise', [0, 1]),
                             quantization_params=params.get('quantization', None))
    samples = list(samples)


    decode_signal(samples, symbol)
    LOGGER.debug("Done...")

    if params.get('command_line', False):
        fig = plt.figure(figsize=(8,3))
        ax1 = fig.add_subplot(111)
        ax1.plot(np.linspace(0, len(samps) * 0.456, len(samps)), np.array(samps))
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Signal (dBm)")
        plt.tight_layout()
        plt.savefig('signal.pdf')

        fig = plt.figure(figsize=(8,3))
        ax3 = fig.add_subplot(111)
        ax3.plot(np.linspace(0, len(correlation) * 0.456, len(correlation)), np.array(correlation))
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Cross-correlation")
        plt.tight_layout()
        plt.savefig('corr.pdf')


if __name__ == '__main__':
    import yaml
    with open('experiment_config.yaml') as f:
        config = yaml.load(f)

    config['command_line'] = True

    main(__name__, None, config)

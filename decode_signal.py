from collections import deque
from itertools import chain
import logging
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
CORR_BUFFER_SIZE = SYMBOL_SIZE * 5
CORR_STD_FACTOR = 2
SYNC_WORD = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0])
SYNC_WORD_FUZZ = 3
DATA_SIZE = 16

signals = []
samps = []
correlation = []
correlation_threshold = []
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
        index += 1

    for corr in correlations:
        index += 1
        corr_buffer = np.roll(corr_buffer, -1)
        corr_buffer[-1] = corr

        corr_threshold = corr_buffer.mean() + CORR_STD_FACTOR * corr_buffer.std()

        LOGGER.debug("DETECT Corr Buffer: \n%s", corr_buffer)
        LOGGER.debug("DETECT Mean: %s", corr_buffer.mean())
        LOGGER.debug("DETECT Std: %s", corr_buffer.std())
        LOGGER.debug("DETECT threshold: %s", corr_threshold)
        LOGGER.debug("DETECT value: %s", corr_buffer[-1])
        LOGGER.debug("DETECT index: %s", index)

        # correlation.append(corr_buffer[-1])
        # correlation_threshold.append(corr_threshold)
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
            LOGGER.debug("DETECT Bit 2: %s", corr)
            yield corr

            try:
                for i in range(len(SYNC_WORD) - 2):
                    LOGGER.debug("DETECT Looking for %s bit of sync word", i + 2)
                    for _ in range(SYMBOL_SIZE):
                        corr = next(correlations)
                        index += 1
                    LOGGER.debug("DETECT Bit %s: %s", i + 2, corr)
                    yield corr

                # Found the sync word, keep going!
                for i in range(DATA_SIZE):
                    LOGGER.debug("DETECT Looking for %s bit of data", i + 1)
                    for _ in range(SYMBOL_SIZE):
                        corr = next(correlations)
                        index += 1
                    LOGGER.debug("DETECT Bit %s: %s", i + 1, corr)
                    yield corr
            except ValueError:
                # Our consumer is letting us know that it can't find the sync word.
                # Must have been a false alarm, go back to looking for symbols.
                LOGGER.debug("DETECT Give up trying to find bits")
                continue

        if abs(corr_buffer[-1]) > corr_threshold and (peak is None or abs(corr_buffer[-1]) > abs(peak)):
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

        if bits_since_sync > len(SYNC_WORD) + 1:
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
    corr = graph_corr(correlate_samples(iter(samples), symbol))
    symbols = detect_symbols(corr)
    bits = bit_decision(symbols)
    packets = get_packet(bits)

    for packet in packets:
        LOGGER.debug("PACKET packet: %s", packet)
        LOGGER.debug("Result: %s", (np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]) == np.array(packet)).all())


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


def create_samples(symbol, data, rng, signal_range, noise_range):
    LOGGER.debug("Signal Range: %s", signal_range)
    LOGGER.debug("Noise Range: %s", noise_range)

    def h():
        signals.append(1)
        return rng.randint(*signal_range)

    def n():
        signals.append(0)
        return rng.randint(*noise_range)

    def noise(num):
        yield from (n() for _ in range(num))

    def one():
        yield from (h() if i == 1 else n() for i in symbol)

    def zero():
        yield from (n() if i == 1 else h() for i in symbol)


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


def graph_corr(corrs):
    for corr in corrs:
        correlation.append(corr)
        yield corr


def main(id_, folder, params):
    # Set up logging
    global LOGGER
    LOGGER = logging.getLogger("{}".format(id_))
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    fileHandler = logging.FileHandler('{}/{:04}.log'.format(folder, id_), mode='w')
    fileHandler.setFormatter(formatter)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(fileHandler)


    if 'max_len_seq' in params:
        global SYMBOL_SIZE
        # Set up symbol
        LOGGER.debug("Max Length Sequence: %s", params['max_len_seq'])
        SYMBOL_SIZE = 2 ** params['max_len_seq'] - 1
        symbol, state = signal.max_len_seq(params['max_len_seq'])
        symbol = symbol * 2 - 1
        LOGGER.debug("Symbol: %s", symbol)

    # 0xDEAD
    data = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])

    seed = random.randrange(sys.maxsize)
    LOGGER.debug("Seed is: %s", seed)
    rng = random.Random(seed)

    samples = create_samples(symbol, data, rng,
                             signal_range=params.get('signal_range', [1, 2]),
                             noise_range=params.get('noise_range', [0, 1]))
    samples = list(samples)


    decode_signal(samples, symbol)
    LOGGER.debug("Done...")


    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)

    # ax1.plot(np.array(samps))
    # ax1.plot(np.array(signals))
    # ax2.plot(np.array(correlation))
    # # ax.plot(np.array(correlation_threshold))
    # # ax.plot(np.array(correlation_threshold) * -1)

    # plt.savefig('test.pdf')


if __name__ == '__main__':
    main()

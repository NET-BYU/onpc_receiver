import click
import numpy as np
import pyperclip
from scipy import signal

# http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr/10stages.txt

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_num_list(ctx, param, value):
    if value is None:
        return None
    try:
        if value[0] == '[' and value[-1] == ']':
            value = value[1:-1]

        return [int(x) for x in value.split(',') if x]
    except ValueError:
        raise click.BadParameter('List must only contain numbers')


@click.command(help="To find taps, visit http://in.ncu.edu.tw/ncume_ee/digilogi/prbs.htm."
                    " The number of bits corresponds to the number of stages. Using this list,"
                    " remove the first element. For example, in the table if the tap you select"
                    " for 10 stage is [10, 9, 7, 6], then remove the first value (10). In the"
                    " script you will enter 9,7,6.")
@click.argument('nbits', nargs=1, type=click.INT)
@click.argument('taps', callback=split_num_list)
@click.option('-t', '--type', default='c', type=click.Choice(['c', 'yaml', 'plain']), help='Type of output')
@click.option('-w', '--width', default=32, type=click.INT, help='Width of the lines produced')
@click.option('--clipboard/--no-clipboard', default=True)
def get_symbol(nbits, taps, width, type, clipboard):
    symbol, _ = signal.max_len_seq(nbits, taps=taps)
    symbol_size = len(symbol)
    str_symbol = str(list(symbol))

    # Remove brackets
    str_symbol = str_symbol[1:-1]

    # Break into lines
    lines = chunks(str_symbol.split(), width)

    # Join it all up with new lines between lines

    out = ""
    if type == 'c':
        symbol = '\n    '.join([' '.join(line) for line in lines])
        out += f"// taps: {taps}\n"
        out += f"// nbits: {nbits}\n"
        out += f"// size: {symbol_size}\n"
        out += f"uint8_t symbol[{symbol_size}] = {{\n\t{symbol}\n}};"
    elif type == 'yaml':
        symbol = '\n         '.join([' '.join(line) for line in lines])
        out += f"# taps: {taps}\n"
        out += f"# nbits: {nbits}\n"
        out += f"# size: {symbol_size}\n"
        out += f"symbol: [{symbol}]"
    elif type == 'plain':
        out += str(list(symbol))
    else:
        out += "Unknown type!"

    if not clipboard:
        print(out)
    else:
        pyperclip.copy(out)
        print("Result copied to clipboard")




if __name__ == '__main__':
    get_symbol()

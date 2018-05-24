import click
import numpy as np
import pyperclip
from scipy import signal


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_num_list(ctx, param, value):
    if value is None:
        return None
    try:
        return [int(x) for x in value.split(',') if x]
    except ValueError:
        raise click.BadParameter('List must only contain numbers')


@click.command()
@click.argument('size', nargs=1, type=click.INT)
@click.option('-t', '--type', default='c', type=click.Choice(['c', 'yaml', 'plain']), help='Type of output')
@click.option('-w', '--width', default=32, type=click.INT, help='Width of the lines produced')
@click.option('--clipboard/--no-clipboard', default=True)
@click.option('--taps', default=None, callback=split_num_list)
def get_symbol(size, width, type, clipboard, taps):
    symbol, _ = signal.max_len_seq(size, taps=taps)
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
        out += f"// size: {symbol_size}\n"
        out += f"uint8_t symbol[{symbol_size}] = {{\n\t{symbol}\n}};"
    elif type == 'yaml':
        symbol = '\n         '.join([' '.join(line) for line in lines])
        out += f"# taps: {taps}\n"
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
        print(symbol_size)
        print("Result copied to clipboard")




if __name__ == '__main__':
    get_symbol()

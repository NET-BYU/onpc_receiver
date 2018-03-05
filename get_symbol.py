import click
import numpy as np
import pyperclip
from scipy import signal


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


@click.command()
@click.argument('size', nargs=1, type=click.INT)
@click.option('-t', '--type', default='c', type=click.Choice(['c', 'yaml']), help='Type of output')
@click.option('-w', '--width', default=32, type=click.INT, help='Width of the lines produced')
@click.option('--clipboard/--no-clipboard', default=True)
@click.option('--state', default=1, type=click.INT)
def get_symbol(size, width, type, clipboard, state):
    np.set_printoptions(threshold=1e100)

    format_str = f'{{:#0{size + 2}b}}'
    state = [int(x) for x in format_str.format(state)[2:]]

    symbol, _ = signal.max_len_seq(size, state=state)
    symbol_size = len(symbol)
    symbol = str(list(symbol))

    # Remove brackets
    symbol = symbol[1:-1]

    # Break into lines
    lines = chunks(symbol.split(), width)

    # Join it all up with new lines between lines


    if type == 'c':
        symbol = '\n    '.join([' '.join(line) for line in lines])
        out = f"uint8_t symbol[{symbol_size}] = {{\n\t{symbol}\n}};"
    elif type == 'yaml':
        symbol = '\n         '.join([' '.join(line) for line in lines])
        out = f"symbol: [{symbol}]"
    else:
        out = "Unknown type!"

    if not clipboard:
        print(out)
    else:
        pyperclip.copy(out)
        print(symbol_size)
        print("Result copied to clipboard")




if __name__ == '__main__':
    get_symbol()

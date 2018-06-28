import json

import arrow
import glob


def add_chip_time(data, name):
    capture_time = arrow.get(data['capture_time'])
    change_time = arrow.get('2018-06-05')

    if 'chip_time' in data:
        print(f'Skipping {name}')
        return

    if capture_time > change_time:
        data['chip_time'] = '13.424 ms'
    else:
        data['chip_time'] = '11.54 ms'


def add_symbol_number(data, name):
    if 'symbol_number' in data:
        print(f'Skipping {name}')
        return

    print(data['location'])
    print(data['description'])
    symbol_number = int(input('Symbol number: '))
    print('\n')

    data['symbol_number'] = symbol_number



data_files = glob.glob('collected_data/*.json')

for data_file in data_files:
    with open(data_file) as f:
        data = json.load(f)

    # add_chip_time(data, data_file)
    add_symbol_number(data, data_file)

    with open(data_file, 'w') as f:
        json.dump(data, f)




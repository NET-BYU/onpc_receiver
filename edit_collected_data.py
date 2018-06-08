import json

import arrow
import glob


data_files = glob.glob('collected_data/*.json')


for data_file in data_files:
    with open(data_file) as f:
        data = json.load(f)

    capture_time = arrow.get(data['capture_time'])
    change_time = arrow.get('2018-06-05')

    if 'chip_time' in data:
        print(f"Skipping {data_file}")
        continue

    if capture_time > change_time:
        data['chip_time'] = "13.424 ms"
    else:
        data['chip_time'] = "11.54 ms"

    with open(data_file, 'w') as f:
        json.dump(data, f)




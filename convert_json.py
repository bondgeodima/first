import json
import os

dirname_in = 'D:/TEMP/_____NIGD/mirbank/_in/upload/out_json/'
dirname_out = 'D:/TEMP/_____NIGD/mirbank/_in/upload/out_json_convert/'

for filename in os.listdir(dirname_in):
    with open(dirname_in + filename) as json_file_in:
        data = json.load(json_file_in)
        with open(dirname_out + filename, 'w') as json_file_out:
            print (filename)
            json.dump(data['data']['geojson'], json_file_out)
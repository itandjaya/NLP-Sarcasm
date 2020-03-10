## import_data.py
## Data provided by https://rishabhmisra.github.io/publications/.

import os;
import json;
import random;
import pickle;

DATA_PATH           =   os.getcwd() + r'/data/';

def parse_data(file):
    json_file   =   open(file,'r');
    for l in json_file:
        yield json.loads(l);
    json_file.close();


def import_json( data_path = DATA_PATH, reshuffle = False):

    #raw_data    =   [];
    x_data, y_data  =   [], [];
    #max_len_word = 0;

    if reshuffle:

        for (dirpath, dirnames, filenames) in os.walk(data_path):
            for filename in filenames:

                if filename.startswith("Sarcasm_"):

                    full_file_name  =   data_path + filename;
                    json_data   = list(  parse_data(full_file_name)  ); 
                    data_size   =   len(json_data);

                    i_random    =   list(range(data_size));
                    random.shuffle(i_random);

                    for i in i_random:
                        x_data.append(  json_data[i]['headline']);
                        y_data.append(  json_data[i]['is_sarcastic']);  

                    #max_len_word = max(max_len_word, len(   x_data[-1].split(" ")));

        with open( DATA_PATH + "data.p", "wb" ) as file_o:       
            pickle.dump( [x_data, y_data],  file_o);

    
    #print(x_data[300], y_data[300]);
    #print(len(x_data), len(y_data));
    #print("max_len_word = ", max_len_word);

    with open( DATA_PATH + "data.p", "rb" ) as file_o:
        x_data, y_data  =   pickle.load( file_o );

    return  (x_data, y_data);
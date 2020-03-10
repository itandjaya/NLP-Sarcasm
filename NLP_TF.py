## NLP_TF.py
## To predict if a sentence is sarcasm (1) or not (0).
## The NLP model will automatically saves training variable on each epoch.
## Download the dataset manually for this program.

## 3 Models to choose:
    -. Standard embedding, Bidirectional LSTM, or Conv1D.

## Data provided by:
## Sarcasm Detection using Hybrid Neural Network
## Rishabh Misra, Prahal Arora
## Arxiv, August 2019.
## https://rishabhmisra.github.io/publications/
## https://scholar.google.com/citations?view_op=list_works&hl=en&user=EN3OcMsAAAAJ#


from __future__ import division, absolute_import, print_function, unicode_literals;

import tensorflow as tf;
from tensorflow import keras;
from tensorflow.keras.preprocessing.text import Tokenizer;
from tensorflow.keras.preprocessing.sequence import pad_sequences;


import os;
import matplotlib.pyplot as plt;    #   To display plot.
import numpy as np;

from import_data import import_json;


## Model Hyper-parameters & constants.

DATA_PATH           =   os.getcwd() + r'/data/';

STR_OOV     =   r'<00V>';
MAX_LEN     =   150;

# Variables for training.
EPOCHS      =   20;
BATCH_SIZE  =   512;

## Variables needed for Embedding layer.
VOCAB_SIZE      =   50000;
EMBEDDED_DIMS   =   20;
PADDING_TYPE    =   'post';


## If RESET_TRAINING is True, this resets all the pre-trained variables.
## Sets this to True if you run this the first time.
RESET_TRAINING  =   False;

def main():

    dataset    =   import_json( reshuffle = RESET_TRAINING);
    (x_train, y_train), (x_val, y_val), (x_test, y_test), tokenizer = \
            tokenize_data(dataset);

    ## Build & compile the NLP Layer.
    '''
    model   =   tf.keras.Sequential([
                            tf.keras.layers.Embedding(
                                    input_dim   = VOCAB_SIZE, 
                                    output_dim  = EMBEDDED_DIMS,
                                    input_length  = MAX_LEN,  
                            ),

                            #tf.keras.layers.Flatten(),
                            tf.keras.layers.GlobalAveragePooling1D (),
                            tf.keras.layers.Dense(64, activation = 'relu'),
                            tf.keras.layers.Dense(1, activation = 'sigmoid'),

    ]);

    
    ## Model with LSTM:
    model   =   tf.keras.Sequential([
                            tf.keras.layers.Embedding(
                                    input_dim   = VOCAB_SIZE, 
                                    output_dim  = 64,
                                    input_length  = MAX_LEN,  
                            ),

                            #tf.keras.layers.Flatten(),
                            #tf.keras.layers.GlobalAveragePooling1D (),
                            tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(64)),
                            
                            tf.keras.layers.Dense(64, activation = 'relu'),
                            tf.keras.layers.Dense(1, activation = 'sigmoid'),

    ]);
    '''
    ## Model with Conv1D:

    model   =   tf.keras.Sequential([
                            tf.keras.layers.Embedding(
                                    input_dim   = VOCAB_SIZE, 
                                    output_dim  = 64,
                                    input_length  = MAX_LEN,  
                            ),

                            #tf.keras.layers.Flatten(),
                            tf.keras.layers.Conv1D(128, 5, activation = 'relu'),
                            tf.keras.layers.GlobalAveragePooling1D (),
                            #tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(64)),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dense(24, activation = 'relu'),
                            tf.keras.layers.Dense(1, activation = 'sigmoid'),

    ]);

    model.compile(  optimizer   =   'adam',
                    loss        =   'binary_crossentropy',
                    metrics     =   ['accuracy']  );

    model.summary();


    # Create a callback that saves the model's weights.
    ckpt_path   =   os.getcwd() + "/training_1/model1"
    cp_callback =   tf.keras.callbacks.ModelCheckpoint( filepath =  ckpt_path,
                                                        save_weights_only = True,
                                                        verbose = 1);

    if not RESET_TRAINING:  model.load_weights(ckpt_path);

    ## Train the model and save the loss values per epoch for plotting.
    history =   model.fit(  x = x_train,
                            y = y_train,
                            epochs = EPOCHS,
                            callbacks = [cp_callback],
                            validation_data = (x_val, y_val),
                            verbose = 1,
                            );    

    if EPOCHS:  plot_training_model(history);

    #Evaluate against the test data.    
    (test_loss, test_acc)  =   model.evaluate(     x_test,  y_test, verbose=2); # loss: 0.1443 - accuracy: 0.9845.

    

    ## Testing sentences to predict whether each sentence is sarcasm.
    test_sentences   = [    'Granny starting to fear spiders in the garden might be real',
                            'the weather today is bright and sunny.',
                            'Good job, you broke the model.',
                            'You are too smart that you receive a fail grade.',
                            'Iwan is a good boy',
                            'I love those mustard stains on your oversized hoodie. They really bring out the color in your eyes.',
                            'That\'s just what I needed today!',
                            'I work 40 hours a week for us to be this poor.',
                            'Well, what a surprise.',
                            'Is it time for your medication or mine?',
                            'Really, Sherlock? No! You are clever.',
                            'Nice perfume. How long did you marinate in it?',
                            'Very good; well done!',
    ];

    padded      =   pad_sequences(  tokenizer.texts_to_sequences(test_sentences),
                                    padding = 'post',
                                    truncating = 'post',
                                    maxlen = MAX_LEN);
    
    print(model.predict(x = padded));

    return 0;

def plot_training_model(history = []):

    if not history:     return;

    plt.plot(   history.history['accuracy'],        label = 'train_accuracy');
    plt.plot(   history.history['val_accuracy'],    label = 'val_accuracy');

    plt.xlabel('Epoch');
    plt.ylabel('Accuracy');
    plt.legend(loc='lower right');

    plt.ylim(   [0.6, 1]);
    plt.show();

    return;

def tokenize_data( data_pair):

    x_data, y_data = data_pair;

    ## Indexes to split the data into 60% train_set, 20% validation_set, 20% test_set.
    i_60, i_80          =   round(len(x_data)*0.6), round(len(x_data)*0.8);

    tokenizer   =   Tokenizer(num_words = VOCAB_SIZE, oov_token = STR_OOV);
    tokenizer.fit_on_texts( x_data[:i_60]);
    #print("word_index: ", tokenizer.word_index);
    #print(len(tokenizer.word_index));

    x_data      =   tokenizer.texts_to_sequences(x_data);
    x_data      =   pad_sequences(  x_data,
                                    padding = 'post',
                                    truncating = 'post',
                                    maxlen = MAX_LEN);
    
    #print(len(x_data), len(y_data));


    x_data      =   tf.convert_to_tensor(   x_data, dtype = tf.int32);
    y_data      =   tf.convert_to_tensor(   y_data, dtype = tf.int32);


    ## Add a dimension.
    #x_data  = x_data[..., tf.newaxis];
    #y_data  = y_data[..., tf.newaxis];    
    
    x_train, y_train    =   x_data[:i_60],      y_data[:i_60];
    x_val, y_val        =   x_data[i_60:i_80],  y_data[i_60:i_80];
    x_test, y_test      =   x_data[i_80:],      y_data[i_80:];

    #print(x_train.shape, x_val.shape, x_test.shape);
    #print(y_train.shape, y_val.shape, y_test.shape);

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), tokenizer;


if __name__ == '__main__':  main();

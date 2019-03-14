from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import re


import pickle
import tensorflow as tf
import sys
import math
import json
import pickle



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    sentence_list_fname = './assignment_training_data_word_segment.json'
    sentence_list = json.load(open(sentence_list_fname , 'r'))

    raw_sentence_list_total = []
    label = 0

    for sentence in sentence_list:
        raw_sentence_list = []
        for time in sentence['times']:
            for attr in sentence['attributes']:
                for value in sentence['values']:
                    raw_sentence = sentence['sentence'].replace(sentence['words'][time], ' ').replace(sentence['words'][attr], ' ').replace(sentence['words'][value], ' ')
                    if([time, attr, value] in sentence['results']):
                        label = 1
                    else:
                        label = 0
                    raw_sentence_list.append([sentence['sentenceId'],raw_sentence,label])
                
        raw_sentence_list_total.append(raw_sentence_list)
    
    positive_examples = []
    negative_examples = []

    for raw_sentence_list in raw_sentence_list_total:
        for raw_sentence in raw_sentence_list:
            if raw_sentence[2] == 1:
                positive_examples.append(raw_sentence[1])
            else:
                negative_examples.append(raw_sentence[1])

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

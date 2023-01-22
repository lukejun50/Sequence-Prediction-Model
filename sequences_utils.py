import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def read_sequences_from_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().replace("\n", "")
    sequences_list = lines.split(",")
    return sequences_list

def convert_to_num(sequence):
    letter_to_num_map = {"A":"1", "C":"2", "T":"3", "G":"4", "N":"5"}
    converted_sequence = ""
    for s in sequence:
        s_num = letter_to_num_map[s]
        converted_sequence = converted_sequence + s_num
    ## converted_sequence_int = int(converted_sequence)
    return converted_sequence

def pad_sequence(sequence, desired_length):
    sequence_length = len(sequence)
    if sequence_length < desired_length:
        num_pads_needed = desired_length - sequence_length
        pad = ""
        for i in range(num_pads_needed):
            pad = pad + "0"
        result = sequence + pad
        return result
    else:
        return sequence
    
def split_or_pad(sequence, desired_length):
    seq_length = len(sequence)
    if seq_length > desired_length:
        new_sequences = []
        for i in range(0, seq_length, desired_length):
            current_chunk = sequence[i: i + desired_length]
            current_chunk_num = convert_to_num(current_chunk)
            current_chunk_num_padded = pad_sequence(current_chunk_num, desired_length)
            new_sequences.append(current_chunk_num_padded)
        return new_sequences
    else:
        sequence_num = convert_to_num(sequence)
        return [pad_sequence(sequence_num, desired_length)]
    
def convert_data_to_batch_size_tensor(cleaned_data, i, j):
    cleaned_data = cleaned_data[i:j]
    xs = np.array([],dtype = np.float64).reshape(0, len(cleaned_data[0]))
    for sequence in cleaned_data:
        sequence_list = [*sequence]
        sequence_array = np.asarray(sequence_list)
        sequence_array_float = sequence_array.astype(np.float64)
        xs = np.vstack([sequence_array_float, xs])
    return xs

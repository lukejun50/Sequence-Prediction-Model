import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sequences_utils import *

pathogenic_sequences_list = read_sequences_from_file("Pathogenic2.txt")
print(len(pathogenic_sequences_list))

desired_sequence = pathogenic_sequences_list[1]

# Prediction Software
def predict_sequence(sequence, model_path):
    cleaned_data = []
    desired_sequence_length = 2020
    cleaned_sequence_list = split_or_pad(sequence, desired_sequence_length)
    for s in cleaned_sequence_list:
        cleaned_data.append(s)
    # print(cleaned_data)
    t = convert_data_to_batch_size_tensor(cleaned_data, 0, len(cleaned_data))
    data_x = torch.FloatTensor(t)
    n_input, n_hidden_1, n_hidden_2, n_out, batch_size, learning_rate = desired_sequence_length, 500, 50, 1, 4, 0.0001
    model = nn.Sequential(nn.Linear(n_input, n_hidden_1), nn.ReLU(), nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(), nn.Linear(n_hidden_2, n_out), nn.Sigmoid())
    model.load_state_dict(torch.load(model_path))
    model.eval()
    pred_y = model(data_x)
    pred_y_array = pred_y.detach().numpy()
    pred_y_array = np.round(pred_y_array)
    num_ones = np.count_nonzero(pred_y_array==1)
    num_zeroes = len(pred_y_array) - num_ones
    if num_ones >= num_zeroes:
        return "Pathogenic"
    else:
        return "Benign"
    
print(len(desired_sequence))
print(predict_sequence(desired_sequence, "sequence_model.pt"))

 
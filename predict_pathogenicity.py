import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sequences_utils import *

pathogenic_sequences_list = read_sequences_from_file("Pathogenic.txt")
print(len(pathogenic_sequences_list))

desired_sequence = pathogenic_sequences_list[1]

# Prediction Software
def predict_sequence(sequence, model_path):
    if len(sequence) > 180000000:
        return "Sequence is too long"
    cleaned_data = []
    cleaned_sequence_list = split_or_pad(sequence, 5000)
    for s in cleaned_sequence_list:
        cleaned_data.append(s)
    # print(cleaned_data)
    t = convert_data_to_batch_size_tensor(cleaned_data, 0, len(cleaned_data))
    data_x = torch.FloatTensor(t)
    n_input, n_hidden, n_out, batch_size, learning_rate = 5000, 500, 1, 1000, 0.01
    model = nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out), nn.Sigmoid())
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
    
    
print(predict_sequence(desired_sequence, "sequence_model.pt"))

 
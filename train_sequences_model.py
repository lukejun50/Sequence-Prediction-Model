import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sequences_utils import *

# Load Sequences
pathogenic_sequences_list = read_sequences_from_file("Pathogenic.txt")
benign_sequences_list = read_sequences_from_file("Benign.txt")
likelybenign_sequences_list = read_sequences_from_file("Likely-Benign.txt")
likelypathogenic_sequences_list = read_sequences_from_file("Likely-Pathogenic.txt")
notreported_sequences_list = read_sequences_from_file("Not-Reported.txt")

# Compile
file_pathogenicity = [1,1,0,0]
file_pathogenicity_data = [pathogenic_sequences_list, likelypathogenic_sequences_list, benign_sequences_list, likelybenign_sequences_list]
all_sequences_data = []
all_sequences_data_target = []
for i in range(len(file_pathogenicity_data)):
    current_sequence_list = file_pathogenicity_data[i]
    for sequence in current_sequence_list:
        all_sequences_data.append(sequence)
        all_sequences_data_target.append(file_pathogenicity[i])
sequence_lengths = []
for s in all_sequences_data:
    sequence_lengths.append(len(s))

# Clean Data
cleaned_data = []
cleaned_data_targets = []
for i in range(len(all_sequences_data)):
    current_sequence = all_sequences_data[i]
    if len(current_sequence) > 180000000:
           continue
    cleaned_sequence_list = split_or_pad(current_sequence, 5000)
    for s in cleaned_sequence_list:
        cleaned_data.append(s)
        cleaned_data_targets.append(all_sequences_data_target[i])

#Shuffle
cleaned_data, cleaned_data_targets = shuffle(cleaned_data, cleaned_data_targets)

# Defining Model
n_input, n_hidden, n_out, batch_size, learning_rate = 5000, 500, 1, 1000, 0.01
model = nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out), nn.Sigmoid())

# Train Model
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
losses = []
batch_num = 0
batch_length = 1000
for i in range(0,28000,batch_length):
    t = convert_data_to_batch_size_tensor(cleaned_data, i, i + batch_length)
    t_target = np.asarray(cleaned_data_targets[i:i+batch_length]).reshape(batch_length,1)
    data_x = torch.FloatTensor(t)
    data_y = torch.FloatTensor(t_target)
    pred_y = model(data_x)
    loss = loss_function(pred_y, data_y)
    losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    optimizer.step()
    batch_num = batch_num + 1
    print(batch_num)

# Visualize Losses
plt.plot(losses)
plt.ylabel("losses")
plt.xlabel("epoch")
plt.title("Learning Rate %f" %(learning_rate))
plt.show()

# Save Model
torch.save(model.state_dict(), "sequence_model_new.pt")


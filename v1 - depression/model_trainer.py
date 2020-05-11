MODELS_PATH = "C:\\OneDrive - Cumberland Valley School District\\EEG ScienceFair\\src\\v1 - depression\\models"
import sys
sys.path.append(MODELS_PATH)
import os
def is_model(filename):
    if filename[0] == 'm':
        return int(filename[6:-3])
    return 0
MODEL_NAME = f"model_{max([is_model(filename) for filename in os.listdir(MODELS_PATH)])}"
print(f"Model Name: {MODEL_NAME}")
Net = getattr(__import__(MODEL_NAME, fromlist=["Net"]), "Net")
#######################################################################################################################
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

####################
torch.manual_seed(1)
####################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on - {device}")

net = Net().to(device)

def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = criterion(outputs, torch.argmax(y, 1))
    if train:
        loss.backward()
        optimizer.step()
    del outputs
    del matches
    return acc, loss

npdataset = np.load(r"v1 - depression\pre-processed_data\qEEGs_balanced.npy", allow_pickle=True)
dataset = []
for i in npdataset:
    dataset.append([torch.Tensor(i[0]), torch.Tensor(i[1])])

# find a better way to split data
dataset_len = len(dataset)
train_len, val_len, test_len = int(.80*dataset_len), int(.10*dataset_len), int(.10*dataset_len)
train_len += dataset_len-train_len-val_len-test_len
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=250, shuffle=True) # model trains on
val_loader = torch.utils.data.DataLoader(val_set, batch_size=250, shuffle=False) # validates during model training
test_loader = torch.utils.data.DataLoader(test_set, batch_size=250, shuffle=True) # after training is completed

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
EPOCHS = 25

train_losses, val_losses = [], []
train_accs, val_accs = [], []

with open(f"v1 - depression\\training_data\\{MODEL_NAME}@{int(time.time())}.log", 'a') as f:
    f.write(f"loss:{str(criterion)}, optimizer:{str(optimizer)}")
    f.write("times, training_accuracy, validation_accuracy, training_loss, validation_loss\n")
    for epoch in tqdm(range(1,EPOCHS+1)):
        for [qEEGs, labels], [val_qEEGs, val_labels] in zip(train_loader, val_loader):

            qEEGs, labels = qEEGs.to(device).view(-1, 1, 64, 64), labels.to(device)
            val_qEEGs, val_labels = val_qEEGs.to(device).view(-1, 1, 64, 64), val_labels.to(device)
            
            train_acc, train_loss = fwd_pass(qEEGs, labels, train=True)
            val_acc, val_loss = fwd_pass(val_qEEGs, val_labels)
            del qEEGs
            del labels
            del val_qEEGs
            del val_labels
            f.write(f"{round(time.time(),3)},{round(float(train_acc),4)},{round(float(val_acc), 4)},{round(float(train_loss),5)},{round(float(val_loss),5)},{epoch}\n")

        # should be in nested for loop (did to save some memeory)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            del train_loss
            del train_acc
            del val_loss
            del val_acc
f.close()

plt.plot(train_losses, 'b',label="Training Set")
plt.plot(val_losses, 'r',label="Validation Set")
plt.show()

plt.plot(train_accs, 'b',label="Training Set")
plt.plot(val_accs, 'r',label="Validation Set")
plt.show()
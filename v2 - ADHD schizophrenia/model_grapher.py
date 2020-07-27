import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

MODELS_PATH = "C:\\OneDrive - Cumberland Valley School District\\EEG ScienceFair\\src\\v1 - depression\\training_data"
MODEL_NAME = None
recent = 0
for filename in os.listdir(MODELS_PATH):
    if filename[0]=='m' and int(filename[18:-4])>=recent:
        recent = int(filename[18:-4])
        MODEL_NAME = filename

print(f"Model Name: {MODEL_NAME}")
contents = open(MODELS_PATH+"\\"+MODEL_NAME, "r").read().split("\n")
contents = [c.split(',') for c in contents[:-1]]
times, train_accs, val_accs, train_losses, val_losses, epochs = [], [], [], [], [], []

for sclice in contents[14:]:
    times.append(float(sclice[0]))
    train_accs.append(float(sclice[1]))
    val_accs.append(float(sclice[2]))
    train_losses.append(float(sclice[3]))
    val_losses.append(float(sclice[4]))
    epochs.append(float(sclice[5]))

times = ((np.array(times)-min(times)) / (max(times)-min(times))) * max(epochs)
fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

ax1.set_ylim([0,1])
ax1.set_ylabel('Accuracy (0.0-1.0)')
ax1.plot(times, train_accs, label="Training Set Accuracy")
ax1.plot(times, val_accs, label="Validation Set Accuracy")
#ax1.plot(times, abs(np.array(val_accs)-np.array(accuracies)), label="Δ Accuracy (between Training & Validation Set)")
ax1.legend(loc=2)

ax2.set_ylim([0,2])
ax2.set_ylabel('Loss (0.0-2.0)')
ax2.set_xlabel(f'Epochs (0-{max(epochs)})')
ax2.plot(times, train_losses, label="Training Set Loss")
ax2.plot(times, val_losses, label="Validation Set Loss")
#ax2.plot(times, abs(np.array(val_losses)-np.array(losses)), label="Δ Loss (between Training & Validation Set)")
ax2.legend(loc=2)
plt.savefig("v1 - depression/graphs/"+MODEL_NAME[:-4])
plt.show()
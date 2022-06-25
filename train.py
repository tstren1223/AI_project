# %%
#-----------------
#user defined
#-----------------
import cnn
import loader
#------------------
#lib defined
#------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.utils.data import random_split
import pandas as pd
from pathlib import Path

# ----------------------------
# Training Loop
# ----------------------------


def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(
        len(train_dl)), epochs=num_epochs, anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score in label dimension
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('Finished Training')
def read(data_path=Path.cwd(),dir_name=None,mata=True):
  #data_path=Path.cwd()/'UrbanSound8K'
  # Read metadata file
  data_path=data_path/dir_name
  if mata:
    metadata_file = data_path/'metadata'/(dir_name+'.csv')
    df = pd.read_csv(metadata_file)
  # Construct file path by concatenating fold and file name
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

  # Take relevant columns
    df = df[['relative_path', 'classID']]#replace with only two column
    return df

def dataSet_load(name,mata=True):
  'return train dataset and label dataset'
  myds = loader.SoundDS(read(dir_name=name,mata=mata),Path.cwd()/(name+'/audio'))

# Random split of 80:20 between training and validation
  num_items = len(myds)
  num_train = round(num_items * 0.8)
  num_val = num_items - num_train
  train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
  val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
  return (train_dl,val_dl)
def dataSet_load_test(name,mata=True):
  'return train dataset and label dataset'
  myds = loader.SoundDS(read(dir_name=name,mata=mata),Path.cwd()/(name/'audio'))

# Random split of 80:20 between training and validation
  num_items = len(myds)
  num_train = round(num_items * 0.8)
  num_val = num_items - num_train
  train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
  val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
  return (train_dl,val_dl)



if __name__ == '__main__':
  num_epochs = 10   # Just for demo, adjust this higher.
  myModel = cnn.AudioClassifier()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  myModel = myModel.to(device)
  # Check that it is on Cuda
  #print(next(myModel.parameters()).device)
  t_dl,v_dl=dataSet_load('UrbanSound8K');
  myModel.train();
  training(myModel, t_dl, num_epochs)
  myModel.eval();
  dataiter = iter(v_dl)
  sounds, labels = dataiter.next()
  num_examples = sounds.size()[0]

# move the examples and the labels to the GPU if available (if the network is on the GPU, we have to move any data there before processing it!)
  sounds = sounds.to(device)
  labels = labels.to(device)

# compute predictions and bring them back to the CPU
  outputs = myModel(sounds)
  _, predictions = torch.max(outputs, 1)


#print(' '.join(('{}'.format(j) for j in labels)))
  print("Ground truth: ", " ".join("{}".format(labels[j]) for j in range(num_examples)))
  print("Predicted:    ", " ".join("{}".format(predictions[j]) for j in range(num_examples)))
  acc = 100*(labels==predictions).sum() // num_examples
  print("Accuracy: {}%".format(acc))

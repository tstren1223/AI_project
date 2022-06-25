from project import *
import torch
import sys
import pandas as pd
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from pathlib import Path
import random
import torch
from torchaudio import transforms
import sounddevice as Audio
hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": 1,
        "epochs": 5
    }
use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")
model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
print(device)
#print(model)

#print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

optimizer = optim.AdamW(model.parameters(), hparams
                        ['learning_rate'])
criterion = nn.CTCLoss(blank=28).to(device)

train_dataset = torchaudio.datasets.LIBRISPEECH("",download=False)
test_dataset = torchaudio.datasets.LIBRISPEECH("",url="test-clean",download=False)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], steps_per_epoch=int(len(train_loader)),epochs=hparams['epochs'],anneal_strategy='linear')

for epoch in range(1, 5 + 1):
  train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
  #test(model, device, test_loader, criterion, epoch)
  torch.save(model.state_dict(), str("sound.pth"))
#tsv_to_csv("en","en/train","en/validated");
#tsv_to_csv("ch","ch/train","ch/validated");
#tsv_to_csv("tw","tw/train","tw/validated");

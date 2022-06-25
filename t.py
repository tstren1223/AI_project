from project  import *
import torch
import speech_recognition as sr
import pyttsx3
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
import wavio as wv
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
        "batch_size": 10,
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

model.load_state_dict(torch.load("sound.pth",map_location='cpu'))
#print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

optimizer = optim.AdamW(model.parameters(), hparams
                        ['learning_rate'])
criterion = nn.CTCLoss(blank=28).to(device)

train_dataset = torchaudio.datasets.LIBRISPEECH("",download=False)
test_dataset = torchaudio.datasets.LIBRISPEECH("",url="test-clean",download=False)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
model.eval()
torch.no_grad();
import os
file_path= 'test'
def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))
def spectro_gram(aud,sr, n_mels=64, n_fft=1024, hop_len=None,top=80):
        sig= aud
        top_db = top

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(
        sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
def record_and_trans(mode):
    record_to_file('r.wav')
    if mode==0:
        s,rate=rechannel(torchaudio.load("r.wav"),1)
        spectrograms = []
        w = valid_audio_transforms(s).squeeze(0).transpose(0, 1)
        spectrograms.append(w)
        s = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        s=torch.squeeze(s,2)
        input=s.to(device)
        output = model(input)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        # Get the predicted class with the highest score
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), [], 1,out=True)
        print(decoded_preds)
        return decoded_preds
    elif mode==1:
        r=sr.Recognizer()
        with sr.WavFile("r.wav")  as source2:
            try:
                audio2=r.listen(source2)
                MyText=r.recognize_google(audio2)
                MyText=MyText.lower()
                print(MyText)
                return MyText
            except:
                return""
record_and_trans(1)

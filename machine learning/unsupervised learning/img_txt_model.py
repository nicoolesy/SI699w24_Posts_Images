import argparse
import os
import numpy as np
import pandas as pd
import random
import re
import sys
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from PIL import Image

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_model_name = 'google/vit-base-patch16-224-in21k'
# img_model_name = 'facebook/deit-base-patch16-224'
img_model = AutoModel.from_pretrained(img_model_name)
img_processor = AutoImageProcessor.from_pretrained(img_model_name)

txt_model_name = "bert-base-multilingual-cased"
txt_tokenizer = AutoTokenizer.from_pretrained(txt_model_name)
txt_model = AutoModel.from_pretrained(txt_model_name)

img_dir = "images/img_resized/"

def get_img_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = img_processor(image, return_tensors="pt")
    with torch.no_grad():
        inputs.to(device)
        img_model.to(device)
        outputs = img_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def get_txt_vector(txt):
    txt = "" if pd.isna(txt) else txt
    if txt == "":
        zero_tensor = torch.zeros(1, txt_model.config.hidden_size).to(device)
        return zero_tensor
    inputs = txt_tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        inputs.to(device)
        txt_model.to(device)
        outputs = txt_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def weighted_mse_loss(outputs, targets, weights):
    weights = weights.to(device)
    squared_diff = (outputs - targets)**2
    # squared_diff.to(device)
    weighted_squared_diff = squared_diff * weights
    return torch.mean(weighted_squared_diff)

class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_vector = get_img_vector(img_dir + self.df.iloc[idx]['image_file'])
        txt_vector = get_txt_vector(self.df.iloc[idx]['post_text'])
        ER = torch.tensor(self.df.iloc[idx]['ER'], dtype=torch.float32)
        time_features = torch.tensor([
            # self.df.iloc[idx]['year'],
            self.df.iloc[idx]['month'], 
            self.df.iloc[idx]['day'], 
            self.df.iloc[idx]['hour'], 
            self.df.iloc[idx]['weekday'], 
            self.df.iloc[idx]['season'], 
            self.df.iloc[idx]['hour_category'], 
            self.df.iloc[idx]['working_hour']
        ], dtype=torch.float32).unsqueeze(0)
        # weight = self.df.iloc[idx]['weight']
        # weight = torch.tensor(self.df.iloc[idx]['weight'], dtype=torch.float32)

        sample = {'img': img_vector, 'txt': txt_vector, 'ER': ER, 'time': time_features}
        # sample = {'img': img_vector, 'txt': txt_vector, 'ER': ER, 'time': time_features, 'weight': weight}
        return sample

class ERModel(nn.Module):
    def __init__(self):
        super(ERModel, self).__init__()
        self.fc_img = nn.Linear(768, 128)   # get 128 features from img
        self.fc_txt = nn.Linear(768, 128)   # get 128 features from txt
        self.fc_time = nn.Linear(7, 128)    # get 128 features from time (just to match the size of img & txt)
        self.fc1 = nn.Linear(128*3, 256)    # input dim = 128(img) + 128(txt) + 128(time)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, img, txt, t):
        img_out = torch.relu(self.fc_img(img))
        txt_out = torch.relu(self.fc_txt(txt))
        time_out = torch.relu(self.fc_time(t))
        combined_vector = torch.cat((img_out, txt_out, time_out), dim=1)
        flattened_vector = torch.flatten(combined_vector, start_dim=1)
        
        x = torch.relu(self.fc1(flattened_vector))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=5, patience=2):
    model.to(device)
    scaler = GradScaler()
    best_valid_loss = float('inf')
    no_improve_cnt = 0
    best_epoch = 0
    best_prediction = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            imgs, txts, ERs = data['img'].to(device), data['txt'].to(device), data['ER'].float().to(device)
            t = data['time'].to(device)
            # weights = data['weight'].to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs, txts, t)
                loss = criterion(outputs, ERs)
                # loss = weighted_mse_loss(outputs, ERs, weights)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        valid_loss, predictions = valid_model(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}, MSE: {running_loss / len(train_loader)}, Validation MSE: {valid_loss}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improve_cnt = 0
            best_epoch = epoch+1
            best_prediction = predictions
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_cnt += 1

        if no_improve_cnt >= patience:
            print(f"No improvement in validation loss for {patience} epochs. Early stopping...")
            break

    print(f"Training finished. Best model found at epoch {best_epoch} with Validation MSE: {best_valid_loss}")
    print("Best validation predictions are saved in best_predictions.txt")

    with open('best_predictions.txt', 'w') as f:
        for pred in best_prediction:
            f.write(f'{pred}\n')

def valid_model(model, valid_loader, criterion, device):
    model.to(device)
    model.eval()
    valid_loss = 0.0
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            imgs, txts, ERs = data['img'].to(device), data['txt'].to(device), data['ER'].float().to(device)
            t = data['time'].to(device)
            # weights = data['weight'].to(device)
            outputs = model(imgs, txts, t)
            loss = criterion(outputs, ERs)
            # loss = weighted_mse_loss(outputs, ERs, weights)
            valid_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())

    return valid_loss / len(valid_loader), predictions

def main():
    df = pd.read_csv('data_w_time_feature.csv')
    batch_size = 128

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=699)

    train_dataset = CustomDataset(train_df)
    test_dataset = CustomDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with open('test_ER_values.txt', 'w') as f:
        for er_value in test_df['ER']:
            f.write(f'{er_value}\n')

    er_model = ERModel()
    criterion = nn.MSELoss()
    optimizer = AdamW(er_model.parameters(), lr=0.001, weight_decay=0.001)

    train_model(er_model, train_loader, test_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()
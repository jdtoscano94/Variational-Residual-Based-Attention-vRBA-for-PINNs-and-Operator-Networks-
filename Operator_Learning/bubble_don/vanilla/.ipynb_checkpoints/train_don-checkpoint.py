import os
import sys

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from don import DeepONet
from scipy.interpolate import interp1d
from torchinfo import summary

# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import pickle

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true, Par, Lambda=None):
        # Implement your custom loss calculation here
        if Lambda is not None:
            residue = torch.absolute(y_true - y_pred)
            Lambda = Par['gamma']*Lambda + Par['eta']*residue/torch.max(residue)
            loss = torch.mean(torch.square(Lambda*residue)) 
        
        else:
            loss = torch.mean(torch.square(y_true - y_pred)) 

        return loss, Lambda

def preprocess(X_func, X_loc, y, Par):
    X_func = (X_func - Par['X_func_shift'])/Par['X_func_scale']
    X_loc  = (X_loc - Par['X_loc_shift'])/Par['X_loc_scale']
    y      = (y - Par['out_shift'])/(Par['out_scale'])
    print('X_func: ', X_func.shape)
    print('X_loc : ', X_loc.shape)
    print('y     : ', y.shape)

    return X_func, X_loc, y

    
def data_prep(dataset, m, npoints_output):
    p = dataset['del_p']
    t = dataset['t']
    r = dataset['R']

    P = interp1d(t, p, kind='cubic')
    R = interp1d(t, r, kind='cubic')

    t_min = 0
    t_max = 5 * 10**-4

    X_func = P(np.linspace(t_min, t_max, m)) #[1500, m] 
    X_loc  = np.linspace(t_min, t_max, npoints_output)[:, None] #[npoints_output,1]
    y      = R(np.ravel(X_loc)) #[1500, npoints_output] 

    return X_func, X_loc, y


# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################

debug = False

dataset = np.load('../data/0.1/res_1000.npz')

m = 200
npoints_output = 500

X_func, X_loc, y = data_prep(dataset, m, npoints_output) 

idx1 = int(0.8*X_func.shape[0])
idx2 = int(0.9*X_func.shape[0])

X_func_train = X_func[:idx1]
X_func_val   = X_func[idx1:idx2]
X_func_test  = X_func[idx2:]

X_loc_train = X_loc
X_loc_val   = X_loc 
X_loc_test  = X_loc

y_train = y[:idx1]
y_val   = y[idx1:idx2]
y_test  = y[idx2:]

Par = {
       'bn_res'        : X_func_train.shape[1],
       'tn_res'        : X_loc_train.shape[1],
       'ld'            : 100,   
       'X_func_shift'  : np.mean(X_func_train),
       'X_func_scale'  : np.std(X_func_train),
       'X_loc_shift'   : np.min(X_loc_train),
       'X_loc_scale'   : np.max(X_loc_train)-np.min(X_loc_train),
       'out_shift'     : np.mean(y_train),
       'out_scale'     : np.std(y_train),
       'eta'           : 0.1,
       'gamma'         : 0.99
       }

Par['Lambda_max'] = Par['eta']/(1 - Par['gamma'])


if debug:
    Par['num_epochs']  = 5
else:
    Par['num_epochs']  = 10000

print('\nTrain Dataset')
X_func_train, X_loc_train, y_train = preprocess(X_func_train, X_loc_train, y_train, Par)
Lambda = np.ones((y_train.shape[0], y_train.shape[1]), dtype=np.float32)*Par['Lambda_max']/2.0
print("Lambda: ", Lambda.shape)

print('\nValidation Dataset')
X_func_val, X_loc_val, y_val = preprocess(X_func_val, X_loc_val, y_val, Par)
print('\nTest Dataset')
X_func_test, X_loc_test, y_test = preprocess(X_func_test, X_loc_test, y_test, Par)

print('Par:\n', Par)

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################

# Create custom datasets
X_func_train_tensor = torch.tensor(X_func_train, dtype=torch.float32)
X_loc_train_tensor = torch.tensor(X_loc_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
Lambda_tensor = torch.tensor(Lambda, dtype=torch.float32)

X_func_val_tensor = torch.tensor(X_func_val, dtype=torch.float32)
X_loc_val_tensor = torch.tensor(X_loc_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

X_func_test_tensor = torch.tensor(X_func_test, dtype=torch.float32)
X_loc_test_tensor = torch.tensor(X_loc_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

# Define data loaders
train_batch_size = 50
val_batch_size   = 50
test_batch_size  = 50

# Initialize your Unet2D model
model = DeepONet(Par).to(device).to(torch.float32)
summary(model, input_size=((1,)+X_func_train.shape[1:], X_loc_train.shape)  )

# Define loss function and optimizer
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * int(y_train.shape[0]/train_batch_size) )  # Adjust T_max as needed

# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)

for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0
    counter=0

    for start in range(0, X_func_train.shape[0]-1, train_batch_size):
        end = start + train_batch_size
        x = X_func_train_tensor[start:end]
        y_true = y_train_tensor[start:end]  
        Lambda = Lambda_tensor[start:end]      
        toss = True
        if toss:
            optimizer.zero_grad()
            with autocast():
                y_pred = model(x.to(device), X_loc_train_tensor.to(device))
                loss, temp_Lambda   = criterion(y_pred, y_true.to(device), Par, Lambda.to(device))
            # print(temp_Lambda.detach())
            Lambda_tensor[start:end] = temp_Lambda.detach()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            counter += 1
            

        # Update learning rate
        scheduler.step()

    train_loss /= counter

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for start in range(0, X_func_val.shape[0]-1, val_batch_size):
            end = start + val_batch_size
            x = X_func_val_tensor[start:end]
            y_true = y_val_tensor[start:end]  
            with autocast():
                y_pred = model(x.to(device), X_loc_val_tensor.to(device))
                loss, _ = criterion(y_pred, y_true.to(device), Par)
                loss = loss.item()
            val_loss += loss
    val_loss /= int(y_val.shape[0]/val_batch_size)

        # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/best_model.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}')

print('Training finished.')

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for start in range(0, X_func_test.shape[0]-1, test_batch_size):
        end = start + test_batch_size
        x = X_func_test_tensor[start:end]
        y_true = y_test_tensor[start:end]  
        with autocast():
            y_pred = model(x.to(device), X_loc_test_tensor.to(device))
            loss, _ = criterion(y_pred, y_true.to(device), Par)
            loss = loss.item()
        test_loss += loss 
test_loss /= int(y_test.shape[0]/test_batch_size)
print(f'Test Loss: {test_loss:.4e}')

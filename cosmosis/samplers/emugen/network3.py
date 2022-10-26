import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

    
class NNEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std, model=None, optim=None, device='cpu'):
        self.N_DIM = N_DIM
        self.model = model
        self.optim = optim
        self.device = device
        self.trained = False
        self.dv_fid = torch.Tensor(dv_fid)
        self.dv_std = torch.Tensor(dv_std)
        
        if model is None:
            self.model = nn.Sequential(
                                nn.Linear(N_DIM, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, OUTPUT_DIM),
                                Affine()
                                )
        elif(model=='resnet'):
            self.model = nn.Sequential(
                    nn.Linear(N_DIM, 128),
                    ResBlock(128, 256),
                    ResBlock(256, 256),
                    ResBlock(256, 256),
                    ResBlock(256, 512),
                    ResBlock(512, 512),
                    ResBlock(512, 512),
                    ResBlock(512, 1024),
                    ResBlock(1024, 1024),
                    ResBlock(1024, 1024),
                    ResBlock(1024, 1024),
                    ResBlock(1024, 1024),
                    Affine(),
                    nn.PReLU(),
                    nn.Linear(1024, OUTPUT_DIM),
                    Affine()
                )

        self.model.to(device)

        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)

    def do_pca(self, data_vector, N_PCA):
        self.N_PCA = N_PCA
        pca = PCA(self.N_PCA)
        pca.fit(data_vector)
        self.pca = pca
        pca_coeff = pca.transform(data_vector)
        return pca_coeff
    
    def do_inverse_pca(self, pca_coeff):
        return self.pca.inverse_transform(pca_coeff)
    
    def train(self, X, y, test_split=None, batch_size=32, n_epochs=100):
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True))
            self.X_std  = torch.Tensor(X.std(axis=0, keepdims=True))
            # self.y_mean = self.dv_fid
            # self.y_std  = self.dv_std
            self.y_mean = torch.Tensor(y.mean(axis=0, keepdims=True))
            self.y_std  = torch.Tensor(y.std(axis=0, keepdims=True))


        X_train = (X - self.X_mean) / self.X_std
        y_train = (y - self.y_mean) / self.y_std

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        epoch_range = tqdm(range(n_epochs))
        
        losses = []
        for _ in epoch_range:
            for i, data in enumerate(trainloader):
                X_batch = data[0]
                y_batch = data[1]

                y_pred = self.model(X_batch)
                loss = torch.mean(torch.abs(y_batch - y_pred))
                losses.append(loss)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
            epoch_range.set_description('Loss: {0}'.format(loss))
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean.clone().detach()
            X_std  = self.X_std.clone().detach()

            X_norm = (X - X_mean) / X_std
            y_pred = self.model.eval()(X_norm).cpu()
            
        y_pred = y_pred * self.y_std + self.y_mean

        return y_pred.numpy()

    def save(self, filename):
        torch.save(self.model, filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['Y_mean'] = self.y_mean
            f['Y_std']  = self.y_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
        
    def load(self, filename):
        self.trained = True
        self.model = torch.load(filename)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:])
            self.X_std  = torch.Tensor(f['X_std'][:])
            self.y_mean = torch.Tensor(f['Y_mean'][:])
            self.y_std  = torch.Tensor(f['Y_std'][:])
            self.dv_fid = torch.Tensor(f['dv_fid'][:])
            self.dv_std = torch.Tensor(f['dv_std'][:])

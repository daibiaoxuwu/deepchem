from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn as nn
import time


with open('data.txt') as f:
    X = []
    Y = []
    for l in f.readlines():
        l1,l2=l.split()
        peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(l1));
        print(peptide_smiles)
        m = Chem.MolFromSmiles(peptide_smiles)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
        X.append(fp)
        logp = MolLogP(m)
        Y.append(logp)
    X = torch.from_numpy(np.array(X)).float()
    Y = torch.from_numpy(np.array(Y)).float()
    print(X.size())
    print(Y.size())
    num_train = int(Y.size()[0]*0.8)
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_test = X[num_train:]
    Y_test = Y[num_train:]
    print(f'Num train data: {len(X_train)}')
    print(f'Num test data: {len(X_test)}')
class LinearRegressor(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(LinearRegressor, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        retval = self.linear(x)
        return retval
lr = 1e-4
model = LinearRegressor(1024,1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
loss_list = []
st = time.time()
for i in range(500001):
    pred = model(X_train)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, Y_train)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data.cpu().numpy())
    if i%1000==0:
        print (i, loss.data.cpu().numpy())
        end = time.time()
        print ('Time:', end-st)
        if(loss.data.cpu().numpy() < 0.3):break


y_pred_train = model(X_train).squeeze(-1)
y_pred_test = model(X_test).squeeze(-1)
print(X_train, Y_train, y_pred_train)
print(X_test, Y_test, y_pred_test)
loss_train = loss_fn(Y_train, y_pred_train)
loss_test = loss_fn(Y_test, y_pred_test)

print ('Train loss:', loss_train.data.cpu().numpy())
print ('Test loss:', loss_test.data.cpu().numpy())

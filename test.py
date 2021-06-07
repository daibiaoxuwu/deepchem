
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import time
import deepchem as dc


tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets
for X, y, w, id in test_dataset.itersamples():
    print(y, id)
from rdkit import Chem
import numpy as np
import deepchem as dc

test_example = "EQQQQ"

X = []
Y = []
w = []

with open('data.txt') as f:
    for l in f.readlines():
        l1, l2 = l.split()
        if l1 == test_example: continue
        peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(l1))
        X.append(Chem.MolFromSmiles(peptide_smiles))
        Y.append([1])
        w.append(1 / float(l2))
X_train = X[:int(0.8 * len(X))]
Y_train = Y[:int(0.8 * len(X))]
w_train = w[:int(0.8 * len(X))]
X_test = X[int(0.8 * len(X)):]
Y_test = Y[int(0.8 * len(X)):]
w_test = w[int(0.8 * len(X)):]
X = []
Y = []
w = []
with open('data2.txt') as f:
    for l in f.readlines():
        l = l.strip()
        if l == test_example: continue
        peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(l))
        X.append(Chem.MolFromSmiles(peptide_smiles))
        Y.append([0])
        w.append(1)

X_train.extend(X[:int(0.8 * len(X))])
Y_train.extend(Y[:int(0.8 * len(X))])
w_train.extend(w[:int(0.8 * len(X))])
X_test.extend(X[int(0.8 * len(X)):])
Y_test.extend(Y[int(0.8 * len(X)):])
w_test.extend(w[int(0.8 * len(X)):])

print("Read complete")

featurizer = dc.feat.ConvMolFeaturizer()
X_train = featurizer.featurize(X_train)
train_dataset = dc.data.NumpyDataset(X=np.array(X_train), y=np.array(Y_train), w=w_train)
X_test = featurizer.featurize(X_test)
test_dataset = dc.data.NumpyDataset(X=np.array(X_test), y=np.array(Y_test), w=w_test)

# transformer= dc.trans.NormalizationTransformer( transform_y=True, dataset=X, move_mean=True)
# train_dataset = transformer.transform(train_dataset)
# test = transformer.transform(test_dataset)

model = dc.models.GraphConvModel(1, mode='classification')

print("Fitting the model")

model.fit(train_dataset)
metric = dc.metrics.Metric(dc.metrics.accuracy_score)
print('training set score:', model.evaluate(train_dataset, [metric], []))
print('testing set score:', model.evaluate(test_dataset, [metric], []))

print("testing",test_example)
peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(test_example))
X2 = [Chem.MolFromSmiles(peptide_smiles)]
Y2 = [0]
w2 = [1]
featurizer = dc.feat.ConvMolFeaturizer()
X2 = featurizer.featurize(X2)
test_dataset2 = dc.data.NumpyDataset(X=np.array(X2), y=np.array(Y2), w=w2)
ans = model.predict(test_dataset2, [])
print("bitter: ",ans[0,0,0], "umami: ",ans[0,0,1])

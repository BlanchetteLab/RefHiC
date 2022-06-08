import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cooler

from scipy import stats
from sklearn.model_selection import train_test_split
import h5py

import numpy as np
import numba as nb


@nb.njit(parallel=True)
def rankdata_core(data):
    """
    parallelized version of scipy.stats.rankdata along  axis 1 in a 2D-array
    """
    ranked = np.empty(data.shape, dtype=np.float64)
    for j in nb.prange(data.shape[0]):
        arr = np.ravel(data[j, :])
        sorter = np.argsort(arr)

        arr = arr[sorter]
        obs = np.concatenate((np.array([True]), arr[1:] != arr[:-1]))

        dense = np.empty(obs.size, dtype=np.int64)
        dense[sorter] = obs.cumsum()

        # cumulative counts of each unique value
        count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))
        ranked[j, :] = count[dense - 1]
    return ranked


def rankdata(data, axis=1):
    """
    parallelized version of scipy.stats.rankdata
    """
    shape = data.shape
    dims = len(shape)
    if axis + 1 > dims:
        raise ValueError('axis does not exist')
    if axis < dims - 1:
        data = np.swapaxes(data, axis, -1)
        shape = data.shape
    if dims > 2:
        data = data.reshape(np.prod(shape[:-1]), shape[-1])

    ranked = rankdata_core(data)

    if dims > 2:
        data = data.reshape(shape)
        ranked = ranked.reshape(shape)
    if axis < dims - 1:
        data = np.swapaxes(data, -1, axis)
        ranked = np.swapaxes(ranked, -1, axis)
    return ranked


class indicesOnlyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return idx


class loopDatasetHDF5(Dataset):
    def __init__(self, hdf5File, index, samples=None):
        self.data = hdf5File
        self.samples = samples
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        idx = self.index[idx]
        y = torch.tensor([self.data[str(idx)].attrs['target']]).float()
        test = torch.tensor(self.data[str(idx)]['test'])
        db = torch.tensor(self.data[str(idx)]['db'])
        if self.samples:
            selectedIdx = np.random.choice(db.shape[0], self.samples, replace=False)
            db = db[selectedIdx, :]
        X = (test, db)
        return X, y


class loopDataset(Dataset):
    def __init__(self, extrafiles, testfile, positions, labels, win=5, resol=5000, samples=None, transform=None,
                 target_transform=None):
        self.extraMats = []
        for extrafile in extrafiles:
            # print(extrafile + '::/resolutions/'+str(resol))
            self.extraMats.append(
                cooler.Cooler(extrafile + '::/resolutions/' + str(resol)).matrix(balance=True)
            )
        self.testMat = cooler.Cooler(testfile + '::/resolutions/' + str(resol)).matrix(balance=True)
        self.positions = positions
        self.labels = torch.from_numpy(labels).float()
        self.transform = transform
        self.samples = samples
        self.win = win * resol
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[0]

    def individualData(self, coolMat, regionX, regionY):
        mat = coolMat.fetch(regionX, regionY).flatten()
        ranks = stats.rankdata(mat, method='average')
        return np.hstack((mat, ranks))

    def __getitem__(self, idx):
        y = self.labels[idx]
        chrom = str(self.positions[idx]['chrom'])
        pos1 = int(self.positions[idx]['pos1'])
        pos2 = int(self.positions[idx]['pos2'])
        regionX = chrom + ':' + str(pos1 - self.win) + '-' + str(pos1 + self.win + 1)
        regionY = chrom + ':' + str(pos2 - self.win) + '-' + str(pos2 + self.win + 1)
        # print(regionX,regionY)
        if self.samples:
            selectedIdx = np.random.choice(len(self.extraMats), self.samples, replace=False)
            X = self.individualData(self.extraMats[selectedIdx[0]], regionX, regionX)
            for i in selectedIdx[1:]:
                x = self.individualData(self.extraMats[i], regionX, regionY)
                X = np.vstack((x, X))
        else:
            X = self.individualData(self.extraMats[0], regionX, regionX)
            for i in range(1, len(self.extraMats)):
                x = self.individualData(self.extraMats[i], regionX, regionY)
                X = np.vstack((x, X))
        testData = self.individualData(self.testMat, regionX, regionY)
        np.nan_to_num(X, copy=False)
        np.nan_to_num(testData, copy=False)
        testData = torch.from_numpy(testData).float()

        X = torch.from_numpy(X).float()
        X = (testData, X)

        if self.transform:
            return self.transform(X), y
        return X, y


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x


class attentionToAdditionalHiC(nn.Module):
    def __init__(self, input_size, encoding_dim=128):
        super(attentionToAdditionalHiC, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.encoding_dim),
            nn.ReLU(),
        )

        self.attentionMLP = nn.Sequential(
            nn.BatchNorm1d(self.encoding_dim),
            nn.Linear(self.encoding_dim, self.encoding_dim),
            # nn.BatchNorm1d(self.encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            # nn.LayerNorm(self.encoding_dim*2),
            nn.BatchNorm1d(self.encoding_dim * 2),
            nn.Linear(self.encoding_dim * 2, self.encoding_dim // 2),
            # nn.BatchNorm1d(self.encoding_dim//2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.encoding_dim // 2, 1),
        )

        self.att = nn.Parameter(torch.zeros((self.encoding_dim, self.encoding_dim)))
        nn.init.xavier_uniform_(self.att)

    def alphaCalc(self, x, dim):
        x = torch.sigmoid(x)
        reciprocal_marginal = 1 / torch.sum(x, dim=dim)
        x = x * reciprocal_marginal[:, None]
        return x

    def forward(self, x):
        x[0] = x[0].float()
        x[1] = x[1].float()
        batchSize = x[0].shape[0]
        sampleSize = x[1].shape[1]
        # print('batchsize',batchSize)
        # print('sampleSize',sampleSize)
        # print('x.shape',x[0].shape,'xs.shape',x[1].shape)
        testEncoding = self.encoder(x[0])
        restEncoding = self.encoder(x[1].view(batchSize * sampleSize, self.input_size)).view(batchSize, sampleSize,
                                                                                             self.encoding_dim)
        # print('x_encoding.shape',testEncoding.shape,'xs_encoding.shape',restEncoding.shape)
        K = restEncoding.permute(0, 2, 1)
        #         print('Q',testEncoding.shape,'W',self.att.shape,'Key',K.shape)
        QW = torch.matmul(testEncoding.unsqueeze(-2), self.att)
        # alpha = self.alphaCalc(torch.matmul(QW, K) / (self.encoding_dim ** 0.5), dim=-1)
        alpha = torch.softmax(torch.matmul(QW, K) / (self.encoding_dim ** 0.5), dim=-1)
        attention = torch.matmul(alpha, restEncoding).squeeze(-2)
        attention = self.attentionMLP(attention)
        # print('attention.shape',attention.shape,'testEncoding.shape',testEncoding.shape)
        output = torch.cat((testEncoding, attention), -1)
        output = self.MLP(output)
        output = torch.sigmoid(output)
        # print('output.shape',output.shape)
        return output


def train(model, X, Xs, y, train_loader, optimizer, criterion, epoch, device, attention=False, samples=None,
          batch_size=128):
    model.train()
    selectDBIndices = []
    NumSamplesInDB = Xs.shape[1]
    for i in range(batch_size):
        selectDBIndices.append(np.random.permutation(NumSamplesInDB))
    selectDBIndices = torch.tensor(selectDBIndices, dtype=torch.int64)
    for batch_idx, indices in enumerate(train_loader):
        batch_X = X[indices]
        batch_y = y[indices]
        batch_Xs = Xs[indices]
        if samples:
            k = selectDBIndices[0:len(indices), :]
            k = k[:, torch.randperm(NumSamplesInDB)[0:samples]]
            k = k[:, :, None]
            k = k.repeat(1, 1, batch_Xs.shape[2])
            batch_Xs = torch.gather(batch_Xs, 1, k)
        target = batch_y.to(device)
        if not attention:
            data = batch_X
            data = data.to(device)
            # print(data.shape,target.shape)
            # print(batch_idx)
        else:
            data = []
            data.append(batch_X.to(device))
            data.append(batch_Xs.to(device))
        print('batch_X.shape,batch_Xs.shape',batch_X.shape,batch_Xs.shape)

        optimizer.zero_grad()
        output = model(data)
        # print(output,target)
        loss = criterion(output, target.view(-1, 1))
        loss.backward()
        optimizer.step()
        if batch_idx == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, X, Xs, y, test_loader, device, attention=False, printData=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for indices in test_loader:
            batch_X = X[indices]
            batch_y = y[indices]
            batch_Xs = Xs[indices]
            target = batch_y.to(device)
            if not attention:
                data = batch_X
                data = data.to(device)
            else:
                data = []
                data.append(batch_X.to(device))
                data.append(batch_Xs.to(device))

            output = model(data)
            pred = output > 0.5  # get the index of the max log-probability
            if printData:
                a, b, c = batch_X.cpu().numpy(), pred.cpu().numpy() * 1, target.cpu().numpy()
                c = c[:, np.newaxis]
                out = np.hstack((a, b, c))
                np.savetxt('x_pred_target.txt', out)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #
    # return model


def trainMLPtorch(device, hdf5URI, batch_size=128, inMemoryData=True):
    hdf5File = h5py.File(hdf5URI, 'r')
    numOfSamples = hdf5File.attrs['samples']
    width = hdf5File.attrs['win']
    if inMemoryData:
        X = []
        Xs = []
        y = []
        idx = 0

        for key in hdf5File:
            idx = idx + 1
            yi = int(hdf5File[key].attrs['target'])
            xi = np.asarray(hdf5File[key]['test'])
            xsi = np.asarray(hdf5File[key]['db'])
            X.append(xi)
            y.append(yi)
            Xs.append(xsi)
        X = torch.tensor(X)
        y = torch.tensor(y).float()
        Xs = torch.tensor(Xs)

    indices = np.linspace(0, numOfSamples - 1, numOfSamples, dtype=int)
    epochs = 20
    indices_train, indices_test = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    training_data = indicesOnlyDataset(indices_train)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data = indicesOnlyDataset(indices_test)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # model = MLP((width*2+1)**2*2).to(device)
    model = MLP((width * 2 + 1) ** 2 + 2 * (width * 2 + 1) + 3).to(device)

    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        train(model, X, Xs, y, train_dataloader, optimizer, criterion, epoch, device, attention=False,
              batch_size=batch_size)
        test(model, X, Xs, y, test_dataloader, device, attention=False)
    #
    # return model
    torch.save(model.state_dict(), 'mlpModel.h5')


def trainAttention(device, hdf5URI, batch_size=128, inMemoryData=True, epochs=100, lr=1e-3, skipChrom=None):
    hdf5File = h5py.File(hdf5URI, 'r')
    numOfSamples = 0  # hdf5File.attrs['samples']
    width = hdf5File.attrs['win']
    if inMemoryData:
        X = []
        Xs = []
        y = []
        idx = 0

        for chrom in hdf5File:
            if chrom == skipChrom:
                continue
            print('read chrom', chrom)
            for key in hdf5File[chrom]:
                idx = idx + 1
                yi = int(hdf5File[chrom][key].attrs['target'])
                xi = np.asarray(hdf5File[chrom][key]['test'])
                xsi = np.asarray(hdf5File[chrom][key]['db'])
                numOfSamples = numOfSamples + 1
                # if yi ==1:
                #     print(hdf5File[key].attrs['frag1']+','+hdf5File[key].attrs['frag2'])
                # xi = xi[:441]
                # xi = np.hstack((xi,rankdata(xi[None,:],axis=1).flatten()))
                # xsi = xsi[:,:441]
                # xsi = np.hstack((xsi,rankdata(xsi,axis=1)))
                X.append(xi)
                y.append(yi)
                Xs.append(xsi)
        X = torch.tensor(X)
        y = torch.tensor(y).float()
        Xs = torch.tensor(Xs)

    indices = np.linspace(0, numOfSamples - 1, numOfSamples, dtype=int)

    indices_train, indices_test = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    training_data = indicesOnlyDataset(indices_train)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data = indicesOnlyDataset(indices_test)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # model = attentionToAdditionalHiC((width*2+1)**2*2).to(device)
    model = attentionToAdditionalHiC((width * 2 + 1) ** 2 + 2 * (width * 2 + 1) + 2, encoding_dim=256).to(device)
    # model = attentionToAdditionalHiC((width * 2 + 1) ** 2).to(device)

    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        train(model, X, Xs, y, train_dataloader, optimizer, criterion, epoch, device, attention=True, samples=10,
              batch_size=batch_size)
        test(model, X, Xs, y, test_dataloader, device, attention=True)

    test(model, X, Xs, y, test_dataloader, device, attention=True, printData=True)
    #
    # return model
    torch.save(model.state_dict(), 'attentionModel_' + skipChrom + '_bias.h5')


print("Attention")
hdf5URI = '/home/yanlin/workspace/PhD/project3/peakachu/trainingSet21x21.hdf5'
hdf5URI = '/home/yanlin/workspace/PhD/project3/peakachu/newInfoData.h5'

device = torch.device("cuda")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batchSize', type=int, default=4096)
parser.add_argument('-i', type=str, required=True)
parser.add_argument('--chr', type=str, required=True)
args = parser.parse_args()
hdf5URI = args.i
print(hdf5URI)
lr = args.lr
epochs = args.epochs
batch_size = args.batchSize
chrom = args.chr

trainAttention(device, hdf5URI, batch_size, epochs=epochs, lr=lr, skipChrom=chrom)
# print("FC")
# trainMLPtorch(device,hdf5URI,batch_size)

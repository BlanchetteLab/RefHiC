import numpy as np
from torch.utils.data import Dataset
import h5py
import torch

class indexNamedHDF5Dataset(Dataset):
    '''
    warning: does not work efficiently. Since the random access to hdf5 may be very slow
    '''
    def __init__(self, labels, indices, hdf5_path,featureMask,samples=None):
        self.indices = indices
        self.labels = labels
        self.hdf5_path = hdf5_path
        self.featureMask = featureMask
        with h5py.File(self.hdf5_path,'r') as h5:
            self.numOfSamples = h5['feature']['1']['X'][()].shape[0]
            self.numOfExtras  = h5['feature']['1']['Xs'][()].shape[0] # num of samples in database
        self.samples = samples

    def __selectSamples(self,N,n=1):
        '''
        :param N: total samples
        :param n: selected samples
        :return: masked index
        '''
        mask = np.zeros(N)
        mask[np.random.choice(N, n, replace=False)] = 1
        return np.ma.make_mask(mask)

    def __len__(self):
        return len(self.indices)

    def __open_hdf5(self):
        self.h5 = h5py.File(self.hdf5_path,'r')
        self.feature = self.h5['feature']


    def __getitem__(self, idx):
        if not hasattr(self,'h5'):
            self.__open_hdf5()
        label = self.labels[idx]
        idx = str(self.indices[idx])
        if self.numOfSamples > 1:
            X = self.feature[idx]['X'][self.__selectSamples(self.numOfSamples),:][:,self.featureMask]
        else:
            X = self.feature[idx]['X'][:, self.featureMask]
        if self.samples is not None:
            Xs = self.feature[idx]['Xs'][self.__selectSamples(self.numOfExtras,self.samples),:][:,self.featureMask]
        else:
            Xs = self.feature[idx]['Xs'][:, self.featureMask]
        X = X.flatten()
        return X,Xs,label

class diagBcoolsDataset(Dataset):
    def __init__(self, test, extra,w,resol,samples=None):

        self.test = test
        self.extra = extra
        self.numOfExtras = len(self.extra)
        self.samples = samples
        self.w = w
        self.max_distance_bins = self.test[0].max_distance_bins
        self.resol = resol

        bin2bp = dict((v, k) for k, v in self.test[0].bp2bin.items())
        loci = np.argwhere(test[0].bmatrix[:,[0]]>=0)

        loci = loci[ (loci[:,0]>3*w) & (loci[:,0]<self.test[0].bmatrix.shape[0]-w)]
        self.loci = [(bin2bp[x + self.test[0].offset], bin2bp[y + x + self.test[0].offset]) for x, y in loci]

    def __selectSamples(self,N,n=1):
        '''
        :param N: total samples
        :param n: selected samples
        :return: indices of extra samples
        '''
        return np.random.choice(N, n, replace=False)

    def __len__(self):
        return len(self.loci)

    def __getitem__(self, idx):
        xCenter,yCenter = self.loci[idx]

        X = []
        Xs = []
        for i in range(len(self.test)):
            mat, meta = self.test[i].square(xCenter, yCenter, self.w, 'b',cache=False)
            # X = np.concatenate((mat.flatten(), meta)).flatten()
            X.append(
                np.concatenate((mat.flatten(), meta))
            )

        if self.samples is not None:
            for i in self.__selectSamples(self.numOfExtras,self.samples):
                mat, meta = self.extra[i].square(xCenter, yCenter, self.w, 'b',cache=False)
                Xs.append(np.concatenate((mat.flatten(), meta)))
        else:
            for i in range(self.numOfExtras):
                mat, meta = self.extra[i].square(xCenter, yCenter, self.w, 'b',cache=False)
                Xs.append(np.concatenate((mat.flatten(), meta)))
        Xs = np.vstack(Xs)
        X = np.vstack(X)
        Xs = torch.from_numpy(Xs).float()
        X = torch.from_numpy(X).float()
        return X,Xs,(xCenter,yCenter)

class bcoolsDataset(Dataset):
    def __init__(self, test, extra,w,resol,samples=None):

        self.test = test
        self.extra = extra
        self.numOfExtras = len(self.extra)
        self.samples = samples
        self.w = w
        self.max_distance_bins = self.test[0].max_distance_bins
        self.resol = resol

        self.nne = []
        for i in range(len(test)):
            bin2bp = dict((v, k) for k, v in self.test[i].bp2bin.items())
            nne = np.argwhere(self.test[i].bmatrix > 0)

            nne = nne[(nne[:, 1] > 5) & (nne[:, 1] < self.test[i].bmatrix.shape[1] - 2 * w) & (
                        nne[:, 0] + nne[:, 1] < self.test[i].bmatrix.shape[0] - 2 * w-1)]
            nne = nne[ (nne[:,0]>3*w) & (nne[:,0]<self.test[i].bmatrix.shape[0]-2*w)]
            self.nne += [(bin2bp[x + self.test[i].offset], bin2bp[y + x + self.test[i].offset]) for x, y in nne]
        self.nne=list(dict.fromkeys(self.nne))

    def __selectSamples(self,N,n=1):
        '''
        :param N: total samples
        :param n: selected samples
        :return: indices of extra samples
        '''
        return np.random.choice(N, n, replace=False)

    def __len__(self):
        return len(self.nne)

    def __getitem__(self, idx):
        xCenter,yCenter = self.nne[idx]
        X = []
        Xs = []
        for i in range(len(self.test)):
            mat, meta = self.test[i].square(xCenter, yCenter, self.w, 'b',cache=False)
            # X = np.concatenate((mat.flatten(), meta)).flatten()
            X.append(
                np.concatenate((mat.flatten(), meta))
            )

        if self.samples is not None:
            for i in self.__selectSamples(self.numOfExtras,self.samples):
                mat, meta = self.extra[i].square(xCenter, yCenter, self.w, 'b',cache=False)
                Xs.append(np.concatenate((mat.flatten(), meta)))
        else:
            for i in range(self.numOfExtras):
                mat, meta = self.extra[i].square(xCenter, yCenter, self.w, 'b',cache=False)
                Xs.append(np.concatenate((mat.flatten(), meta)))

        Xs = np.vstack(Xs)
        X = np.vstack(X)
        Xs = torch.from_numpy(Xs).float()
        X = torch.from_numpy(X).float()
        return X,Xs,(xCenter,yCenter)

class inMemoryDataset(Dataset):
    def __init__(self, X, Xs, y,samples=None,ti=None,multiTest=False):
        self.X = X
        self.Xs = Xs
        self.y = y
        self.numOfSamples = self.X[0].shape[0]
        self.numOfExtras = self.Xs[0].shape[0]
        self.samples = samples
        self.ti = ti
        self.multiTest = multiTest

    def __selectSamples(self,N,n=1):
        '''
        :param N: total samples
        :param n: selected samples
        :return: masked index
        '''
        mask = np.zeros(N)
        mask[np.random.choice(N, n, replace=False)] = 1
        return np.ma.make_mask(mask)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        label = self.y[idx]
        if self.multiTest:
            X = self.X[idx]
        elif self.ti is not None:
            X = self.X[idx][self.ti,:]
        else:
            X = self.X[idx][self.__selectSamples(self.numOfSamples),:]

        if self.samples is not None:
            Xs = self.Xs[idx][self.__selectSamples(self.numOfExtras,self.samples),:]
        else:
            Xs = self.Xs[idx]
        if not self.multiTest:
            X = X.flatten()
        label = torch.tensor([label]).float()
        Xs = torch.from_numpy(Xs).float()
        X = torch.from_numpy(X).float()

        return idx,X,Xs,label


class matricesPatchDataset(Dataset):
    def __init__(self, featureMatrices: list, patches: list, targetMatrix=None):
        self.patches = patches
        self.featureMatrices = featureMatrices
        self.targetMatrix = targetMatrix

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        top,bottom,left,right = self.patches[idx]
        feature = []
        for i in range(len(self.featureMatrices)):
            feature.append(self.featureMatrices[i][top:bottom,left:right].toarray()[None])
        feature = np.concatenate(feature)
        if self.targetMatrix is None:
            return torch.from_numpy(feature).float()
        target = self.targetMatrix[top:bottom,left:right].toarray()
        return torch.from_numpy(feature).float(), torch.from_numpy(target).float()

class bedwriter():
    def __init__(self,file_path, resol):
        self.f = open(file_path,'w')
        self.resol = resol
    def write(self,chrom,x,prob,TADs):
        for i in range(len(x)):
            self.f.write(chrom+'\t'+str(x[i])+'\t'+str(x[i]+self.resol)
                         +'\t'+str(prob[i])
                         + '\t' + str(TADs[i])
                         +'\n')

class bedpewriter():
    def __init__(self,file_path, resol):
        self.f = open(file_path,'w')
        self.resol = resol
    def write(self,chrom,x,y,prob,val,p2ll,labels,isbad=None):
        for i in range(len(x)):
            if isbad is not None and isbad[i]:
                # print('skip bad ',labels[i])
                pass
            else:
                self.f.write(chrom+'\t'+str(x[i])+'\t'+str(x[i]+self.resol)
                             +'\t'+chrom+'\t'+str(y[i])+'\t'+str(y[i]+self.resol)
                             +'\t'+str(prob[i])
                             +'\t'+str(val[i])
                             + '\t' + str(p2ll[i])
                             + '\t' + str(labels[i])
                             +'\n')

class bcoolDataset(Dataset):
    def __init__(self, test,w,resol):

        self.test = test
        self.w = w
        self.max_distance_bins = self.test.max_distance_bins
        self.resol = resol

        bin2bp = dict((v, k) for k, v in self.test.bp2bin.items())
        nne = np.argwhere(self.test.bmatrix > 0)
        # nne = nne[ (nne[:, 1] > w+1) & (nne[:, 1] < self.test.bmatrix.shape[1] - 2*w)& (nne[:, 0]+nne[:, 1] < self.test.bmatrix.shape[0] - 2*w)]
        nne = nne[(nne[:, 1] > 5) & (nne[:, 1] < self.test.bmatrix.shape[1] - 2 * w) & (
                    nne[:, 0] + nne[:, 1] < self.test.bmatrix.shape[0] - 2 * w-1)]
        nne = nne[ (nne[:,0]>3*w) & (nne[:,0]<self.test.bmatrix.shape[0]-2*w)]
        self.nne = [(bin2bp[x + self.test.offset], bin2bp[y + x + self.test.offset]) for x, y in nne]


    def __len__(self):
        return len(self.nne)

    def __getitem__(self, idx):
        xCenter,yCenter = self.nne[idx]

        mat, meta = self.test.square(xCenter, yCenter, self.w, 'b',cache=False)
        X = np.concatenate((mat.flatten(), meta)).flatten()

        X = torch.from_numpy(X).float()
        return X,(xCenter,yCenter)


import pickle as pkl
def loadCERandomStudyPkl(file):
    # with open(file, 'rb') as pickle_file:
    #     studyMats,referencedMats,targetMat=pkl.load(pickle_file)
    #     studyMat = studyMats[[np.random.randint(studyMats.shape[0])],...]
    #     referencedMats = referencedMats[np.random.choice(referencedMats.shape[0], 10, replace=False),...]
    #     x = np.vstack((studyMat,referencedMats))
    # return torch.tensor(x)*1000,torch.tensor(targetMat)*1000
    with open(file, 'rb') as pickle_file:
        studyMats,referencedMats,targetMat=pkl.load(pickle_file)
        studyMat = studyMats[[np.random.randint(studyMats.shape[0])],...]
        referencedMats = referencedMats[[0],...]
        x = np.vstack((studyMat,referencedMats))
    return torch.tensor(x)*1000,torch.tensor(targetMat)*1000




class inMemoryCEPDataset(Dataset):
    def __init__(self, X, Xs, y,samples=None,ti=None,multiTest=False):
        self.X = X
        self.Xs = Xs
        self.y = y
        self.multiTest=multiTest
        self.numOfSamples = self.X[0].shape[0]
        self.numOfExtras = self.Xs[0].shape[0]
        self.samples = samples
        self.ti = ti

    def __selectSamples(self,N,n=1):
        '''
        :param N: total samples
        :param n: selected samples
        :return: masked index
        '''
        mask = np.zeros(N)
        mask[np.random.choice(N, n, replace=False)] = 1
        return np.ma.make_mask(mask)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        target = self.y[idx]

        if self.multiTest:
            X = self.X[idx]
        elif self.ti is not None:
            X = self.X[idx][self.ti,:]
        else:
            X = self.X[idx][self.__selectSamples(self.numOfSamples),:]
        if self.samples is not None:
            Xs = self.Xs[idx][self.__selectSamples(self.numOfExtras,self.samples),:]
        else:
            Xs = self.Xs[idx]

        target = torch.tensor(target).float()
        Xs = torch.from_numpy(Xs).float()
        X = torch.from_numpy(X).float()


        return X,Xs,target

def loadCEPkl(file):
    with open(file, 'rb') as pickle_file:
        studyMats, referencedMats, targetMat = pkl.load(pickle_file)
    return studyMats, referencedMats, targetMat
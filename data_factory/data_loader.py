import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataset.preprocessing.preprocess import UCR_AnomalySequence

def inject_trend_change(
    X,
    magnitude=(0.1, 0.5),   # (min, max)
    feature_ratio=1.0,
    time_ratio=0.01,
    win_size=100,
    seed=None,
):
    """
    Inject multiple non-overlapping random abrupt trend changes.

    Each segment has:
      - Random level-shift magnitude.
      - Random slope magnitude.
      - Random perturbation at every time point.
    """

    rng = np.random.default_rng(seed)

    X_new = X.copy()
    N, D = X.shape

    # Randomly select affected features
    n_features = max(1, int(np.ceil(feature_ratio * D)))
    feature_idx = rng.choice(D, n_features, replace=False)

    # Total perturbed samples
    total_len = max(1, int(np.ceil(time_ratio * N)))

    seg_len = min(win_size, total_len)
    n_segments = max(1, total_len // seg_len)

    possible_starts = np.arange(0, N - seg_len + 1)
    rng.shuffle(possible_starts)

    selected_starts = []

    for start in possible_starts:

        overlap = any(
            (start < s + seg_len) and (s < start + seg_len)
            for s in selected_starts
        )

        if not overlap:
            selected_starts.append(start)

        if len(selected_starts) == n_segments:
            break

    segments = []

    mag_min, mag_max = magnitude

    for start in sorted(selected_starts):

        end = start + seg_len

        # Random magnitude for THIS segment
        level_mag = rng.uniform(mag_min, mag_max)
        slope_mag = rng.uniform(mag_min, mag_max)

        # Random sign
        level_mag *= rng.choice([-1, 1])
        slope_mag *= rng.choice([-1, 1])

        # Random level at every point
        level = rng.uniform(
            -abs(level_mag),
             abs(level_mag),
            size=(seg_len, 1)
        )

        # Random slope increment at every point
        slope = np.linspace(0, slope_mag, seg_len).reshape(-1, 1)

        X_new[start:end, feature_idx] += level
        X_new[start:end, feature_idx] += slope

        segments.append({
            "start": start,
            "end": end,
            "level_magnitude": level_mag,
            "slope_magnitude": slope_mag,
        })

    return X_new, segments, feature_idx

def scalerfunc(train, test, labels, val_ratio, noise_ratio):
    N = train.shape[0]

    seg_len = int(val_ratio * N)
    start = N - seg_len
    end = start + seg_len

    val = train[start:end]
    train = np.concatenate([train[:start], train[end:]], axis=0)

    # --------------------
    # Scaling
    # --------------------
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train)

    train = scaler.transform(train).astype(np.float32)
    val   = scaler.transform(val).astype(np.float32)
    test  = scaler.transform(test).astype(np.float32)

    val  = np.clip(val, -4, 4)
    test = np.clip(test, -4, 4)

    # --------------------
    # Inject trend change ONLY into test data
    # --------------------
    if noise_ratio > 0:

        test, segments, feature_idx = inject_trend_change(
            test,
            feature_ratio=noise_ratio,
            time_ratio=noise_ratio,
            win_size=128,
            seed=42,
        )

    return train, val, test, labels


class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.0,noise_ratio=0.0):
        self.mode = mode
        self.step = step
        self.win_size = win_size

        data = pd.read_csv(data_path + '/train.csv', header=1).ffill().bfill()
        data = data.values[:, 1:-1]

        x,y=data.shape
        test_data = pd.read_csv(data_path + '/test.csv').ffill().bfill()
        y = test_data.iloc[:,-1].to_numpy()
        labels = []
        for i in y:
            if i == 'Attack':
                labels.append(1)
            elif i == 'A ttack':
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)
        test_data = test_data.values[:, 1:-1]
        data[:, [5, 10]] = 0
        test_data[:, [5, 10]] = 0
        self.train,self.val,self.test,self.test_labels=scalerfunc(data,test_data,labels.reshape(-1, 1),val_ratio,noise_ratio)

        self.dim=self.test.shape[1]
        
    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    def return_dim(self):
        return self.dim
    

class WADISegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.0,noise_ratio=0.0):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        data = pd.read_csv(data_path + '/train.csv').ffill().bfill()
        test_data = pd.read_csv(data_path + '/test.csv').ffill().bfill()
        data = data.values[:, 1:]
        test_data = test_data.values[:, 1:]
        labels = pd.read_csv(data_path + '/test_label.csv').values
        self.train,self.val,self.test,self.test_labels=scalerfunc(data,test_data,labels,val_ratio,noise_ratio)
        self.dim=test_data.shape[1]

        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    def return_dim(self):
        return self.dim

class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.0,noise_ratio=0.0):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        data = pd.read_csv(data_path + '/train.csv').ffill().bfill()
        test_data = pd.read_csv(data_path + '/test.csv').ffill().bfill()
        data = data.values[:, 1:]
        test_data = test_data.values[:, 1:]
        labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]
        self.train,self.val,self.test,self.test_labels=scalerfunc(data,test_data,labels,val_ratio,noise_ratio)
        self.dim=test_data.shape[1]

        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    def return_dim(self):
        return self.dim

class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.0,noise_ratio=0.0):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        data = pd.DataFrame(np.load(data_path + "/train.npy")).ffill().bfill().values
        test_data = pd.DataFrame(np.load(data_path + "/test.npy")).ffill().bfill().values
        labels = np.load(data_path + "/test_label.npy")
        self.train,self.val,self.test,self.test_labels=scalerfunc(data,test_data,labels,val_ratio,noise_ratio)
        self.dim=self.test.shape[1]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        
    def return_dim(self):
        return self.dim
        
class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.0,noise_ratio=0.0):
        self.mode = mode
        self.step = step
        self.win_size = win_size

        labels = []
        total_anomaly_points = 0
        pth=data_path+"/"
        labeled_anomalies = pd.read_csv(data_path + '/labeled_anomalies.csv').ffill().bfill()
        data_dims = {'SMAP': 25, 'MSL': 55}
        insert=False
        for smap_or_msl in ['SMAP']:
            for i in range(len(labeled_anomalies)):
                #print(f'  -> {labeled_anomalies["chan_id"][i]} ({i+1} / {len(labeled_anomalies)})')
                if labeled_anomalies['spacecraft'][i] == smap_or_msl:
                    # load corresponding .npy file in test and train
                    item = np.load(pth + 'train/' + labeled_anomalies['chan_id'][i] + '.npy')
                    assert item.shape[-1] == data_dims[smap_or_msl]
                    item2 = np.load(pth + 'test/' + labeled_anomalies['chan_id'][i] + '.npy')
                    assert item2.shape[-1] == data_dims[smap_or_msl]
                    labs = labeled_anomalies['anomaly_sequences'][i]
                    labs_s = labs.replace('[', '').replace(']', '').replace(' ', '').split(',')
                    labs_i = [[int(labs_s[i]), int(labs_s[i+1])] for i in range(0, len(labs_s), 2)]
                    
                    assert labeled_anomalies['num_values'][i] == len(item2)
                    #item,item2=scalerfunc(item,item2)
                    item3 = np.zeros(len(item2))
                    for sec in labs_i:
                        item3[sec[0]:sec[1]] = 1
                        total_anomaly_points += sec[1] - sec[0]

                    if insert:
                        data=np.concatenate((data,item),axis=0)
                        test_data=np.concatenate((test_data,item2),axis=0)
                        labels=np.concatenate((labels,item3),axis=0)
                    else:
                        insert=True
                        data=item
                        test_data=item2
                        labels=item3
        
        self.train,self.val,self.test,self.test_labels=scalerfunc(data,test_data,labels,val_ratio,noise_ratio)
        
        self.dim=test_data.shape[1]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    def return_dim(self):
        return self.dim
class UCRSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.0,noise_ratio=0.0):
        dataset_ind=[i for i in range(50, 75)]
        self.mode = mode
        self.step = step
        self.win_size = win_size
        insert=False
        test_data, labels = [], []
        for idx in dataset_ind:
            dataset_idx = int(idx)
            dataset_importer = UCR_AnomalySequence.create_by_id(dataset_idx)
            trn = dataset_importer.train_data[:, None]  # add channel dim; (ts_len, 1)
            tst = dataset_importer.test_data[:, None]  # (ts_len, 1)
            print(trn.shape)
            print(tst.shape)
                # anomaly data
            self.anom_start = dataset_importer.anom_start - dataset_importer.train_stop  # relative to `test_data`
            self.anom_stop = dataset_importer.anom_stop - dataset_importer.train_stop  # relative to `test_data`
            label = np.zeros_like(tst)[:, 0]  # (ts_len,)
            label[self.anom_start:self.anom_stop] = 1
        
            if insert:
                tri=np.repeat([[dataset_idx]], trn.shape[0], axis=0)
                trn=np.concatenate((trn,tri),1)
                tri=np.repeat([[dataset_idx]], tst.shape[0], axis=0)
                tst=np.concatenate((tst,tri),1)
                data=np.concatenate((data,trn),axis=0)
                test_data=np.concatenate((test_data,tst),axis=0)
                labels=np.concatenate((labels,label),axis=0)
            else:
                insert=True
                tri=np.repeat([[dataset_idx]], trn.shape[0], axis=0)
                data=np.concatenate((trn,tri),1)
                tri=np.repeat([[dataset_idx]], tst.shape[0], axis=0)
                test_data=np.concatenate((tst,tri),1)
                labels=label
        self.train,self.val,self.test,self.test_labels=scalerfunc(data,test_data,labels,val_ratio,noise_ratio)
        self.dim=self.test.shape[1]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

    def return_dim(self):
        return self.dim

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.0,noise_ratio=0.0):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        insert=False
        test_data, labels = [], []
        for ent_name in os.listdir(data_path + '/train'):
            item=pd.read_csv(data_path + '/train/' + ent_name, header=None)
            item2=pd.read_csv(data_path + '/test/' + ent_name, header=None)
            item3=np.squeeze(pd.read_csv(data_path + '/test_label/' + ent_name, header=None).to_numpy())
            
            if insert:
                data=np.concatenate((data,item),axis=0)
                test_data=np.concatenate((test_data,item2),axis=0)
                labels=np.concatenate((labels,item3),axis=0)
            else:
                insert=True
                data=item
                test_data=item2
                labels=item3
        self.train,self.val,self.test,self.test_labels=scalerfunc(data,test_data,labels,val_ratio,noise_ratio)
        self.dim=self.test.shape[1]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

    def return_dim(self):
        return self.dim

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.0 ,noise_ratio=0.0):
    '''
    model : 'train' or 'test'
    '''
    if mode == 'train' or mode=='val':
        step=step
    else:
        step=win_size

    if (dataset == 'SMD'):
        dataset= SMDSegLoader(data_path, win_size, step, mode,val_ratio,noise_ratio)
        dimret = dataset.return_dim()
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, mode,val_ratio,noise_ratio)
        dimret= dataset.return_dim()
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode,val_ratio,noise_ratio)
        dimret= dataset.return_dim()
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, mode,val_ratio,noise_ratio)
        dimret= dataset.return_dim()
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, mode,val_ratio,noise_ratio)
        dimret= dataset.return_dim()
    elif (dataset == 'UCR'):
        dataset = UCRSegLoader(data_path, win_size, step, mode,val_ratio,noise_ratio)
        dimret= dataset.return_dim()
    elif (dataset == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, step, mode,val_ratio,noise_ratio)
        dimret= dataset.return_dim()

    if mode == 'train' or mode =='val':
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        return train_loader, dimret
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16)
        
        return data_loader,dimret
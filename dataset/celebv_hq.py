import os
from abc import ABC, abstractmethod
from itertools import islice
from typing import Optional
import random
import ffmpeg
import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch import Tensor
from marlin_pytorch.util import read_video, padding_video
from util.misc import sample_indexes, read_text, read_json
import glob
import pandas as pd

class CelebvHqBase(LightningDataModule, ABC):

    def __init__(self, data_root: str, split: str, task: str, data_ratio: float = 1.0, take_num: int = None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert task in ("appearance", "action")
        self.task = task
        self.take_num = take_num
    
        self.name_list = list(
            filter(lambda x: x != "", read_text(os.path.join(data_root, f"{self.split}.txt")).split("\n")))
        self.metadata = read_json(os.path.join(data_root, "celebvhq_info.json"))

        if data_ratio < 1.0:
            self.name_list = self.name_list[:int(len(self.name_list) * data_ratio)]
        if take_num is not None:
            self.name_list = self.name_list[:self.take_num]

        print(f"Dataset {self.split} has {len(self.name_list)} videos")

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    def __len__(self):
        return len(self.name_list)
class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

# for fine-tuning
class CelebvHq(CelebvHqBase):

    def __init__(self,
        root_dir: str,
        split: str,
        task: str,
        clip_frames: int,
        temporal_sample_rate: int,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None
    ):
        super().__init__(root_dir, split, task, data_ratio, take_num)
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.img_size = 224
    
    def subsample(y, limit=256, factor=2):
        """
        If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
        """
        # if len(y) > limit:
        #     return y[::factor].reset_index(drop=True)
        # return y
        cp = random.randint(0, len(y) - 1 - limit) if limit < len(y) else 0
        return y[cp: cp + limit].reset_index(drop=True)
    def normalize_columns(tensor):
        # Clone the tensor to avoid in-place modification
        normalized_tensor = tensor.clone()
        
        # Normalize each column independently
        for i in range(tensor.size(1)):
            col_min = tensor[:, i].min()
            col_max = tensor[:, i].max()
            
            if col_min != col_max:  # Avoid division by zero
                normalized_tensor[:, i] = 2 * (tensor[:, i] - col_min) / (col_max - col_min) - 1
            else:
                normalized_tensor[:, i] = 0  # or any other value you deem appropriate
                
        return normalized_tensor
    def __getitem__(self, index: int):
        y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task]
        video_path = os.path.join(self.data_root, self.name_list[index] + ".mp4")
        # # print('dfssssssssssssssss', video_path)
        # probe = ffmpeg.probe(video_path)["streams"][0]
        # n_frames = int(probe["nb_frames"])

        # # ### Load fNIRS
        # limit = 200
        # csv_file = video_path.replace('cropped', 'fnirs').replace('mp4','csv')
        # df = pd.read_csv(csv_file)
        # df = df.dropna(axis=0)
        # series_length = len(df.index)
        # #Normalizing we simply subtract the mean and divide by standard deviation
        # # df.iloc[:,0:-1] = df.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        # # df = df.apply(lambda col: 0 if col.min() == col.max() else 2 * (col - col.min()) / (col.max() - col.min()) - 1, axis=0)
        # cp = random.randint(0, series_length - 1 - limit) if limit < series_length else 0
        # # df = df.iloc[cp: cp + limit].reset_index(drop=True)
        # fnirs = torch.tensor(df.values, dtype=torch.float32)

        # for i in range(fnirs.size(1)):
        #     col_min = fnirs[:, i].min()
        #     col_max = fnirs[:, i].max()
            
        #     if col_min != col_max:  # Avoid division by zero
        #         fnirs[:, i] = 2 * (fnirs[:, i] - col_min) / (col_max - col_min) - 1
        #     else:
        #         fnirs[:, i] = 0  # or any other value you deem appropriate
        # fnirs = fnirs[cp: cp + limit]
        # if fnirs.isnan().any():
        #     print('dddddddddddddddddddddddd', df.shape, fnirs.isnan().any())


        ### Load BioVid Signals
        limit = 400
        csv_file = video_path.replace('cropped', 'biosignals_filtered').replace('.mp4','_bio.csv')
        df = pd.read_csv(csv_file, delimiter='\t')
        df = df.dropna()
        # df = df.drop(df.columns[0], axis=1)
        df = df.drop(df.columns[0], axis=1)
        # df = df.drop(df.columns[-2:], axis=1)
        series_length = len(df.index)
        #Normalizing we simply subtract the mean and divide by standard deviation
        # df.iloc[:,0:-1] = df.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        # df = df.apply(lambda col: 0 if col.min() == col.max() else 2 * (col - col.min()) / (col.max() - col.min()) - 1, axis=0)
        cp = random.randint(0, series_length - 1 - limit) if limit < series_length else 0
        # df = df.iloc[cp: cp + limit].reset_index(drop=True)


        # predefined_max = {'gsr': 2.762477e+01, 'ecg': 5.495286e+03, 'emg_trapezius': 2.846792e+03}
        # predefined_min = {'gsr': 0.463, 'ecg': -5807.562, 'emg_trapezius': -2743.538}

        # df_normalized = df.copy()
        # for column in df.columns:
        #     df_normalized[column] = (df[column] - predefined_min[column]) / (predefined_max[column] - predefined_min[column])
        fnirs = torch.tensor(df.values, dtype=torch.float32)
        
        for i in range(fnirs.size(1)):
            col_min = fnirs[:, i].min()
            col_max = fnirs[:, i].max()
            
            if col_min != col_max:  # Avoid division by zero
                fnirs[:, i] = 2 * (fnirs[:, i] - col_min) / (col_max - col_min) - 1
            else:
                fnirs[:, i] = 0  # or any other value you deem appropriate
        # fnirs = fnirs[cp: cp + limit]
        fnirs = fnirs[cp: cp + limit]
        

        # if fnirs.isnan().any():
        #     print('dddddddddddddddddddddddd', df.shape, fnirs.isnan().any(), video_path)

        self.temporal_sample_rate = 48
        
        
        ### Load video
        # if n_frames <= self.clip_frames:
        #     video = read_video(video_path, channel_first=True).video / 255
        #     # pad frames to 16
        #     video = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
        #     video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        #     return video, torch.tensor(y, dtype=torch.long)
        # elif n_frames <= self.clip_frames * self.temporal_sample_rate:
        #     # reset a lower temporal sample rate
        #     sample_rate = n_frames // self.clip_frames
        # else:
        #     sample_rate = self.temporal_sample_rate
        # # sample frames
        # video_indexes = sample_indexes(n_frames, self.clip_frames, sample_rate)
        # reader = torchvision.io.VideoReader(video_path)
        # fps = reader.get_metadata()["video"]["fps"][0]
        # reader.seek(video_indexes[0].item() / fps, True)
        # frames = []
        
        # for frame in islice(reader, 0, self.clip_frames * sample_rate, sample_rate):
        #     frames.append(frame["data"])
        # video = torch.stack(frames) / 255  # (T, C, H, W)
        # video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        # assert video.shape[1] == self.clip_frames, video_path
        ## Load by images
        video_path = os.path.join(self.data_root, self.name_list[index])
        self.temporal_sample_rate = 1
        files = sorted(glob.glob(video_path + '/keyframe_*.jpg'))
        if len(files)<16:
            files = sorted(glob.glob(video_path + '/*.jpg'))
            files = [file for file in files if 'keyframe' not in file]
            self.temporal_sample_rate = 8
        indexes = self._sample_indexes(len(files))
        assert len(indexes) == self.clip_frames
        video = torch.zeros(self.clip_frames, self.img_size, self.img_size, 3, dtype=torch.float32)
        for i in range(self.clip_frames):
            img = cv2.imread(os.path.join(files[indexes[i]]))
            video[i] = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255)
        video = rearrange(video, "t h w c -> c t h w")
        # return video, torch.tensor(y, dtype=torch.long).bool()
        return video, fnirs, torch.tensor(y, dtype=torch.long)
    
    def _sample_indexes(self, num_frames: int) -> Tensor:
        return sample_indexes(num_frames, self.clip_frames, self.temporal_sample_rate)

class CelebvHq_Test(CelebvHqBase):

    def __init__(self,
        root_dir: str,
        split: str,
        task: str,
        clip_frames: int,
        temporal_sample_rate: int,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None
    ):
        super().__init__(root_dir, split, task, data_ratio, take_num)
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate

    def __getitem__(self, index: int):
        # y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task]
        video_path = os.path.join(self.data_root, self.name_list[index] + ".mp4")
        probe = ffmpeg.probe(video_path)["streams"][0]
        n_frames = int(probe["nb_frames"])

        if n_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True).video / 255
            # pad frames to 16
            video = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
            return video
        elif n_frames <= self.clip_frames * self.temporal_sample_rate:
            # reset a lower temporal sample rate
            sample_rate = n_frames // self.clip_frames
        else:
            sample_rate = self.temporal_sample_rate
        # sample frames
        video_indexes = sample_indexes(n_frames, self.clip_frames, sample_rate)
        reader = torchvision.io.VideoReader(video_path)
        fps = reader.get_metadata()["video"]["fps"][0]
        reader.seek(video_indexes[0].item() / fps, True)
        frames = []
        cp = random.randint(0, n_frames - 1 - self.clip_frames) if self.clip_frames < n_frames else 0
        for frame in islice(reader, 0, self.clip_frames * sample_rate, sample_rate):
            frames.append(frame["data"])
        video = torch.stack(frames) / 255  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        assert video.shape[1] == self.clip_frames, video_path
        # return video, torch.tensor(y, dtype=torch.long).bool()
        return video, video_path


# For linear probing
class CelebvHqFeatures(CelebvHqBase):

    def __init__(self, root_dir: str,
        feature_dir: str,
        split: str,
        task: str,
        temporal_reduction: str,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None
    ):
        super().__init__(root_dir, split, task, data_ratio, take_num)
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction

    def __getitem__(self, index: int):
        feat_path = os.path.join(self.data_root, self.name_list[index] + ".npy")

        x = torch.from_numpy(np.load(feat_path)).float()

        if x.size(0) == 0:
            x = torch.zeros(1, 768, dtype=torch.float32)

        if self.temporal_reduction == "mean":
            x = x.mean(dim=0)
        elif self.temporal_reduction == "max":
            x = x.max(dim=0)[0]
        elif self.temporal_reduction == "min":
            x = x.min(dim=0)[0]
        else:
            raise ValueError(self.temporal_reduction)

        y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task]

        # return x, torch.tensor(y, dtype=torch.long).bool()
        return x, torch.tensor(y, dtype=torch.long)


class CelebvHqDataModule(LightningDataModule):

    def __init__(self, root_dir: str,
        load_raw: bool,
        task: str,
        batch_size: int,
        num_workers: int = 0,
        clip_frames: int = None,
        temporal_sample_rate: int = None,
        feature_dir: str = '',
        temporal_reduction: str = "mean",
        data_ratio: float = 1.0,
        take_train: Optional[int] = None,
        take_val: Optional[int] = None,
        take_test: Optional[int] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.load_raw = load_raw
        # self.load_raw = False
        self.data_ratio = data_ratio
        self.take_train = take_train
        self.take_val = take_val
        self.take_test = take_test

        if load_raw:
            assert clip_frames is not None
            assert temporal_sample_rate is not None
        else:
            assert feature_dir is not None
            assert temporal_reduction is not None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.load_raw:
            self.train_dataset = CelebvHq(self.root_dir, "train", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_train)
            self.val_dataset = CelebvHq(self.root_dir, "val", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_val)
            self.test_dataset = CelebvHq_Test(self.root_dir, "test", self.task, self.clip_frames,
                self.temporal_sample_rate, 1.0, self.take_test)
        else:
            self.train_dataset = CelebvHqFeatures(self.root_dir, self.feature_dir, "train", self.task,
                self.temporal_reduction, self.data_ratio, self.take_train)
            self.val_dataset = CelebvHqFeatures(self.root_dir, self.feature_dir, "val", self.task,
                self.temporal_reduction, self.data_ratio, self.take_val)
            self.test_dataset = CelebvHqFeatures(self.root_dir, self.feature_dir, "test", self.task,
                self.temporal_reduction, 1.0, self.take_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda x: collate_test(x),
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_test(x),
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

### Masking fNIRS
def collate_test(data):
    batch_size = len(data)
    video, features, y = zip(*data)
    video = torch.stack(video, dim=0)
    y = torch.stack(y, dim=0)

    # fNIRS masking
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series

    max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    
    return video, X, padding_masks, y

    print('maskmaskmaskmaskmaskmaskmask', features.shape, y.shape)
def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks, IDs

def transduct_mask(X, mask_feats, start_hint=0.0, end_hint=0.0):
    """
    Creates a boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        mask_feats: list/array of indices corresponding to features to be masked
        start_hint:
        end_hint: proportion at the end of time series which will not be masked

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """

    mask = np.ones(X.shape, dtype=bool)
    start_ind = int(start_hint * X.shape[0])
    end_ind = max(start_ind, int((1 - end_hint) * X.shape[0]))
    mask[start_ind:end_ind, mask_feats] = 0

    return mask


def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


def collate_unsuperv(data, max_len=None, mask_compensation=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, masks, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(X,
                                    dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks, IDs


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim tha

import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from utility.data_loader import DataObject
import torch


import sys
sys.path.append("/home/jiazhen/emotiv-ml")
from axon.ml import Corpus
from axon.ml import Dict

def pretrain_corpus(config):
    """
    Load or construct the binary-tree based corpus dataset for pretraining.
    """
    corpus_config = Dict({
        'write_path'    : '/work/corpus/subcorpus/epoc/corpus-subset{}-{}.db',
        'read_path'     : config.corpus_data_path,
        'distribution'  : [1.0, 0.0, 0.0],
        'batch_size'    : config.batch_size,
        'hz'            : config.hertz,
        'channels'      : config.channels,
        'seconds'       : config.seconds,
        'stride'        : config.stride,
        'num_samples'   : int(config.seconds * config.hertz),
        'is_torch'      : True
    })

    def get_signature():
        dir_path = '/work/corpus/subcorpus/epoc'
        subset_str = str(config.corpus_subset)
        pattern = f"corpus-subset{subset_str}-"
        for fname in os.listdir(dir_path):
            if fname.startswith(pattern) and fname.endswith(".db"):
                return fname[len(pattern):-3]  # 去掉后缀 .db
        raise FileNotFoundError(f"No corpus file found for subset {subset_str} in {dir_path}")

    # Get corpus file path
    subset = config.corpus_subset
    signature = get_signature()
    fname = corpus_config.write_path.format(subset, signature)
    print(f'[INFO] Loading corpus from: {fname}')

    # Try load or build
    try:
        corpus = Corpus.load(fname)
        print(f'[INFO] Corpus loaded successfully.')
    except Exception as e:
        print(f'[WARNING] Failed to load corpus: {e}, building new one...')
        corpus = Corpus(
            path=corpus_config.read_path,
            recLen=int(corpus_config.channels * corpus_config.hz * corpus_config.seconds),
            skipLen=int(corpus_config.channels * corpus_config.stride),
            numSamples=corpus_config.num_samples,
            targets=corpus_config.distribution,
            batchSize=corpus_config.batch_size,
            isTorch=corpus_config.is_torch
        )
        corpus.save(fname)
        print(f'[INFO] Corpus saved to: {fname}')

    # old one
    # data = Dict()
    # data['pretrain_data_list'] = corpus.data  
    # data['pretrain_data_list_clean'] = getattr(corpus, 'data2', corpus.data)
    # data['shape'] = [1, config.channels, config.seconds * config.hertz]
    # data['num_labels'] = 2  
    # return data

    # JH 2025-0530 for EEGMS
    tensor_data = torch.from_numpy(
        np.stack([np.squeeze(d, axis=0) for d in getattr(corpus, 'data2', corpus.data)])
    ).float()  # -> shape: (N, C, T)
    dataset = TensorDataset(tensor_data, torch.zeros(len(tensor_data)))  # dummy label

    # Split data & build dataloader
    test_ratio = getattr(config, "test_ratio", 0.0)
    val_ratio = getattr(config, "val_ratio", 0.2)
    train_ratio = 1.0 - val_ratio - test_ratio
    total_len = len(dataset)
    train_len, val_len = int(train_ratio * total_len), int(val_ratio * total_len)
    test_len = total_len - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return DataObject(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_df=pd.read_excel(config.Emotiv_coord_path, index_col=0),  # Input montage
        target_df=pd.read_excel(config.Standard_coord_path, index_col=0),  # Target montage
        in_channels=config.channels,
        in_times=config.seconds * config.hertz
    )
    
 
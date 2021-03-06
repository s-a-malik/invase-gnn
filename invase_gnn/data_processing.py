"""Functions for loading/generating graph data.
"""

# import packages
from sklearn.model_selection import train_test_split as split

from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T


# TODO synthetic task
def syn1_data(seed=0):
    """Synthetic graph dataset 
    Task: Graph Classification - with 

    Returns:
    dataset - PyG dataset
    """
    raise NotImplementedError()


def load_data(task, seed, val_size, test_size):
    """Load dataset
    Task: Graph Classification - benchmark datasets. Consisting of various graphs we want to classify.

    Returns:
    - train_dataset: PyG dataset
    - val_dataset: PyG dataset
    - test_dataset: PyG dataset
    - test_idx: indices of the original dataset used for testing - to recover original data.
    """
    path = "../data/TUDataset"
    
    if task == 'mutag':
        dataset = 'Mutagenicity'
        dataset = TUDataset(path, dataset, transform=T.NormalizeFeatures(), cleaned=True)
    elif task == 'enzymes':
        dataset = 'ENZYMES'
        dataset = TUDataset(path, dataset, transform=T.NormalizeFeatures(), cleaned=True)
    elif task == 'proteins':
        dataset = 'PROTEINS'
        dataset = TUDataset(path, dataset, transform=T.NormalizeFeatures(), cleaned=True)
    else:
         NameError(f"task {args.task} not allowed")
    
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    indices = [i for i in range(len(dataset))]
    train_idx, test_idx = split(indices, random_state=seed, test_size=test_size)
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]

    indices = [i for i in range(len(train_dataset))]
    train_idx, val_idx = split(indices, random_state=seed, test_size=val_size/(1-test_size))
    val_dataset = train_dataset[val_idx]
    train_dataset = train_dataset[train_idx]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of val graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset, test_idx
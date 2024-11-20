from torch.utils.data import DataLoader
# from DataSet.MyDataset import CsvDataset, MatDataset, NpzDataset
from MyDataset import *


def get_dataloader(model_config: dict):
    dataset_name = model_config['dataset_name']


    if dataset_name in ['arrhythmia', 'breastw', 'cardio', 'glass', 'ionosphere', 'mammography', 'pima', 'satellite', 'satimage-2', 'shuttle', 'thyroid', 'wbc']:
        train_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')

    elif dataset_name in ['census', 'campaign', 'cardiotocography', 'fraud', 'nslkdd', 'optdigits', 'pendigits', 'wine']:
        train_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')
        
    else:
        train_set = CsvDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = CsvDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')

    train_loader = DataLoader(train_set, batch_size=model_config['batch_size'], num_workers=model_config['num_workers'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)
    return train_loader, test_loader


def main():
    model_config = {
    'dataset_name': 'wbc',
    'data_dim': 30,
    'epochs': 200,
    'learning_rate': 0.05,
    'sche_gamma': 0.98,
    'mask_num': 15,
    'lambda': 5,
    'device': 'cuda:0',
    'data_dir': '../Data/',
    'runs': 1,
    'batch_size': 512, 
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 256,
    'z_dim': 128,
    'mask_nlayers': 3,
    'random_seed': 42,
    'num_workers': 0
    }
    train_loader, test_loader = get_dataloader(model_config)
    # 遍历 train_loader 并打印数据的形状
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, (list, tuple)):  # 如果数据是列表或元组
            print(f"Batch {batch_idx}:")
            for i, d in enumerate(data):
                print(f"  Element {i} shape: {d.shape}")
        else:  # 如果数据是单个张量
            print(f"Batch {batch_idx} shape: {data.shape}")
    
    # 如果只需要查看第一个 batch，退出循环
        break

main()
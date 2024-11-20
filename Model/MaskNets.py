import torch
import torch.nn as nn

# self.maskmodel = Generator(MultiNets(), model_config)
# 使用多个子网络(masks)对输入数据进行处理
# 生成处理后的数据以及对应的掩码
class Generator(nn.Module):
    def __init__(self, model, config):
        super(Generator, self).__init__()
        self.masks = model._make_nets(config['data_dim'], config['mask_nlayers'], config['mask_num'])
        self.mask_num = config['mask_num']
        self.device = config['device']

    def forward(self, x):
        # print(f"input shape: {x.shape}")
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0], self.mask_num, x.shape[-1]).to(x)
        # print(f"initialize x_T shape: {x_T.shape}")
        # initialize an empty tensor
        
        masks = []
        for i in range(self.mask_num): # ensemble
            mask = self.masks[i](x)
            # print(f"mask{i} shape: {mask.shape}")
            mask_unsqueeze = mask.unsqueeze(1)
            # print(f"mask unsqueeze{i} shape: {mask_unsqueeze.shape}")
            masks.append(mask_unsqueeze)
            mask = torch.sigmoid(mask) # 对特征维度进行操作
            x_T[:, i] = mask * x # X \cdot M_k
            # print(f"mask * x: {mask * x}")
        masks = torch.cat(masks, axis=1)
        return x_T, masks


# 构造一个特征generator
class SingleNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(SingleNet, self).__init__()
        net = []
        input_dim = x_dim
        # input_dim and output_dim are x_dim
        # hidden_dim is always h_dim
        # 上一层的输出是下一层的输入
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim, h_dim, bias=False))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim, x_dim, bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


# ensemble method
class MultiNets():
    def _make_nets(self, x_dim, mask_nlayers, mask_num):
        multinets = nn.ModuleList([SingleNet(x_dim, x_dim, mask_nlayers) for _ in range(mask_num)])
        return multinets
    
    
""" def main():
    model_config = {
    'dataset_name': 'wbc',
    'data_dim': 4,
    'epochs': 200,
    'learning_rate': 0.05,
    'sche_gamma': 0.98,
    'mask_num': 3,
    'lambda': 5,
    'device': 'cuda:0',
    'data_dir': 'Data/',
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
    x = torch.rand((2, 4))
    # x = x.to(model_config['device'])
    print(f"x: {x}")
    print(f"x shape: {x.shape}")
    gene = Generator(MultiNets(), model_config)
    x_T, masks = gene(x)
    print(f"x_T shape: {x_T.shape}")
    print(f"x_T: {x_T}")
    print(f"masks shape: {masks.shape}")
    print(f"masks: {masks}")
main() """
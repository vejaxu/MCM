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
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0], self.mask_num, x.shape[-1]).to(x)
        masks = []
        for i in range(self.mask_num):
            mask = self.masks[i](x)
            masks.append(mask.unsqueeze(1))
            mask = torch.sigmoid(mask)
            x_T[:, i] = mask * x
        masks = torch.cat(masks, axis=1)
        return x_T, masks


class SingleNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(SingleNet, self).__init__()
        net = []
        input_dim = x_dim
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


class MultiNets():
    def _make_nets(self, x_dim, mask_nlayers, mask_num):
        multinets = nn.ModuleList([SingleNet(x_dim, x_dim, mask_nlayers) for _ in range(mask_num)])
        return multinets
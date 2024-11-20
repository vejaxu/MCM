import torch
import torch.nn as nn
from torch.autograd import Variable

from Model.MaskNets import MultiNets, Generator


class MCM(nn.Module):
    def __init__(self, model_config):
        super(MCM, self).__init__()
        self.data_dim = model_config['data_dim'] # input dim
        self.hidden_dim = model_config['hidden_dim']
        self.z_dim = model_config['z_dim'] # encoder-decoder hidden layer z's dim
        self.mask_num = model_config['mask_num'] # ensemble
        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.maskmodel = Generator(MultiNets(), model_config)

        # encoder input dim = self.data_dim
        encoder = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim, self.hidden_dim, bias=False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim

        encoder.append(nn.Linear(encoder_dim, self.z_dim, bias=False))
        self.encoder = nn.Sequential(*encoder)

        # decoder input dim = encoder output dim: self.z_dim
        decoder = []
        decoder_dim = self.z_dim
        for _ in range(self.de_nlayers-1):
            decoder.append(nn.Linear(decoder_dim, self.hidden_dim,bias=False))
            decoder.append(nn.LeakyReLU(0.2, inplace=True))
            decoder_dim = self.hidden_dim

        decoder.append(nn.Linear(decoder_dim, self.data_dim, bias=False))
        self.decoder = nn.Sequential(*decoder)
        # decoder output dim = self.data_dim
        # then we can calculate the reconstruction loss

    def forward(self, x_input):
        # print(f"MCM input shape: {x_input.shape}")
        x_mask, masks = self.maskmodel(x_input) # Generator(x_input)
        # x_mask shape (x_input[0], num_masks, x_input[-1])
        B, T, D = x_mask.shape # batch_size, mask_num, dim
        x_mask = x_mask.reshape(B*T, D)
        
        z = self.encoder(x_mask)
        x_pred = self.decoder(z)
        
        z = z.reshape(x_input.shape[0], self.mask_num, z.shape[-1])
        x_pred = x_pred.reshape(x_input.shape[0], self.mask_num, x_input.shape[-1])
        return x_pred, z, masks

    def print_weight(self, x_input):
        x_input = Variable(x_input, requires_grad=False)
        z = self.encoder(x_input)
        fea_mem = self.fea_mem(z)
        fea_att_w = fea_mem['att']
        out = torch.max(fea_att_w, dim=0).view(8, 8).detach().cpu().numpy()
        return out
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import wargs

class ConvLayer(nn.Module):

    def __init__(self, in_feats_num, ffs, fws):

        super(ConvLayer, self).__init__()

        self.conv1 = nn.Conv1d(in_feats_num, ffs[0], kernel_size=fws[0], stride=1, padding=(fws[0]-1)/2)
        self.lr1 = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm1d(ffs[0])

        #nn.Conv1d(in_channels=ffs[i], out_channels=ffs[i], kernel_size=fws[i], stride=2, padding=1),
        self.conv2 = nn.Conv1d(ffs[0], ffs[0], kernel_size=fws[0], stride=1, padding=(fws[0]-1)/2)
        self.lr2 = nn.LeakyReLU(0.1)
        self.bn2 = nn.BatchNorm1d(ffs[0])

        #nn.Conv1d(in_channels=ffs[i], out_channels=ffs[i], kernel_size=fws[i], stride=2, padding=1),
        self.conv3 = nn.Conv1d(ffs[0], ffs[0], kernel_size=fws[0], stride=1, padding=(fws[0]-1)/2)
        self.lr3 = nn.LeakyReLU(0.1)
        self.bn3 = nn.BatchNorm1d(ffs[0])

        #nn.Conv1d(in_channels=ffs[i], out_channels=ffs[i], kernel_size=fws[i], stride=2, padding=1),
        self.conv4 = nn.Conv1d(ffs[0], ffs[0], kernel_size=fws[0], stride=1, padding=(fws[0]-1)/2)
        self.lr4 = nn.LeakyReLU(0.1)
        self.bn4 = nn.BatchNorm1d(ffs[0])

    def forward(self, x):

        """convolution"""
        # (B, enc_size, L) -> (B, feats_size[i], L)
        #x = [(self.cnns[_i](x)) for _i in range(self.n_layers)]
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        #x = tc.cat(x, dim=1)
        # (B, sum_feats_size, L')
        x = self.conv1(x)
        x = self.lr1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.lr2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.lr3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.lr4(x)
        x = self.bn4(x)

        return x

class RelationLayer(nn.Module):

    def __init__(self, input_size, output_size, filter_window_size, filter_feats_size, mlp_size=128):

        super(RelationLayer, self).__init__()

        self.C_in = input_size

        self.fws = filter_window_size
        self.ffs = filter_feats_size
        self.N = len(self.fws)
        self.mlp_size = mlp_size

        self.convLayer = ConvLayer(input_size, filter_feats_size, filter_window_size)
        cnn_feats_size = sum([k for k in self.ffs])

        self.g_mlp = nn.Sequential(
            nn.Linear(2 * (cnn_feats_size+2) + wargs.enc_hid_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1)
        )

        self.f_mlp = nn.Sequential(
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, wargs.dec_hid_size),
            nn.LeakyReLU(0.1)
        )

    # prepare coord tensor
    def cvt_coord(self, idx, L):
        return [( idx / np.sqrt(L) - 2 ) / 2., ( idx % np.sqrt(L) - 2 ) / 2.]

    def forward(self, x, h, xs_mask=None):

        L, B, E = x.size()
        x = x.permute(1, 2, 0)    # (B, E, L)

        ''' CNN Layer '''
        x = self.convLayer(x)   # (B, sum_feats_size, L')
        L = x.size(-1)

        # (B, sum_feats_size, L) -> (B, L, sum_feats_size)
        x = x.permute(0, 2, 1)
        # (B, L, sum_feats_size)

        self.coord_tensor = tc.FloatTensor(B, L, 2)
        if wargs.gpu_id is not None: self.coord_tensor = self.coord_tensor.cuda(wargs.gpu_id[0])
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((B, L, 2))
        for _i in range(L): np_coord_tensor[:, _i, :] = np.array( self.cvt_coord(_i, L) )
        self.coord_tensor.data.copy_(tc.from_numpy(np_coord_tensor))

        # add coordinates
        x = tc.cat([x, self.coord_tensor], dim=2)
        # (B, L, sum_feats_size+2)

        # add question everywhere
        h = tc.unsqueeze(h, 1)      # (B, E) -> (B, 1, E)
        h = h.repeat(1, L, 1)     # (B, 1, E) -> (B, L, E)
        h = tc.unsqueeze(h, 2)      # (B, L, E) -> (B, L, 1, E)

        # cast all pairs against each other
        x_i = tc.unsqueeze(x, 1)        # (B, 1, L, sum_feats_size+2)
        x_i = x_i.repeat(1, L, 1, 1)    # (B, L, L, sum_feats_size+2)

        x_j = tc.unsqueeze(x, 2)        # (B, L, 1, sum_feats_size+2)
        x_j = tc.cat([x_j, h], 3)       # (B, L, 1, sum_feats_size+2)
        x_j = x_j.repeat(1, 1, L, 1)    # (B, L, L, sum_feats_size+2+E)

        # concatenate all together
        x = tc.cat([x_i, x_j], 3)       # (B, L, L, 2*(sum_feats_size+2)+E)

        ''' Graph Propagation Layer '''
        #if xs_mask is not None: xs_h = xs_h * xs_mask[:, :, None]
        #x = x[None, :, :, :].expand(L, L, B, self.cnn_feats_size)
        #x = tc.cat([x, x.transpose(0, 1)], dim=-1)

        x = self.g_mlp(x)
        #if xs_mask is not None: xs_h = xs_h * xs_mask[:, :, None]

        x = x.view(B, L*L, self.mlp_size)    # (B, L*L, mlp_size)
        x = x.sum(1).squeeze()

        ''' MLP Layer '''
        return self.f_mlp(x)


class RelationLayer_Old(nn.Module):

    def __init__(self, input_size, output_size, filter_window_size, filter_feats_size, mlp_size=128):

        super(RelationLayer, self).__init__()

        self.C_in = 1

        self.fws = filter_window_size
        self.ffs = filter_feats_size
        self.N = len(self.fws)

        '''
            nn.Sequential(
                nn.Conv1d(self.C_in, self.ffs[i], kernel_size=output_size*self.fws[i],
                          padding=((self.fws[i]-1)/2) * output_size, stride=output_size),
                nn.BatchNorm2d(self.ffs[i]),
                nn.LeakyReLU(0.1),
                nn.Conv1d(self.ffs[i], self.ffs[i], kernel_size=output_size*self.fws[i],
                          padding=((self.fws[i]-1)/2) * output_size, stride=output_size),
                nn.BatchNorm2d(self.ffs[i]),
                nn.LeakyReLU(0.1),
                nn.Conv1d(self.ffs[i], self.ffs[i], kernel_size=output_size*self.fws[i],
                          padding=((self.fws[i]-1)/2) * output_size, stride=output_size),
                nn.BatchNorm2d(self.ffs[i]),
                nn.LeakyReLU(0.1),
                nn.Conv1d(self.ffs[i], self.ffs[i], kernel_size=output_size*self.fws[i],
                          padding=((self.fws[i]-1)/2) * output_size, stride=output_size),
                nn.BatchNorm2d(self.ffs[i]),
                nn.LeakyReLU(0.1)
            )
        modules = []
        for i in range(self.N):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(self.C_in, self.ffs[i], kernel_size=output_size*self.fws[i],
                              padding=((self.fws[i]-1)/2) * output_size, stride=output_size),
                    nn.BatchNorm1d(self.ffs[i]),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(self.ffs[i], self.ffs[i], kernel_size=output_size*self.fws[i],
                              padding=((self.fws[i]-1)/2) * output_size, stride=output_size),
                    nn.BatchNorm1d(self.ffs[i]),
                    nn.LeakyReLU(0.1)
                )
            )
        '''
        #self.cnnlayer = nn.ModuleList([nn.Conv2d(self.C_in, self.C_out, (k, input_size),
        #                                      padding=((k-1)/2, 0)) for k in kernels])
        self.cnnlayer = nn.ModuleList([nn.Conv1d(self.C_in, self.ffs[i],
                                                 kernel_size=output_size*self.fws[i],
                                                 padding=((self.fws[i]-1)/2) * output_size,
                                                 stride=output_size) for i in range(self.N)])
        #self.cnnlayer = nn.ModuleList(modules)
        # (B, in, enc_size * L) -> (B, feats_size[i], L)

        self.bns = nn.ModuleList([nn.BatchNorm1d(self.ffs[i]) for i in range(self.N)])

        self.leakyRelu = nn.LeakyReLU(0.1)
        #self.bn = nn.BatchNorm1d(mlp_dim)

        self.cnn_feats_size = sum([k for k in self.ffs])

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.cnn_feats_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1)
        )

        self.mlp_layer = nn.Sequential(
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_size, output_size),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, xs_mask=None):

        L, B, E = x.size()
        x = x.permute(1, 0, 2)    # (B, L, E)

        ''' CNN Layer '''
        # (B, 1, L, E)
        #x = x[:, None, :, :].expand((B, self.C_in, L, E))
        # (B, L, E) -> (B, E*L) -> (B, 1, E*L)
        x = x.contiguous().view(B, -1)[:, None, :]

        # (B, feats_size[i], L, 1) -> (B, feats_size[i], L)
        #x = [self.leakyRelu(conv(x)).squeeze(3) for conv in self.cnnlayer]
        # (B, in, enc_size * L) -> (B, feats_size[i], L)
        x = [self.leakyRelu(self.bns[i](self.cnnlayer[i](x))) for i in range(self.N)]
        #x = [self.leakyRelu(self.cnnlayer[i](x)) for i in range(self.N)]

        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = tc.cat(x, dim=1)
        # (B, Sum_feats_size[i], L) -> (L, B, Sum_feats_size[i])
        x = x.permute(2, 0, 1)
        #(L, B, Sum_feats_size[i])

        ''' Graph Propagation Layer '''
        #if xs_mask is not None: xs_h = xs_h * xs_mask[:, :, None]
        x = x[None, :, :, :].expand(L, L, B, self.cnn_feats_size)
        x = tc.cat([x, x.transpose(0, 1)], dim=-1)

        x = self.mlp(x).sum(0)
        #if xs_mask is not None: xs_h = xs_h * xs_mask[:, :, None]

        ''' MLP Layer '''
        return self.mlp_layer(x)



""" 
    Function:  CRNN 
    Refs:  
"""

import torch.nn as nn


class CnnBlock(nn.Module):
    """ Function: Basic convolutional block
        reference: resnet, https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
    """
    # expansion = 1
    def __init__(self, inplanes, planes, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=True, downsample=None):
        super(CnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=(1,1), padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.use_res = use_res

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_res == True:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out

class TCnnBlock(nn.Module):
    """ Function: Basic time convolutional block
    """
    # expansion = 1
    def __init__(self, inplanes, planes, kernel=3, stride=1, padding=1, use_res=True, downsample=None):
        super(TCnnBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride
        self.use_res = use_res

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_res == True:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out

# CRNN
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class crnn(nn.Module):
    """ CRNN
	"""
    def __init__(self, nf = 256, cnn_inplanes=4, planes=[64,64,128,256,512], f_stride = [1,1,4,4,4], res_flag=False, rnn_nlayer=1, rnn_bdflag =True, out_dim=256):
        super(crnn, self).__init__()

        if res_flag:
            downsample0 = nn.Sequential(
                conv1x1(cnn_inplanes, planes[0], stride=(f_stride[0],1)),
                nn.BatchNorm2d(planes[0]),
            )
            downsample1 = nn.Sequential(
                conv1x1(planes[0], planes[1], stride=(f_stride[1],1)),
                nn.BatchNorm2d(planes[1]),
            )
            downsample2 = nn.Sequential(
                conv1x1(planes[1], planes[2], stride=(f_stride[2],1)),
                nn.BatchNorm2d(planes[2]),
            )
            if len(f_stride)>3:
                downsample3 = nn.Sequential(
                    conv1x1(planes[2], planes[3], stride=(f_stride[3],1)),
                    nn.BatchNorm2d(planes[3]),
                )
                if len(f_stride)>4:
                    downsample4 = nn.Sequential(
                        conv1x1(planes[3], planes[4], stride=(f_stride[4],1)),
                        nn.BatchNorm2d(planes[4]),
                    )
        else:
            downsample0 = None
            downsample1 = None
            downsample2 = None
            if len(f_stride)>3:
                downsample3 = None
                if len(f_stride)>4:
                    downsample4 = None
        self.precnn = nn.Sequential(
            # nn.Conv2d(cnn_inplanes, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True)
            CnnBlock(cnn_inplanes, planes[0], kernel=(3, 3), stride=(f_stride[0], 1), padding=(1, 1), use_res=res_flag, downsample=downsample0),
        )
        if len(f_stride)==3:
            self.cnn = nn.Sequential(
                CnnBlock(planes[0], planes[1], kernel=(3, 3), stride=(f_stride[1], 1), padding=(1, 1), use_res=res_flag, downsample=downsample1),
                CnnBlock(planes[1], planes[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),

                CnnBlock(planes[1], planes[2], kernel=(3, 3), stride=(f_stride[2], 1), padding=(1, 1), use_res=res_flag, downsample=downsample2),
                CnnBlock(planes[2], planes[2], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            )
        elif len(f_stride)==4:
            self.cnn = nn.Sequential(
                CnnBlock(planes[0], planes[1], kernel=(3, 3), stride=(f_stride[1], 1), padding=(1, 1), use_res=res_flag, downsample=downsample1),
                CnnBlock(planes[1], planes[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),

                CnnBlock(planes[1], planes[2], kernel=(3, 3), stride=(f_stride[2], 1), padding=(1, 1), use_res=res_flag, downsample=downsample2),
                CnnBlock(planes[2], planes[2], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),

                CnnBlock(planes[2], planes[3], kernel=(3, 3), stride=(f_stride[3], 1), padding=(1, 1), use_res=res_flag, downsample=downsample3),
                CnnBlock(planes[3], planes[3], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            )
        elif len(f_stride)==5:
            self.cnn = nn.Sequential(
                # nn.MaxPool2d(kernel_size=(2, 1)),
                # nn.AvgPool2d(kernel_size=(4, 1))
                # nn.AvgPool2d(kernel_size=(2, 1))
                CnnBlock(planes[0], planes[1], kernel=(3, 3), stride=(f_stride[1], 1), padding=(1, 1), use_res=res_flag, downsample=downsample1),
                CnnBlock(planes[1], planes[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
                # nn.MaxPool2d(kernel_size=(4, 1)),
                # nn.AvgPool2d(kernel_size=(4, 1)),
                CnnBlock(planes[1], planes[2], kernel=(3, 3), stride=(f_stride[2], 1), padding=(1, 1), use_res=res_flag, downsample=downsample2),
                CnnBlock(planes[2], planes[2], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
                # nn.MaxPool2d(kernel_size=(4, 1)),
                # nn.AvgPool2d(kernel_size=(4, 1)),
                CnnBlock(planes[2], planes[3], kernel=(3, 3), stride=(f_stride[3], 1), padding=(1, 1), use_res=res_flag, downsample=downsample3),
                CnnBlock(planes[3], planes[3], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
                # nn.MaxPool2d(kernel_size=(4, 1)),
                # nn.AvgPool2d(kernel_size=(4, 1)),
                CnnBlock(planes[3], planes[4], kernel=(3, 3), stride=(f_stride[4], 1), padding=(1, 1), use_res=res_flag, downsample=downsample4),
                CnnBlock(planes[4], planes[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            )
       
        # self.rnn_bdflag = rnn_bdflag
        if rnn_bdflag:
            rnn_ndirection = 2
        else:
            rnn_ndirection = 1
        rnn_in_dim = nf*planes[len(f_stride)-1]
        for s in f_stride:
            rnn_in_dim = int(rnn_in_dim/s)
        print('rnn_in_dim', rnn_in_dim, 'f_strid', f_stride, 'planes', planes)
        rnn_hid_dim = int(rnn_in_dim/rnn_ndirection)

        # self.rnn = nn.LSTM(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=2, batch_first=True,
        #                                  bias=True, dropout=0.4, bidirectional=rnn_bdflag)
        self.rnn = nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=rnn_nlayer, batch_first=True, bias=True, dropout=0.4, bidirectional=rnn_bdflag)

        self.rnn_fc = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=rnn_ndirection * rnn_hid_dim, out_features=out_dim),  # ,bias=False
            # nn.Tanh(),
        )

    def forward(self, x):
        # x: (nb, nch, nf, nt)
        nb = x.shape[0]
        fea = self.precnn(x)
        fea_cnn = self.cnn(fea)  # (nb, nplanes, nf, nt) 
        fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3))  # (nb, nplanes*nf,nt) 
        fea_rnn_in = fea_rnn_in.permute(0, 2, 1)  # (nb, nt, nfea)
        fea_rnn, _ = self.rnn(fea_rnn_in)
        # fea_rnn = fea_rnn_in
        fea_rnn_fc = self.rnn_fc(fea_rnn) # (nb, nt, *)
        return fea_rnn_fc

class crnn_sim(nn.Module):
    """ CRNN: each conv layer has the same channels
	"""
    def __init__(self, cnn_inplanes=4, res_flag=False, conv_chs=64, rnn_in_dim=256, rnn_hid_dim=256, rnn_nlayer=1, rnn_bdflag =True):
        super(crnn_sim, self).__init__()
        
        self.precnn = nn.Sequential(
            CnnBlock(cnn_inplanes, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=False),
            nn.MaxPool2d(kernel_size=(4, 1))
        )
        
        self.cnn = nn.Sequential(
            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),

            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),

            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            nn.MaxPool2d(kernel_size=(2, 1)),

            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
            CnnBlock(conv_chs, conv_chs, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag),
        )
       
        # self.rnn_bdflag = rnn_bdflag
        if rnn_bdflag:
            rnn_ndirection = 2
        else:
            rnn_ndirection = 1
        # self.rnn = nn.LSTM(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=2, batch_first=True,
        #                                  bias=True, dropout=0.4, bidirectional=rnn_bdflag)
        self.rnn = nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=rnn_nlayer, batch_first=True, bias=True, dropout=0.4, bidirectional=rnn_bdflag)

        self.rnn_fc = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=rnn_ndirection * rnn_hid_dim, out_features=rnn_hid_dim),  # ,bias=False
            # nn.Tanh(),
        )

    def forward(self, x):
        # x: (nb, nch, nf, nt)
        nb = x.shape[0]
        fea = self.precnn(x)
        fea_cnn = self.cnn(fea)  # (nb, nplanes, nf, nt) 
        fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3))  # (nb, nplanes*nf,nt) 
        fea_rnn_in = fea_rnn_in.permute(0, 2, 1)  # (nb, nt, nfea)
        fea_rnn, _ = self.rnn(fea_rnn_in)
        fea_rnn_fc = self.rnn_fc(fea_rnn) # (nb, nt, *)
        return fea_rnn_fc


# TCRNN
def tconv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class tcrnn(nn.Module):
    """ CRNN
	"""
    def __init__(self, cnn_inplanes=4, planes=[1024], res_flag=False, rnn_nlayer=1, rnn_bdflag =True, out_dim=256):
        super(tcrnn, self).__init__()

        if res_flag:
            downsample0 = nn.Sequential(
                tconv1x1(cnn_inplanes, planes[0]),
                nn.BatchNorm1d(planes[0]),
            )
            downsample1 = nn.Sequential(
                tconv1x1(planes[0], planes[1]),
                nn.BatchNorm1d(planes[1]),
            )
            if len(planes)>=3:
                downsample2 = nn.Sequential(
                    tconv1x1(planes[1], planes[2]),
                    nn.BatchNorm1d(planes[2]),
                )
                if len(planes)>=4:
                    downsample3 = nn.Sequential(
                        tconv1x1(planes[2], planes[3]),
                        nn.BatchNorm1d(planes[3]),
                    )
                    if len(planes)>=5:
                        downsample4 = nn.Sequential(
                            tconv1x1(planes[3], planes[4]),
                            nn.BatchNorm1d(planes[4]),
                        )
        else:
            downsample0 = None
            downsample1 = None
            downsample2 = None
            if len(planes)>=4:
                downsample3 = None
                if len(planes)>=5:
                    downsample4 = None

        self.precnn = nn.Sequential(
            TCnnBlock(cnn_inplanes, planes[0], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample0),
        )
        if len(planes)==2:
            self.cnn = nn.Sequential(
                TCnnBlock(planes[0], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample1),
                TCnnBlock(planes[1], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag),
                
            )        
        if len(planes)==3:
            self.cnn = nn.Sequential(
                TCnnBlock(planes[0], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample1),
                TCnnBlock(planes[1], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag),
                
                TCnnBlock(planes[1], planes[2], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample2),
                TCnnBlock(planes[2], planes[2], kernel=3, stride=1, padding=1, use_res=res_flag),
            )
        if len(planes)==4:
            self.cnn = nn.Sequential(
                TCnnBlock(planes[0], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample1),
                TCnnBlock(planes[1], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag),
                
                TCnnBlock(planes[1], planes[2], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample2),
                TCnnBlock(planes[2], planes[2], kernel=3, stride=1, padding=1, use_res=res_flag),
                
                TCnnBlock(planes[2], planes[3], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample3),
                TCnnBlock(planes[3], planes[3], kernel=3, stride=1, padding=1, use_res=res_flag),
            )
        if len(planes)==5:
            self.cnn = nn.Sequential(
                TCnnBlock(planes[0], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample1),
                TCnnBlock(planes[1], planes[1], kernel=3, stride=1, padding=1, use_res=res_flag),
                
                TCnnBlock(planes[1], planes[2], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample2),
                TCnnBlock(planes[2], planes[2], kernel=3, stride=1, padding=1, use_res=res_flag),
                
                TCnnBlock(planes[2], planes[3], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample3),
                TCnnBlock(planes[3], planes[3], kernel=3, stride=1, padding=1, use_res=res_flag),
                
                TCnnBlock(planes[3], planes[4], kernel=3, stride=1, padding=1, use_res=res_flag, downsample=downsample4),
                TCnnBlock(planes[4], planes[4], kernel=3, stride=1, padding=1, use_res=res_flag),
            )
       
        # self.rnn_bdflag = rnn_bdflag
        if rnn_bdflag:
            rnn_ndirection = 2
        else:
            rnn_ndirection = 1
        rnn_in_dim = planes[-1]
        print('rnn_in_dim', rnn_in_dim, 'planes', planes)
        rnn_hid_dim = int(rnn_in_dim/rnn_ndirection)
        # self.rnn = nn.LSTM(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=2, batch_first=True,
        #                                  bias=True, dropout=0.4, bidirectional=rnn_bdflag)
        self.rnn = nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=rnn_nlayer, batch_first=True, bias=True, dropout=0.4, bidirectional=rnn_bdflag)

        self.rnn_fc = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=rnn_ndirection * rnn_hid_dim, out_features=out_dim),  # ,bias=False
            # nn.Tanh(),
        )

    def forward(self, x):
        # x: (nb, nch_in, nf, nt)
        nb, nch, nf, nt = x.shape
        fea = x.view(nb, -1, nt)
        fea = self.precnn(fea)
        fea_cnn = self.cnn(fea)  # (nb, nplanes, nt) 
        fea_rnn_in = fea_cnn.permute(0, 2, 1)  # (nb, nt, nplanes)
        fea_rnn, _ = self.rnn(fea_rnn_in)
        fea_rnn_fc = self.rnn_fc(fea_rnn) # (nb, nt, *)

        return fea_rnn_fc

if __name__ == "__main__":
    import torch
    input = torch.randn((100, 4, 256, 256)) 
    net = crnn(cnn_inplanes=input.shape[1],res_flag=False, rnn_in_dim=256*2, rnn_hid_dim=256, rnn_nlayer=1, rnn_bdflag =True)
    # net = tcrnn(cnn_inplanes=input.shape[1]*input.shape[2],res_flag=False, rnn_in_dim=256*2, rnn_hid_dim=256, rnn_nlayer=1, rnn_bdflag =True)
    output = net(input)
    print(output.shape)
    print('# parameters:', sum(param.numel() for param in net.parameters())/1000000)
import torch
import torch.nn as nn

class FNblock(nn.Module):
    def __init__(self, ori_in_size, hidden_size=256, dropout=0.2, is_online=False, is_first_block=False):
        """the block of full-band and narrow-band fusion
        """
        super(FNblock, self).__init__()

        self.is_first_block = is_first_block
        self.is_online = is_online
        hidden_size_full = hidden_size // 2
        if is_online:
            hidden_size_narr = hidden_size
        else:
            hidden_size_narr = hidden_size  // 2
        self.dropout_full = nn.Dropout(p=dropout)
        self.dropout_narr = nn.Dropout(p=dropout)
        self.fullLstm = nn.LSTM(input_size=ori_in_size, hidden_size=hidden_size_full, batch_first=True, bidirectional=True)
        if self.is_first_block:
            self.narrLstm = nn.LSTM(input_size=hidden_size+ori_in_size, hidden_size=hidden_size_narr, batch_first=True, bidirectional=not self.is_online)
        else:
            self.narrLstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size_narr, batch_first=True, bidirectional=not self.is_online)
        
    def forward(self, x, fullband_skip_in=None, narrband_skip_in=None):
        nb, nf, nt, nc = x.shape
        narrband_skip_in = x.reshape(nb*nf, nt, -1)
        x = x.reshape(nb*nt, nf, -1)
        if not self.is_first_block:
            x = x + fullband_skip_in
        x, _ = self.fullLstm(x)
        
        fullband_skip_out = x #.permute(0, 2, 1)
        x = self.dropout_full(x)
        x = x.view(nb, nt, nf, -1).permute(0, 2, 1, 3).reshape(nb*nf, nt, -1)
        if self.is_first_block:  
            x = torch.cat((x, narrband_skip_in),dim=-1)
        else:
            x = x + narrband_skip_in
        x, _ = self.narrLstm(x)
        narrband_skip_out = x #.permute(0, 2, 1)
        x = self.dropout_narr(x)
        x = x.view(nb, nf, nt, -1) #.permute(0, 2, 1, 3)
        return x, fullband_skip_out, narrband_skip_out

       
class FN_SSL(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, is_online=True, is_doa=False):
        """the block of full-band and narrow-band fusion
        """
        super(FN_SSL, self).__init__()
        self.is_online = is_online
        self.is_doa = is_doa
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock(ori_in_size=self.input_size, hidden_size=hidden_size, is_online=self.is_online, is_first_block=True)
        self.block_2 = FNblock(ori_in_size=self.hidden_size, hidden_size=hidden_size, is_online=self.is_online, is_first_block=False)
        self.block_3 = FNblock(ori_in_size=self.hidden_size, hidden_size=hidden_size, is_online=self.is_online, is_first_block=False)        
        self.emb2ipd = nn.Linear(hidden_size, 2)
        self.pooling = nn.AvgPool2d(kernel_size=(12, 1))
        self.tanh = nn.Tanh()
        if self.is_doa:
            self.ipd2doa = nn.Linear(hidden_size*2, 180)

    def forward(self, x): # x:(nb, nc, nf, nt)
        x = x.permute(0, 2, 3, 1)
        nb, nf, nt, nc = x.shape       
        x, fb_skip, nb_skip = self.block_1(x)
        x, fb_skip, nb_skip = self.block_2(x, fullband_skip_in=fb_skip, narrband_skip_in=nb_skip)
        x, fb_skip, nb_skip = self.block_3(x, fullband_skip_in=fb_skip, narrband_skip_in=nb_skip)  # (nb, nf, nt, hid_dim)
        x = x.reshape(nb*nf, nt, -1)   
        ipd = self.pooling(x)
        ipd = self.tanh(self.emb2ipd(ipd))
        _, nt2, _ = ipd.shape
        ipd = ipd.view(nb, nf, nt2, -1)
        ipd = ipd.permute(0, 2, 1, 3)
        ipd_real = ipd[:, :, :, 0]
        ipd_image = ipd[:, :, :, 1]
        result = torch.cat((ipd_real, ipd_image), dim=2)
        if self.is_doa:
            result = self.ipd2doa(result)
        return result

# class FNblock(nn.Module):
#     def __init__(self, input_size, hidden_size=256, dropout=0.2, is_online=False, is_first=False):
#         """the block of full-band and narrow-band fusion
#         """
#         super(FNblock, self).__init__()
#         self.input_size = input_size
#         self.full_hidden_size =  hidden_size // 2
#         self.is_first = is_first
#         self.is_online = is_online
#         if self.is_online:
#             self.narr_hidden_size = hidden_size
#         else:
#             self.narr_hidden_size = hidden_size  // 2
#         self.dropout = dropout

#         self.dropout_full =  nn.Dropout(p=self.dropout)
#         self.dropout_narr = nn.Dropout(p=self.dropout)
#         self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
#         if self.is_first:
#               self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size+self.input_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)
#         else:
#             self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)
        
#     def forward(self, x, nb_skip=None, fb_skip=None):
#         #shape of x: nb,nv,nf,nt
#         nb,nt,nf,nc = x.shape
#         nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,-1)
#         x = x.reshape(nb*nt,nf,-1)
#         if not self.is_first:
#             x = x + fb_skip
#         x, _ = self.fullLstm(x)
#         fb_skip = x
#         x = self.dropout_full(x)
#         x = x.view(nb,nt,nf,-1).permute(0,2,1,3).reshape(nb*nf,nt,-1)
#         if self.is_first:  
#             x = torch.cat((x,nb_skip),dim=-1)
#         else:
#             x = x + nb_skip
#         x, _ = self.narrLstm(x)
#         nb_skip = x
#         x = self.dropout_narr(x)
#         x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
#         return x, fb_skip, nb_skip

       
# class FN_SSL(nn.Module):
#     """ 
#     """
#     def __init__(self,input_size=4,hidden_size=256,is_online=True,is_doa=False):
#         """the block of full-band and narrow-band fusion
#         """
#         super(FN_SSL, self).__init__()
#         self.is_online = is_online
#         self.is_doa = is_doa
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.block_1 = FNblock(input_size=self.input_size,is_online=self.is_online, is_first=True)
#         self.block_2 = FNblock(input_size=self.hidden_size,is_online=self.is_online, is_first=False)
#         self.block_3 = FNblock(input_size=self.hidden_size,is_online=self.is_online, is_first=False)        
#         self.emb2ipd = nn.Linear(256,2)
#         self.pooling = nn.AvgPool2d(kernel_size=(12, 1))
#         self.tanh = nn.Tanh()
#         if self.is_doa:
#             self.ipd2doa = nn.Linear(512,180)
#     def forward(self,x):
#         x = x.permute(0,3,2,1)
#         nb,nt,nf,nc = x.shape       
#         x, fb_skip, nb_skip = self.block_1(x)
#         x, fb_skip, nb_skip = self.block_2(x,fb_skip=fb_skip, nb_skip=nb_skip)
#         x, fb_skip, nb_skip = self.block_3(x,fb_skip=fb_skip, nb_skip=nb_skip)  
#         #nb nt nf nc
#         x = x.permute(0,2,1,3).reshape(nb*nf,nt,-1)   
#         ipd = self.pooling(x)
#         ipd = self.tanh(self.emb2ipd(ipd))
#         _, nt2, _ = ipd.shape
#         ipd = ipd.view(nb,nf,nt2,-1)
#         ipd = ipd.permute(0,2,1,3)
#         ipd_real = ipd[:,:,:,0]
#         ipd_image = ipd[:,:,:,1]
#         result = torch.cat((ipd_real,ipd_image),dim=2)
#         if self.is_doa:
#             result = self.ipd2doa(result)
#         return result
    

if __name__ == "__main__":
	import torch
	input = torch.randn((1, 4, 256, 298)).cuda()
	net = FN_SSL().cuda()
	# ouput = net(input)
	# print(ouput.shape)
	print('# parameters:', sum(param.numel() for param in net.parameters()))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import common.utils_module as at_module

# Different architechture modules 
from common.Conformer import ConformerEncoder
from common.Transformer import TransformerEncoder
from common.CRNN import crnn, crnn_sim, tcrnn
from common.NBC import NBC
from common.FNSSL import FNblock
from common.UNet import UNet
from common.CNN import resnet50, res2net50, densenet121
# from ks.model_ds_compare import CausCRNN_predhead

class EmbedEncoder(nn.Module):

    def __init__(self, sig_shape, patch_shape, dembed, model=['cnn', 'conformer'], mode='spat', use_cls=False, device='cpu'):
        super(EmbedEncoder, self).__init__()

        self.sig_shape = sig_shape
        self.patch_shape = patch_shape
        self.dembed = dembed
        self.device = device
        self.model = model
        self.use_cls = use_cls
            
        nf, nt, nreim, nmic = sig_shape
        nch = nreim*nmic
        npatch_shape = [int(nf / patch_shape[0]), int(nt / patch_shape[1])]
        dpatch = patch_shape[0] * patch_shape[1]
        npatch = npatch_shape[0] * npatch_shape[1]
        dembed_in = dpatch*nreim*nmic

        if len(model) == 2:
            if mode == 'spec':
                mhsa_nlayer = 1
                conv_chs = 64
            elif mode == 'spat':
                mhsa_nlayer = 3
                conv_chs = 64

            # Local TF processing
            if model[0] == 'fc':
                self.patch_proj = nn.Linear(dembed_in, dembed)
            elif model[0] == 'cnn':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.patch_embed = nn.Sequential(
                    nn.Conv2d(nch, conv_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, nch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                    nn.BatchNorm2d(nch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(nch), dembed, kernel_size=patch_shape, stride=patch_shape, padding=0, bias=False),
                )
            elif model[0] == 'cnn_f_first':
                self.patch_recover = at_module.PatchRecover(output_shape=(nt, nf), patch_shape=[self.patch_shape[-1], self.patch_shape[0]])
                self.patch_embed = nn.Sequential(
                    nn.Conv2d(nch, conv_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, nch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                    nn.BatchNorm2d(nch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(nch), dembed, kernel_size=[patch_shape[-1], patch_shape[0]], stride=[patch_shape[-1], patch_shape[0]], padding=0, bias=False),
                ) 

            # Global TF processing
            if (model[1] == 'transformer') | (model[1] == 'conformer'):
                if self.use_cls:
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dembed))
                    trunc_normal_(self.cls_token, std=.02)
                    nseq = npatch + 1
                else:
                    nseq = npatch
                if model[1] == 'transformer':
                    self.embed = TransformerEncoder(d_model=self.dembed, nlayer=mhsa_nlayer, nhead=4, d_ff=self.dembed*4, dropout=0.1)
                elif model[1] == 'conformer':
                    self.embed = ConformerEncoder(encoder_dim=self.dembed, num_layers=mhsa_nlayer, num_attention_heads=4, feed_forward_expansion_factor=4)

        elif len(model) == 1:
            if model[0] == 'crnn':
                if mode == 'spec':
                    conv_chs = 32
                    f_stride = [1,4,4]
                    res_flag = True
                    rnn_nlayer = 1
                elif mode == 'spat':
                    conv_chs = 16
                    f_stride = [1,1,4,4,4]
                    res_flag = True
                    rnn_nlayer = 1
               
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.crnn = crnn(nf=nf, 
                                cnn_inplanes=nch, 
                                planes=[conv_chs,conv_chs,conv_chs*2,conv_chs*4,conv_chs*8], 
                                f_stride=f_stride, 
                                res_flag=res_flag, 
                                rnn_nlayer=rnn_nlayer, 
                                rnn_bdflag =True,
                                out_dim=dembed)
            elif model[0] == 'crnn-sim':
                if mode == 'spec':
                    rnn_nlayer = 1
                    conv_chs = 64
                elif mode == 'spat':
                    rnn_nlayer = 1
                    conv_chs = 64
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.crnn = crnn_sim(cnn_inplanes=nch, 
                                res_flag=res_flag, 
                                conv_chs=conv_chs, 
                                rnn_in_dim=256*2, 
                                rnn_hid_dim=dembed, 
                                rnn_nlayer=rnn_nlayer, 
                                rnn_bdflag =True)
            elif model[0] == 'tcrnn':
                if mode == 'spec':
                    rnn_nlayer = 1
                    conv_chs = [256*2, 256]
                    res_flag = True
                elif mode == 'spat':
                    rnn_nlayer = 1
                    conv_chs = [256, 256, 128]
                    res_flag = True
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.crnn = tcrnn(cnn_inplanes=nch*nf, 
                                planes = conv_chs,
                                res_flag=res_flag, 
                                rnn_nlayer=rnn_nlayer, 
                                rnn_bdflag =True,
                                out_dim=dembed)
            elif model[0] == 'fn':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                dim = 128
                self.embed1 = FNblock(ori_in_size=nch, hidden_size=dim, is_online=False, is_first_block=True) 
                self.embed2 = FNblock(ori_in_size=dim, hidden_size=dim, is_online=False, is_first_block=False) 
                # self.embed3 = FNblock(ori_in_size=dim, hidden_size=dim, is_online=False, is_first_block=False) 
                self.embed_proj = nn.Linear(dim, nch)
                self.t_embed_proj = nn.Linear(nch*nf, dembed)
            elif model[0] == 'nbc':
                if mode == 'spec':
                    mhsa_nlayer = 1
                elif mode == 'spat':
                    mhsa_nlayer = 3
                if self.use_cls:
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dembed))
                    trunc_normal_(self.cls_token, std=.02)
                    nseq = npatch + 1
                else:
                    nseq = npatch
                self.embed = NBC(hidden_size=self.dembed, n_layers=mhsa_nlayer, n_heads=4, ffn_size=self.dembed*4)
            elif model[0] == 'spatialnet':
                pass

            elif model[0] == 'resnet':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = resnet50(in_planes=nch)
                self.cnn_proj = nn.Conv2d(1024, dembed, kernel_size=(1, 1), stride=(1, 1), padding=0)
            elif model[0] == 'res2net':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = res2net50(in_planes=nch)
                self.cnn_proj = nn.Conv2d(1024, dembed, kernel_size=(1, 1), stride=(1, 1), padding=0)
            elif model[0] == 'densenet':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = densenet121(in_planes=nch)
                self.cnn_proj = nn.Conv2d(1024, dembed, kernel_size=(1, 1), stride=(1, 1), padding=0)
            elif model[0] == 'unet':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = UNet(n_channels=nch, n_classes=1)
                self.cnn_proj = nn.Conv2d(1, dpatch, kernel_size=patch_shape, stride=patch_shape, padding=0, bias=False)
            else:
                pass

        else:
            raise Exception('Unrecognized submodel~')
        
    def forward(self, embed, add_same_one=False):
        # embed: (nbatch, npatch, dpatch*2*nmic)
        nbatch, npatch, dim = embed.shape
        dpatch = self.patch_shape[0]*self.patch_shape[1]
        nch = int(dim/dpatch)

        if len(self.model) == 2:
            if self.model[0] == 'fc':
                embed = self.patch_proj(embed)  # (nbatch, npatch, dembed)
            elif (self.model[0] == 'cnn'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.patch_embed(embed)  # (nbatch, dpatch*nch, ..., ...)
                embed = embed.reshape(nbatch, embed.shape[1], npatch).permute(0, 2, 1)  # (nbatch, npatch, d_patch*nch) / (nbatch, npatch, dembed)
            elif (self.model[0] == 'cnn_f_first'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nt, nf, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nt, nf)
                embed = self.patch_embed(embed)  # (nbatch, dpatch*nch, ..., ...)
                embed = embed.reshape(nbatch, embed.shape[1], npatch).permute(0, 2, 1)  # (nbatch, npatch, d_patch*nch) / (nbatch, npatch, dembed)

            if (self.model[1] == 'transformer') | (self.model[1] == 'conformer'):
                if self.use_cls:
                    cls_token = self.cls_token.expand(nbatch, -1, -1)
                    embed = torch.cat((embed, cls_token), dim=1)
                embed = self.embed.forward(embed, add_same_one)  # (nbatch, npatch, dembed)
        
        if len(self.model) == 1:
            if (model[0] == 'crnn') | (model[0] == 'crnn-sim') | (model[0] == 'tcrnn'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.crnn(embed)  # (nbatch, nt, dembed) = (nbatch, npatch, *) suited for frame-wise patch spliting only
            elif model[0] == 'fn':
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed, fb_skip, nb_skip = self.embed1(embed) # (nb, nf, nt, hid_dim)
                embed, fb_skip, nb_skip = self.embed2(embed, fullband_skip_in=fb_skip, narrband_skip_in=nb_skip) # (nb, nf, nt, hid_dim)
                # embed, fb_skip, nb_skip = self.embed3(embed, fullband_skip_in=fb_skip, narrband_skip_in=nb_skip) # (nb, nf, nt, hid_dim)
                embed = self.embed_proj(embed) # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 2, 1, 3) # (nbatch, nt, nf, nch)
                embed = embed.reshape(nbatch, embed.shape[1], -1)
                embed = self.t_embed_proj(embed)
            elif model[0] == 'nbc':
                if self.use_cls:
                    cls_token = self.cls_token.expand(nbatch, -1, -1)
                    embed = torch.cat((embed, cls_token), dim=1)
                embed = self.embed.forward(embed, add_same_one)  # (nbatch, npatch, dembed)
            elif model[0] == 'spatialnet':
                pass                
            elif (self.model[0] == 'resnet') | (model[0] == 'res2net') | (model[0] == 'densenet'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.cnn(embed)  # (nbatch, nch', nf, nt)  
                embed = self.cnn_proj(embed)  # (nbatch, dembed, nf, nt) 
                embed = embed.reshape(nbatch, embed.shape[1], -1).permute(0, 2, 1)  #  (nbatch, npatch, dembed)
            elif model[0] == 'unet':
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.cnn(embed)  # (nbatch, 1, nf, nt)  
                embed = self.cnn_proj(embed)  # (nbatch, dpatch, npatch_f, npatch_t) 
                embed = embed.reshape(nbatch, embed.shape[1], -1).permute(0, 2, 1)  #  (nbatch, npatch, dpatch)
            else:
                pass

        return embed

class EmbedDecoder(nn.Module):

    def __init__(self, sig_shape, patch_shape, dembed, model=['', 'fc'], use_cls=False):
        super(EmbedDecoder, self).__init__()
        self.model = model
        self.use_cls = use_cls
       
        nf, nt, nreim, nmic = sig_shape
        nch = nreim*nmic
        dpatch = patch_shape[0] * patch_shape[1]
        npatch_shape = [int(nf / patch_shape[0]), int(nt / patch_shape[1])]
        npatch = npatch_shape[0] * npatch_shape[1]

        self.dpatch = dpatch
        dembed_out = dpatch*nreim*nmic
        
        if (model[0] == 'transformer') | (model[0] == 'conformer'):
            if self.use_cls:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, dembed))
                trunc_normal_(self.cls_token, std=.02)
                nseq = npatch + 1
            else:
                nseq = npatch
            mhsa_nlayer = 1
        if model[0] == 'transformer':
            self.embed = TransformerEncoder(d_model=dembed, nlayer=mhsa_nlayer, nhead=4, d_ff=dembed*4, nseq=nseq, dropout=0.1)
        elif model[0] == 'conformer':
            self.embed = ConformerEncoder(encoder_dim=dembed, num_layers=mhsa_nlayer, num_attention_heads=4, feed_forward_expansion_factor=4)
        elif model[0] == '':
            pass

        if model[1] == 'fc':
            dff = dembed_out * 3
            print(dff)
            self.proj = nn.Sequential(
                nn.Linear(dembed, dff), 
                nn.ReLU(), 
                nn.Linear(dff, dembed_out)
                )
        elif model[1] == 'cnn':
            conv_chs = 64
            self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=patch_shape)
            self.proj = nn.Sequential(
                 nn.Conv2d(int(dembed/npatch), conv_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            	nn.BatchNorm2d(conv_chs),
            	nn.ReLU(inplace=True),
                nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            	nn.BatchNorm2d(conv_chs),
            	nn.ReLU(inplace=True),
                nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            	nn.BatchNorm2d(conv_chs),
            	nn.ReLU(inplace=True),
                nn.Conv2d(conv_chs, nch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            	nn.BatchNorm2d(nch),
            	nn.ReLU(inplace=True),
                nn.Conv2d(nch, dembed_out, kernel_size=patch_shape, stride=patch_shape, padding=0, bias=False),
                )

    def forward(self, embed, add_same_one=False):
        nbatch = embed.shape[0]

        if (self.model[0] == 'transformer') | (self.model[0] == 'conformer'):
            if self.use_cls:
                cls_token = self.cls_token.expand(nbatch, -1, -1)
                embed = torch.cat((embed, cls_token), dim=1)
            embed = self.embed(embed, add_same_one)  # (nbatch, npatch, dembed)

        elif self.model[0] == '':
            pass

        if self.model[1] == 'fc':
            embed = self.proj(embed)  # (nbatch, npatch, dpatch)/(nbatch, npatch, dpatch*nch)

        elif self.model[1] == 'cnn':
            npatch = embed.shape[1]
            embed = embed.reshape(nbatch, npatch, self.dpatch, -1) # (nbatch, npatch, dpatch, x）
            embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, x)
            embed = embed.permute(0, 3, 1, 2)  # (nbatch, x, nf, nt)
            embed = self.proj(embed)  # (nbatch, dembed_out/npatch, ..., ...)
            embed = embed.reshape(nbatch, embed.shape[1], npatch).permute(0, 2, 1)  # (nbatch, npatch, dembed_out)

        return embed

 
class SARSSL(nn.Module): 

    def __init__(self, sig_shape=[256,256,2,2], patch_shape=(256,1), patch_mode='T', nmasked_patch=128*1, pretrain=True, use_cls=False, downstream_token='all', downstream_head='mlp', downstream_embed='spec_spat', downstream_dlabel=1, device='cpu', pretrain_frozen_encoder=False):#vis=False):
        # patch_shape: size of each patch, (nf, nt)
        # patch_shape=(16,16), maskedpatch_mode='TF',
        # patch_shape=(256,1), maskedpatch_mode='T'
        super(SARSSL, self).__init__()
        
        nf, nt, nreim, nmic = sig_shape
        print('patch_shape: ', patch_shape)
        npatch_shape = [int(nf / patch_shape[0]), int(nt / patch_shape[1])]
        dpatch = patch_shape[0] * patch_shape[1]
        if nmasked_patch != (npatch_shape[0]*npatch_shape[1]//2):
            nmasked_patch = npatch_shape[0]*npatch_shape[1]//2
            assert nmasked_patch<npatch_shape[0]*npatch_shape[1], 'number of patches is out of range~'
            print('number of masks patches', nmasked_patch ,'/',npatch_shape[0]*npatch_shape[1])
        if patch_shape[1]!=1:
            f_first = True
        else:
            f_first = False

        self.pretrain = pretrain
        self.pretrain_frozen_encoder = pretrain_frozen_encoder
        self.device = device
        self.use_cls = use_cls

        self.patch_split = at_module.PatchSplit(patch_shape=patch_shape, f_first=f_first)
        self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=patch_shape, f_first=f_first)

        spec_dembed = 256*2
        spat_dembed = 256
        self.in_ver = 'separate' # separate mask for spatial and spectral encoders
        # self.in_ver = 'same' # same mask for spatial and spectral encoders
        # self.in_ver = 'single_ch_each_patch' 
        print('version:', self.in_ver)

        if f_first:
            ## MC-Conformer (CNN_Fre_First+Conformer)
            spec_model = ['cnn_f_first', 'conformer']
            spat_model = ['cnn_f_first', 'conformer']
            print('fre first!')
        else:
            ## MC-Conformer (CNN+Conformer)
            spec_model = ['cnn', 'conformer']
            spat_model = ['cnn', 'conformer']
            ## Conformer
            # spec_model = ['fc', 'conformer']
            # spat_model = ['fc', 'conformer']
            ## CNN + Transformer
            # spec_model = ['cnn', 'transformer']
            # spat_model = ['cnn', 'transformer']
            ## Transformer
            # spec_model = ['fc', 'transformer']
            # spat_model = ['fc', 'transformer']

            ## CRNN
            # spec_model = ['crnn']
            # spat_model = ['crnn']
            ## CRNN-Simple
            # spec_model = ['crnn-sim']
            # spat_model = ['crnn-sim']
            ## TCRNN 
            # spec_model = ['tcrnn']
            # spat_model = ['tcrnn']
            # spec_encoder_dembed = 256
            # spat_encoder_dembed = int(256/2)

            ## ?? need to modify
            ## NBC 
            # spec_model = ['cnn', 'conformer']
            # spat_model = ['nbc']
            ## FN
            # spec_model = ['cnn', 'conformer']
            # spat_model = ['fn']
            ## SpatialNet
            # spec_model = ['cnn', 'conformer']
            # spat_model = ['spatialnet']

            ## ?? need to modify
            ## ResNet
            # spec_model = ['resnet']
            # spat_model = ['resnet']
            ## Res2Net
            # spec_model = ['res2net']
            # spat_model = ['res2net']
            ## DenseNet
            # spec_model = ['densenet']
            # spat_model = ['densenet']
            ## UNnet
            # spec_model = ['']
            # spat_model = ['unet']

        # Ablation study - only spatial encoder
        # spec_model = ['fc', '']
        # spec_encoder_dembed = 256*2 

        self.embed_use4ds = downstream_embed

        if (self.in_ver == 'separate') | (self.in_ver == 'same'):
            self.spec_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spec_dembed, model=spec_model, 
                                            mode='spec', use_cls=use_cls, device=self.device)
            self.spat_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spat_dembed, model=spat_model, 
                                            mode='spat', use_cls=use_cls, device=self.device)

        elif self.in_ver == 'single_ch_each_patch':
            self.spec_encoder = EmbedEncoder(sig_shape=[nf*nmic, nt, nreim, 1], patch_shape=patch_shape, dembed=int(spec_dembed/nmic), model=spec_model,
                                            mode='spec', use_cls=use_cls, device=self.device)
            self.spat_encoder = EmbedEncoder(sig_shape=[nf*nmic, nt, nreim, 1], patch_shape=patch_shape, dembed=int(spat_dembed/nmic),  model=spat_model, 
                                            mode='spat', use_cls=use_cls, device=self.device)

        if self.pretrain:
            self.patch_mask = at_module.PatchMask(patch_mode=patch_mode, nmasked_patch=nmasked_patch, npatch_shape=npatch_shape, device=self.device)
            
            if use_cls:
                spec_dembed = spec_dembed + 1 
                spat_dembed = spat_dembed + 1         
            dec_dembed = spec_dembed + spat_dembed

            dec_model = ['', 'fc']
            # dec_model = ['conformer', 'fc']
            self.decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dec_dembed, model=dec_model, use_cls=False)
        
        elif pretrain_frozen_encoder:
            self.patch_mask = at_module.PatchMask(patch_mode=patch_mode, nmasked_patch=nmasked_patch, npatch_shape=npatch_shape, device=self.device)
            
            if use_cls:
                spec_dembed = spec_dembed + 1
                spec_dembed = spat_dembed + 1 
            dec_dembed = spec_dembed + spat_dembed

            de_model = ['', 'fc']
            self.spec_spat_decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dec_dembed, model=de_model, use_cls=False)
            self.spec_decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spec_dembed, model=de_model, use_cls=False)
            self.spat_decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spec_dembed, model=de_model, use_cls=False)

        else:
            # specify downstream embedding
            if self.embed_use4ds == 'spec_spat':
                dembed_ds = spec_dembed + spat_dembed
            elif self.embed_use4ds == 'spec':
                dembed_ds = spec_dembed+0
            elif self.embed_use4ds == 'spat':
                dembed_ds =  spat_dembed +0
            elif self.embed_use4ds == 'noinfo':
                dembed_ds = spec_dembed+0

            # specify pred head for fine-tuning
            if downstream_head == 'mlp':
                if downstream_dlabel == 1:
                    self.mlp_head = nn.Sequential(
                        nn.LayerNorm(dembed_ds), 
                        nn.Linear(dembed_ds, downstream_dlabel)
                        )
                else:
                    self.joint_head = nn.Sequential(
                    nn.LayerNorm(dembed_ds), 
                    nn.Linear(dembed_ds, dembed_ds),
                    nn.ReLU(),
                    nn.Linear(dembed_ds, downstream_dlabel)
                    )
            # elif downstream_head == 'crnn_in':
            #     self.pred_head = CausCRNN_predhead(cnn_in_dim=nmic*nreim+int(dembed_ds/256), cnn_dim = 64, rnn_in_dim = 256, rnn_hid_dim = 256, fc_in_dim = 256, fc_out_dim = 1, res_flag = False, embed_add = 'in')
            # elif downstream_head == 'crnn_med':
            #     self.pred_head = CausCRNN_predhead(cnn_in_dim=nmic*nreim, cnn_dim = 64, rnn_in_dim = 256+dembed_ds, rnn_hid_dim = 256, fc_in_dim = 256, fc_out_dim = 1, res_flag = False, embed_add = 'med')
            # elif downstream_head == 'crnn_out':
            #     self.pred_head = CausCRNN_predhead(cnn_in_dim=nmic*nreim, cnn_dim = 64, rnn_in_dim = 256, rnn_hid_dim = 256, fc_in_dim = 256+dembed_ds, fc_out_dim = 1, res_flag = False, embed_add = 'out')

            self.downstream_head = downstream_head
            self.downstream_dlabel = downstream_dlabel
            self.ds_token = downstream_token

    def forward(self, x):
        nbatch, nmic, nf, nt, nreim = x.shape
        
        if self.pretrain:
            ## Patch (frame) spliting
            data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
            vec_patch = self.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)

            ## Mask generation
            mask_dense, mask_patch_dense, mask_ch_dense, mask_patch_idx, _ = self.patch_mask(data_shape=vec_patch.shape)  
            # mask_dense/mask_patch_dense/mask_ch_dense: (nbatch, npatch, dpatch, nmic), mask_patch_idx: (nbatch, nmasked_patch)

            ## Single-channel masking 
            npatch = vec_patch.shape[1]
            mask_dense_expand = mask_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1)
            mask_patch_dense_expand = mask_patch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1)
            mask_ch_dense_expand = mask_ch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1)
            vec_patch_mask = vec_patch * mask_dense_expand  # (nbatch, npatch, dpatch, 2, nmic)

            ## Embedding encoding
            if self.in_ver=='separate':
                ## spectral encoder
                vec_patch_mask_reshape_spec = vec_patch * (1-mask_patch_dense_expand) * mask_ch_dense_expand + vec_patch * mask_patch_dense_expand * (1-mask_ch_dense_expand) # (nbatch, npatch, d_patch, 2, nmic)
                vec_patch_mask_reshape_spec = vec_patch_mask_reshape_spec.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*2*nmic)
                embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape_spec)  # (nbatch, npatch, dembed)
                
                ## spatial encoder
                # decoupling 1 (f-wise random)
                # random_f_value = torch.rand(nbatch, vec_patch.shape[2]).to(vec_patch.device) # (nb, nf)
                # random_f_value = random_f_value[:, np.newaxis, :, np.newaxis, np.newaxis].expand_as(vec_patch)
                # vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand * random_f_value

                # decoupling 2 (tf-wise random)
                # random_tf_value = torch.rand(nbatch, vec_patch.shape[1], vec_patch.shape[2]).to(vec_patch.device) # (nb, nt, nf)
                # random_tf_value = random_tf_value[:, :, :, np.newaxis, np.newaxis].expand_as(vec_patch)
                # vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand * random_tf_value

                # decoupling 3 (local-tf-wise random)
                # ntf_each_region = [16, 256] # (nt_each_region, nf_each_region)
                # random_tf_region = torch.rand((nbatch, int(nt/ntf_each_region[0]), int(nf/ntf_each_region[1]))).to(vec_patch.device)
                # random_tf_value = torch.repeat_interleave(torch.repeat_interleave(random_tf_region, ntf_each_region[0], dim=1), ntf_each_region[1], dim=2) # (nb, nt, nf)
                # random_tf_value = random_tf_value[:, :, :, np.newaxis, np.newaxis].expand_as(vec_patch)
                # vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand * random_tf_value
                
                vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand  # (nbatch, npatch, d_patch, 2, nreim)
                vec_patch_mask_reshape_spat = vec_patch_mask_reshape_spat.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*nch)
                embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape_spat, add_same_one=False)  # (nbatch, npatch, dembed)
                
            elif self.in_ver == 'single_ch_each_patch': # each patch only has single-channel signal
                vec_patch_mask_reshape = torch.cat([vec_patch_mask[:,:,:,:,0], vec_patch_mask[:,:,:,:,1]], dim=1) # (nbatch, npatch*nmic, dpatch, nreim)
                vec_patch_mask_reshape = vec_patch_mask_reshape.reshape(nbatch, npatch*nmic, -1)  # (nbatch, npatch*nmic, dpatch*nreim)
                embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spec = torch.cat([embed_spec[:, 0:npatch, :], embed_spec[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)
                embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spat = torch.cat([embed_spat[:, 0:npatch, :], embed_spat[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)

            elif self.in_ver == 'same': # without additional spectral/spatial masking operation 
                vec_patch_mask_reshape = vec_patch_mask.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*nch)
                embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape)
                embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape, add_same_one=False)  # (nbatch, npatch, dembed)
        
            ## Embedding decoding
            embed = torch.cat([embed_spec, embed_spat], dim=2)
            vec_patch_pred = self.decoder.forward(embed)  # (nbatch, npatch, dpatch*nch+dembed) / (nbatch, npatch, 2*dembed)

            ## Loss calculation for masked part
            tar_ch_patches = torch.sum(vec_patch * (1 - mask_ch_dense_expand), dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
            tar_anotherch_patches = torch.sum(vec_patch * mask_ch_dense_expand, dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
            
            dpatch = vec_patch.shape[2]
            vec_patch_pred = vec_patch_pred.reshape(nbatch, npatch, dpatch, 2, nmic)  # (nbatch, npatch, dpatch, 2, nmic)
            pred_patches = torch.sum(vec_patch_pred * (1 - mask_ch_dense_expand), dim=-1)  # (nbatch, npatch, dpatch, 2)
       
            loss, diff = self.gen_loss(pred_patches=pred_patches, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches)

            # Additional visualized data
            mask = mask_dense.clone().detach()
            tar = vec_patch.clone().detach()
            pred = vec_patch_pred.clone().detach()
            data_vis = {}
            data_vis['mask'], data_vis['pred'], data_vis['tar'] = self.vis_results(mask_patches=mask, pred_patches=pred, tar_patches=tar)
            
            return loss, diff, data_vis

        elif self.pretrain_frozen_encoder:
            ## Patch (frame) spliting
            data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
            vec_patch = self.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)

            ## Mask generation
            mask_dense, mask_patch_dense, mask_ch_dense, mask_patch_idx, _ = self.patch_mask(data_shape=vec_patch.shape)  
            # mask_dense/mask_patch_dense/mask_ch_dense: (nbatch, npatch, dpatch, nmic), mask_patch_idx: (nbatch, nmasked_patch)

            ## Single-channel masking 
            npatch = vec_patch.shape[1]
            mask_dense_expand = mask_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1)
            mask_patch_dense_expand = mask_patch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1)
            mask_ch_dense_expand = mask_ch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1)
            vec_patch_mask = vec_patch * mask_dense_expand  # (nbatch, npatch, dpatch, 2, nmic)

            ## Embedding encoding
            if self.in_ver=='separate':
                ## spectral information
                # vec_patch_mask_reshape_spec = vec_patch * (1-mask_patch_dense_expand) * mask_ch_dense_expand + vec_patch * mask_patch_dense_expand * (1-mask_ch_dense_expand) # (nbatch, npatch, d_patch, 2, nmic)
                vec_patch_mask_reshape_spec = vec_patch * (1-mask_patch_dense_expand) * mask_ch_dense_expand # only unmasked channel
                # vec_patch_mask_reshape_spec = vec_patch * mask_patch_dense_expand * (1-mask_ch_dense_expand) # only masked channel
                vec_patch_mask_reshape_spec = vec_patch_mask_reshape_spec.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*2*nmic)
                embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape_spec)  # (nbatch, npatch, dembed)

                ## spatial information
                vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand  # (nbatch, npatch, d_patch, 2, nreim)
                vec_patch_mask_reshape_spat = vec_patch_mask_reshape_spat.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*nch)
                embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape_spat, add_same_one=False)  # (nbatch, npatch, dembed)

            ## Embedding decoding
            embed = torch.cat([embed_spec, embed_spat], dim=2)

            ## Loss calculation for masked part
            dpatch = vec_patch.shape[2]
            tar_ch_patches = torch.sum(vec_patch * (1 - mask_ch_dense_expand), dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
            tar_anotherch_patches = torch.sum(vec_patch * mask_ch_dense_expand, dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
            
            ## use spectral encoder
            # vec_patch_pred_spec = self.spec_decoder.forward(embed_spec)  # (nbatch, npatch, dpatch*nch+dembed) / (nbatch, npatch, dembed)
            # pred_patches_spec = vec_patch_pred_spec.reshape(nbatch, npatch, dpatch, 2)  # (nbatch, npatch, dpatch, 2)
            # loss = self.gen_loss_spec(pred_patches=pred_patches_spec, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches, tar_maskch=False)
            # vec_patch_pred = pred_patches_spec
 
            ## use spatial encoder
            # vec_patch_pred_spat = self.spat_decoder.forward(embed_spat)  # (nbatch, npatch, dpatch*nch+dembed) / (nbatch, npatch, dembed)
            # pred_patches_spat = vec_patch_pred_spat.reshape(nbatch, npatch, dpatch, 2)  # (nbatch, npatch, dpatch, 2)
            # loss = self.gen_loss_spec(pred_patches=pred_patches_spat, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches, tar_maskch=False)
            # vec_patch_pred =  pred_patches_spat
            
            ## use spatial & spectral encoders
            vec_patch_pred = self.spec_spat_decoder.forward(embed)  # (nbatch, npatch, dpatch*nch+dembed) / (nbatch, npatch, 2*dembed)
            vec_patch_pred = vec_patch_pred.reshape(nbatch, npatch, dpatch, 2, nmic)  # (nbatch, npatch, dpatch, 2, nmic)
            pred_patches = torch.sum(vec_patch_pred * (1 - mask_ch_dense_expand), dim=-1)  # (nbatch, npatch, dpatch, 2)
            loss = self.gen_loss_spec(pred_patches=pred_patches, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches, tar_maskch=True)
            
            ## Additional visualized data
            mask = mask_dense.clone().detach()
            tar = vec_patch.clone().detach()
            pred = vec_patch_pred.clone().detach()
            data_vis = {}
            data_vis['mask'], data_vis['pred'], data_vis['tar'] = self.vis_results(mask_patches=mask, pred_patches=pred, tar_patches=tar)
            
            return loss, loss*0.0, data_vis
        else:
            ## Patch (frame) spliting
            data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
            vec_patch = self.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)

            ## Embedding encoding
            npatch = vec_patch.shape[1]
 
            if (self.in_ver == 'separate') | (self.in_ver == 'same'):
                vec_patch_reshape = vec_patch.reshape(nbatch, npatch, -1) # (nbatch, npatch, dpatch*nch) 
                embed_spec = self.spec_encoder.forward(vec_patch_reshape)  # (nbatch, npatch, dembed)
                embed_spat = self.spat_encoder.forward(vec_patch_reshape, add_same_one=False)  # (nbatch, npatch, dembed)

            elif self.in_ver == 'single_ch_each_patch':
                vec_patch_reshape = torch.cat([vec_patch[:,:,:,:,0], vec_patch[:,:,:,:,1]], dim=1) # (nbatch, npatch*nmic, dpatch, nreim)
                vec_patch_reshape = vec_patch_reshape.reshape(nbatch, npatch*nmic, -1)  # (nbatch, npatch*nmic, dpatch*nreim)
                embed_spec = self.spec_encoder.forward(vec_patch_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spec = torch.cat([embed_spec[:, 0:npatch, :], embed_spec[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)
                embed_spat = self.spat_encoder.forward(vec_patch_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spat = torch.cat([embed_spat[:, 0:npatch, :], embed_spat[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)

            ## Downstream processing
            if self.use_cls:
                if self.ds_token == 'cls':
                    embed = torch.cat([embed_spec[:, -1:, :], embed_spat[:, -1:, :]], dim=-1)
                elif self.ds_token == 'all':
                    embed = torch.cat([embed_spec[:, :-1, :], embed_spat[:, :-1, :]], dim=-1)
            else:
                if self.embed_use4ds == 'spec_spat':
                    embed = torch.cat([embed_spec, embed_spat], dim=2)
                elif self.embed_use4ds == 'spec':
                    embed = embed_spec+0.0
                elif self.embed_use4ds == 'spat':
                    embed = embed_spat+0.0
                elif self.embed_use4ds == 'noinfo':
                    embed = torch.zeros_like(embed_spec, device=self.device).detach()

            pred = torch.mean(embed, dim=1) # (nbatch, dembed*n)
            if self.downstream_head == 'mlp':
                if self.downstream_dlabel == 1:
                    pred = self.mlp_head(pred)
                else:
                    pred = self.joint_head(pred)

            elif (self.downstream_head == 'crnn_in') | (self.downstream_head == 'crnn_med'):  
                dpatch = vec_patch.shape[2]
                embed = embed.reshape(nbatch, npatch, dpatch, -1) # (nbatch, npatch, dembed*n) → (nbatch, npatch, dpatch, dembed*n/dpatch)
                embed = self.patch_recover(embed).permute(0, 3, 1, 2) # (nbatch, n, nf, nt)
                pred = self.pred_head(x, embed)

            elif self.downstream_head == 'crnn_out': 
                pred = self.pred_head(x, pred)
 
            return pred, torch.mean(embed, dim=1)

    def gen_loss(self, pred_patches, tar_patches, mask_idx, tar_unmaskch_patches=None):
        """ Calculate the generative loss from predited masked patches and target patches 
			Args:	pred_patches - (nbatch, npatch, dpatch, 2)
					tar_patches  - (nbatch, npatch, dpatch, 2)
                    mask_idx     - (nbatch, nmasked_patch)
			Return: loss
		"""

        nmasked_patch = mask_idx.shape[-1]
        nbatch, _, dpatch, nreim = tar_patches.shape
        pred = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        tar = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        if tar_unmaskch_patches is not None:
            tar_unmaskch = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()

        for b_idx in range(nbatch):
            pred[b_idx, ...] = pred_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            tar[b_idx, ...] = tar_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            if tar_unmaskch_patches is not None:
                tar_unmaskch[b_idx, ...] = tar_unmaskch_patches[b_idx, mask_idx[b_idx], :, :]

        # Calculate the MSE loss
        loss = torch.mean((pred - tar)**2)
        if tar_unmaskch_patches is not None:
            diff = torch.mean((tar - tar_unmaskch)**2)
         
        return loss if tar_unmaskch_patches is None else loss, diff
    
    def gen_loss_spec(self, pred_patches, tar_patches, mask_idx, tar_unmaskch_patches=None, tar_maskch=True): 
        """ Calculate the generative loss from predited masked patches and target patches 
			Args:	pred_patches - (nbatch, npatch, dpatch, 2)
					tar_patches  - (nbatch, npatch, dpatch, 2)
                    mask_idx     - (nbatch, nmasked_patch)
			Return: loss
		"""
 
        nmasked_patch = mask_idx.shape[-1]
        nbatch, _, dpatch, nreim = tar_patches.shape
        pred = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        tar = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        tar_unmaskch = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()

        for b_idx in range(nbatch):
            pred[b_idx, ...] = pred_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            tar[b_idx, ...] = tar_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            tar_unmaskch[b_idx, ...] = tar_unmaskch_patches[b_idx, mask_idx[b_idx], :, :]

        # Calculate the MSE loss
        if tar_maskch:
            loss = torch.mean((pred - tar)**2)
        else:
            loss = torch.mean((pred - tar_unmaskch)**2)
         
        return loss 

    def vis_results(self, mask_patches, pred_patches, tar_patches):
        """ Visualized constructed results
			Args:	mask_patches 		- (nbatch, npatch, dpatch, nmic)
					pred_patches 		- (nbatch, npatch, dpatch, 2, nmic)
					tar_patches  		- (nbatch, npatch, dpatch, 2, nmic)
			Return: mask_patches_fold 	- (nbatch, nf, nt, 2, nmic)
					pred_patches_fold 	- (nbatch, nf, nt, 2, nmic)
					tar_patches_fold  	- (nbatch, nf, nt, 2, nmic)
		"""

        mask_patches_fold = self.patch_recover(mask_patches)
        pred_patches_fold = self.patch_recover(pred_patches)
        tar_patches_fold = self.patch_recover(tar_patches)

        return mask_patches_fold, pred_patches_fold, tar_patches_fold

if __name__ == "__main__":
    import torch
    from common.utils import get_nparams

    input = torch.randn((100, 2, 256, 256, 2))

    net = SARSSL(pretrain=True)
    ouput = net(input)
    nparam, nparam_sum = get_nparams(net, ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head'])
    print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 2)) for key in nparam.keys()])

    net = SARSSL(pretrain=False)
    nparam, nparam_sum = get_nparams(net, ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head'])
    print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 3)) for key in nparam.keys()])




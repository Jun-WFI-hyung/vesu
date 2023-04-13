import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class Unet(nn.Module):
    def __init__(self, input_ch_:int, output_ch_:int):
        super(Unet,self).__init__()
        self.input_ch = input_ch_
        self.output_ch = output_ch_
        self.contract_path_feature = []

        self.init_network()
        self.init_weights()

        print(" * Unet init complete")

    
    def init_weights(self):
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None : nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.ConvTranspose2d): 
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None : nn.init.constant_(m.bias, 0)


    def basic_cycle(self, ch_list:list, name:str):
        k_size = 3
        stride_ = 1
        padding_ = 0
        bias_ = True
        seq = nn.Sequential()

        seq.add_module(name + "_Conv1", 
                       nn.Conv2d(in_channels=ch_list[0],
                                 out_channels=ch_list[1],
                                 kernel_size=k_size,
                                 stride=stride_,
                                 padding=padding_,
                                 bias=bias_))
        seq.add_module(name + "_BN1", nn.BatchNorm2d(num_features=ch_list[1]))
        seq.add_module(name + "_ReLU1", nn.ReLU())

        seq.add_module(name + "_Conv2",
                       nn.Conv2d(in_channels=ch_list[2],
                                 out_channels=ch_list[3],
                                 kernel_size=k_size,
                                 stride=stride_,
                                 padding=padding_,
                                 bias=bias_))
        seq.add_module(name + "_BN2", nn.BatchNorm2d(num_features=ch_list[3]))
        seq.add_module(name + "_ReLU2", nn.ReLU())
         
        return seq
    

    def skip_connection(self, output:torch.Tensor):
        prev_feature = self.contract_path_feature.pop()
        transform = CenterCrop(output.shape[-1])
        concat_feat = transform(prev_feature)
        return torch.cat([output, concat_feat], dim=1)

        
    def init_network(self):
        name = "encoder"
        ch_list = [self.input_ch] + [64]*3
        self.enc_maxPool = nn.MaxPool2d(kernel_size=2)
        self.encoder1 = self.basic_cycle(ch_list=ch_list, name=name)

        ch_list = [ch_list[-1]] + [ch_list[-1]*2]*3
        self.encoder2 = self.basic_cycle(ch_list=ch_list, name=name)

        ch_list = [ch_list[-1]] + [ch_list[-1]*2]*3
        self.encoder3 = self.basic_cycle(ch_list=ch_list, name=name)

        ch_list = [ch_list[-1]] + [ch_list[-1]*2]*3
        self.encoder4 = self.basic_cycle(ch_list=ch_list, name=name)

        name = "bottleneck"
        ch_list = [ch_list[-1]] + [ch_list[-1]*2]*2 + [ch_list[-1]]
        self.bottleneck = self.basic_cycle(ch_list=ch_list, name=name)
        self.neck_samp = nn.ConvTranspose2d(in_channels=ch_list[3],
                                            out_channels=ch_list[3],
                                            kernel_size=2, stride=2)

        name = "decoder"
        ch_list = [ch_list[-1]*2] + [ch_list[-1]]*2 + [ch_list[-1]//2]
        self.decoder1 = self.basic_cycle(ch_list=ch_list, name=name)
        self.dec1_samp = nn.ConvTranspose2d(in_channels=ch_list[3],
                                            out_channels=ch_list[3],
                                            kernel_size=2, stride=2)

        ch_list = [ch_list[-1]*2] + [ch_list[-1]]*2 + [ch_list[-1]//2]
        self.decoder2 = self.basic_cycle(ch_list=ch_list, name=name)
        self.dec2_samp = nn.ConvTranspose2d(in_channels=ch_list[3],
                                            out_channels=ch_list[3],
                                            kernel_size=2, stride=2)

        ch_list = [ch_list[-1]*2] + [ch_list[-1]]*2 + [ch_list[-1]//2]
        self.decoder3 = self.basic_cycle(ch_list=ch_list, name=name)
        self.dec3_samp = nn.ConvTranspose2d(in_channels=ch_list[3],
                                            out_channels=ch_list[3],
                                            kernel_size=2, stride=2)

        ch_list = [ch_list[-1]*2] + [ch_list[-1]]*3
        self.decoder4 = self.basic_cycle(ch_list=ch_list, name=name)
        self.dec4_samp = nn.Conv2d(in_channels=ch_list[3],
                                   out_channels=self.output_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True)

    
    def forward(self, mat_:torch.tensor):
        output = self.encoder1(mat_)
        self.contract_path_feature.append(output)
        output = self.enc_maxPool(output)

        output = self.encoder2(output)
        self.contract_path_feature.append(output)
        output = self.enc_maxPool(output)

        output = self.encoder3(output)
        self.contract_path_feature.append(output)
        output = self.enc_maxPool(output)

        output = self.encoder4(output)
        self.contract_path_feature.append(output)
        output = self.enc_maxPool(output)

        output = self.bottleneck(output)
        output = self.neck_samp(output)

        output = self.skip_connection(output)
        output = self.decoder1(output)
        output = self.dec1_samp(output)

        output = self.skip_connection(output)
        output = self.decoder2(output)
        output = self.dec2_samp(output)

        output = self.skip_connection(output)
        output = self.decoder3(output)
        output = self.dec3_samp(output)

        output = self.skip_connection(output)
        output = self.decoder4(output)
        output = self.dec4_samp(output)

        return output

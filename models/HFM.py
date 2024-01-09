

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)


    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x
        return res


class HFM(nn.Module):
    def __init__(self, channle,conv=default_conv):
        super(HFM, self).__init__()
        self.bs1 = Block(conv,channle,3)
        self.bs2 = Block(conv,channle,3)
        self.bs3 = Block(conv,channle,3)
        self.bs4 = Block(conv,channle,3)
        self.bs5 = Block(conv,channle,3)
        self.bs6 = Block(conv,channle,3)

        self.down_1 = nn.Conv2d(channle, channle, kernel_size=3, stride=2, padding=1)
        self.down_2 = nn.Conv2d(channle, channle, kernel_size=3, stride=4, padding=1)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x):
        y1 = self.upsample_1(self.bs1(self.down_2(x)))
        y2 = self.upsample_2(torch.add(self.bs3(self.bs2(self.down_1(x))),y1))
        y3 = torch.add(self.bs6(self.bs5(self.bs4(x))),y2)
        # y1 = self.bs6(self.bs5(self.bs4(x)))
        # y2 = self.bs3(self.bs2(x))
        # y3 = self.bs1(x)

        return y3






if __name__ == "__main__":
    N, C_in, H, W = 1, 64, 16, 16
    x = torch.randn(N, C_in, H, W).float()
    # x2 = torch.randn(N, C_in, H // 4, W // 4).float()
    # y = Fusion(3,3)
    # y = CWA(32)
    y = HFM(64)
    print(y)
    # _,_,result = y(x2,x1,x)
    result = y(x)
    # print(result)
    print(result.shape)
    print("groups=in_channels时参数大小：%d" % sum(param.numel() for param in y.parameters()))

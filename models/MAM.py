

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F



class CWA(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CWA, self).__init__()
        # global average pooling: feature --> point
        self.conv_1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv_2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel , 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.relu(self.conv_1(x))
        y1 = self.relu(self.conv_1(y))
        y2 = self.avg_pool(y1)
        y3 = self.conv_du(y2)
        return x * y3


class PWA(nn.Module):
    def __init__(self, channel):
        super(PWA, self).__init__()
        self.conv_1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv_2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv_3 = nn.Conv2d(channel, channel, 1)
        self.conv_4 = nn.Conv2d(channel, channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.relu(self.conv_1(x))
        y1 = self.relu(self.conv_2(y))
        y2 = self.relu(self.conv_3(y1))
        y3 = self.sigmoid(self.conv_4(y2))
        return x*y3



class MAM(nn.Module):
    def __init__(self, channel):
        super(MAM, self).__init__()
        self.cwa = CWA(channel)
        self.pwa = PWA(channel)

    def forward(self, x):
        cwa_value = self.cwa(x)
        pwa_value = self.pwa(cwa_value)
        mam_value = torch.add(pwa_value, x)
        return mam_value


if __name__ == "__main__":
    N, C_in, H, W = 1, 32, 16, 16
    x = torch.randn(N, C_in, H, W).float()
    # x2 = torch.randn(N, C_in, H // 4, W // 4).float()
    # y = Fusion(3,3)
    # y = CWA(32)
    y = MAM(32)
    print(y)
    # _,_,result = y(x2,x1,x)
    result = y(x)
    # print(result)
    print(result.shape)
    print("groups=in_channels时参数大小：%d" % sum(param.numel() for param in y.parameters()))

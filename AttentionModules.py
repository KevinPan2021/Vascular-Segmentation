import torch
import torch.nn as nn

class LA_RBA_UNet(nn.Module):
    def __init__(self, in_channels,channels, n_classes, large=False):
        super(LA_RBA_UNet, self).__init__()
        self.in_channels = in_channels
        self.channels=channels
        self.n_classes = n_classes
        
        if large: 
            self.up1 = Up((16+8) * channels, 8 * channels)
            self.up2 = Up((8+4) * channels, 4 * channels)
            self.up3 = Up((4+2) * channels, 2 * channels)
            self.up4 = Up((2+1) * channels, channels)
            self.outc = OutConv2d(channels, n_classes)
            
            self.re1 = ReverseEdge(channels * 2)
            self.re2 = ReverseEdge(channels * 4)
            self.re3 = ReverseEdge(channels * 8)
            self.re4 = ReverseEdge(channels * 16)
            self.down = nn.MaxPool2d(2)
            self.dConv1 = DoubleConv(channels,channels)
            self.dConv2 = DoubleConv(channels, channels * 2)
            self.dConv3 = DoubleConv(channels * 2, channels * 4)
            self.dConv4 = DoubleConv(channels * 4, channels * 8)
            self.dConv5 = DoubleConv(channels * 8, channels * 16)
        else:
            self.up1 = Up(channels * 2)
            self.up2 = Up(channels * 2)
            self.up3 = Up(channels * 2)
            self.up4 = Up(channels * 2)
            self.outc = OutConv2d(channels, n_classes)
            
            self.re1 = ReverseEdge(channels)
            self.re2 = ReverseEdge(channels)
            self.re3 = ReverseEdge(channels)
            self.re4 = ReverseEdge(channels)
            self.down = nn.MaxPool2d(2)
            self.dConv1 = DoubleConv(channels,channels)
            self.dConv2 = DoubleConv(channels,channels)
            self.dConv3 = DoubleConv(channels,channels)
            self.dConv4 = DoubleConv(channels,channels)
            self.dConv5 = DoubleConv(channels,channels)
                
            
    def forward(self, x, film):
        enc1 = self.dConv1(x, film)
        down1 = self.down(enc1)
        enc2 = self.dConv2(down1, film)
        x1 = self.re1(enc2, enc1)
        down2 = self.down(enc2)           
        enc3 = self.dConv3(down2, film)
        x2 = self.re2(enc3, enc2)
        down3 = self.down(enc3)
        enc4 = self.dConv4(down3, film)
        x3 = self.re3(enc4, enc3)
        down4 = self.down(enc4)
        enc5 = self.dConv5(down4, film)
        x4 = self.re4(enc5, enc4)
        x = self.up1(enc5, x4, film)

        x = self.up2(x, x3, film)
        x = self.up3(x, x2, film)
        feature = self.up4(x, x1, film)
        x = self.outc(feature)
        return x
        
class ReverseEdge(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1x1 = nn.Conv2d(in_channels, 1, 1)
        self.up = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(1, 1, 1)
    def forward(self, x, x2):
        x = self.up(self.conv1x1_2(self.relu(self.conv1x1(x))))
        x = -1 * (torch.sigmoid(x)) + 1
        if x.shape[2] != x2.shape[2]:
            x = self.up2(x)
        x = x.expand(-1, x2.shape[1], -1, -1).mul(x2)
        x = x + x2
        return x

class OutConv2d(nn.Module):
    def __init__(self, channels, n_class):
        super(OutConv2d, self).__init__()
        self.conv = nn.Conv2d(channels, n_class, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels = None, bilinear=True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels // 2
            
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  
            raise NotImplementedError("ConvTranspose2D interpolation is not implemented")
        
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2, film):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, film)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=32,num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32,num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.film = nn.Linear(64, out_channels*2)
        
    def forward(self, x, film):
        residual = self.conv1x1(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        gamma, beta = torch.chunk(self.film(film), 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        out = gamma * out + beta
        
        out = self.relu(out)
        out = out + residual
        return out
    


class MutiScaleLayerAttention(nn.Module):
    def __init__(self, channels, depth_size, num_groups):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 1)
        self.convR0 = nn.Conv3d(channels, channels, 1)
        self.convR1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.convR2 = nn.Conv3d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.convR3 = nn.Conv3d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.planePooling = nn.AdaptiveAvgPool3d((depth_size, 1, 1))
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.gn = nn.GroupNorm(num_groups=num_groups,num_channels=channels) 
        
        self.film = nn.Linear(64, channels*2)
    def forward(self, x, film):
        x1 = self.relu(self.gn(self.convR0(x)))
        x2 = self.relu(self.gn(self.convR1(x)))
        x3 = self.relu(self.gn(self.convR2(x)))
        x4 = self.relu(self.gn(self.convR3(x)))
        
        x = torch.add(x1,x2)
        x = torch.add(x,x3)
        x = torch.add(x,x4)
        x = self.relu(self.gn(self.conv1(x)))
        

        
        x0 = self.gn(self.conv2(self.planePooling(x)))
        gamma, beta = torch.chunk(self.film(film), 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x0 = gamma * x0 + beta
        x0 = torch.sigmoid(self.relu(x0))
        
        x = torch.add(x,torch.mul(x, x0))
        return x

class InConv3d(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(num_groups=channels//2,num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        self.film = nn.Linear(64, channels*2)
    
    def forward(self, x, film):
        gamma, beta = torch.chunk(self.film(film), 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.gn(self.conv(x))
        x = gamma * x + beta
        x = self.relu(x)
        return x
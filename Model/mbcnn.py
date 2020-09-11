import torch
import torch.nn as nn
import torch.nn.functional as F

class single_conv(nn.Module):
    '''(conv => BN => ReLU) '''
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        # Define the layers here
        # Note: for conv, use a padding of (1,1) so that size is maintained
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        # define forward operation using the layers we have defined
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class single_conv2(nn.Module):
    '''(conv => BN => ReLU) '''
    def __init__(self, in_ch, out_ch):
        super(single_conv2, self).__init__()
        # Define the layers here
        # Note: for conv, use a padding of (1,1) so that size is maintained
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        # define forward operation using the layers we have defined
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        #self.up = nn.Upsample(scale_factor=2)  # use nn.Upsample() with mode bilinear
        self.up=nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2,padding=0)
    def forward(self, x1, x2):  # Takes in smaller x1 and larger x2
        # First we upsample x1 to be same size as x2
        x1 = self.up(x1)
        # This part is tricky so we've completed this
        # Notice that x2 and x1 may not have the same spatial size.
        # This is because when you downsample old_x2(say 25 by 25), you will get x1(12 by 12)
        # Then you perform upsample to x1, you will get new_x1(24 by 24)
        # You should pad a new row and column so that new_x1 and x2 have the same size.

        # diffD = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # diffY = x2.size()[4] - x1.size()[4]
        # x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2,
        #                 diffX // 2, diffX - diffX // 2,
        #                 diffD // 2, diffD - diffD // 2))

        # Now we concatenat x2 and x1 along channel dimension: torch.cat()
        # Note pytorch tensor shape correspond to: (batchsize, channel, x_dim, y_dim)
        x = torch.cat([x1, x2], 1)

        return x
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # 1 conv layer
        # self.conv = nn.Conv3d(
        #     in_channels=in_ch,
        #     out_channels=out_ch,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        # )
        #use 1*1 convolution
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )
    def forward(self, x):
        x = self.conv(x)
        # Apply sigmoid activation: torch.sigmoid()
        x = torch.sigmoid(x)
        return x
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.down = nn.MaxPool3d(2)  # use nn.MaxPool2d( )

    def forward(self, x):
        x = self.down(x)
        return x
class Mul(nn.Module):
    def __init__(self, in_channel, out_channel):
        self.in_channel = in_channel
        self.out_channel = out_channel
        super(Mul, self).__init__()
        self.conv1_T1=single_conv(self.in_channel,16)
        self.conv2_T1 =single_conv(16,32)
        self.conv3_T1 =single_conv(32,64)
        self.conv4_T1 =single_conv(64,64)

        self.conv1_T2 = single_conv(self.in_channel, 16)
        self.conv2_T2 = single_conv(16, 32)
        self.conv3_T2 = single_conv(32, 64)
        self.conv4_T2 = single_conv(64, 64)

        self.conv1_F = single_conv(self.in_channel, 16)
        self.conv2_F = single_conv(16, 32)
        self.conv3_F = single_conv(32, 64)
        self.conv4_F = single_conv(64, 64)

        self.convf1 = single_conv2(3 * 16, 16)
        self.convf2 = single_conv2(3 * 32, 32)
        self.convf3 = single_conv2(3 * 64, 64)
        self.convf4 = single_conv2(3 * 64, 64)


        self.conv5 =single_conv(128,32)
        self.conv6 =single_conv(64,16)
        self.conv7 =single_conv(32,16)
        self.conv8 =outconv(16,self.out_channel)

        self.down1 = down(16, 16)
        self.down2 = down(32, 32)
        self.down3 = down(64, 64)

        self.up1 = up(64, 128)
        self.up2 = up(32, 64)
        self.up3 = up(16, 32)
    def forward(self, x,y,z):
        x1=self.conv1_T1(x)
        x=self.down1(x1)
        x2=self.conv2_T1(x)
        x=self.down2(x2)
        x3=self.conv3_T1(x)
        x=self.down3(x3)
        x4=self.conv4_T1(x)

        y1 = self.conv1_T2(y)
        y = self.down1(y1)
        y2 = self.conv2_T2(y)
        y = self.down2(y2)
        y3 = self.conv3_T2(y)
        y = self.down3(y3)
        y4 = self.conv4_T2(y)

        z1 = self.conv1_F(z)
        z = self.down1(z1)
        z2 = self.conv2_F(z)
        z = self.down2(z2)
        z3 = self.conv3_F(z)
        z = self.down3(z3)
        z4 = self.conv4_F(z)


        new4=self.convf4(torch.cat((x4,y4,z4),1))#3*64->64
        new3 = self.convf3(torch.cat((x3, y3, z3), 1))  # 3*64->64
        new2 = self.convf2(torch.cat((x2, y2, z2), 1))  # 3*32->32
        new1 = self.convf1(torch.cat((x1, y1, z1), 1))  # 3*16->16



        x=self.up1(new4,new3)
        x=self.conv5(x)
        x=self.up2(x,new2)
        x=self.conv6(x)
        x=self.up3(x,new1)
        x=self.conv7(x)
        x=self.conv8(x)
        return x
if __name__=="__main__":
    net=Mul(1,1)
    # This shows the number of parameters in the network
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters in network: ', n_params)

    a=torch.zeros((1,1,64,64,64))
    net(a,a,a)
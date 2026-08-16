import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_chann, out_chann,mid_chann=None):
        super().__init__()

        if mid_chann==None:
            mid_chann = out_chann

        self.doubleConv= nn.Sequential(
            nn.Conv2d(in_channels=in_chann, out_channels=mid_chann,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(mid_chann),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_chann, out_channels=out_chann,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_chann),
            nn.ReLU(inplace=True)
            
        )


    def forward(self,x):
        return self.doubleConv(x)
    

class Down(nn.Module):
    def __init__(self,  in_chann, out_chann,mid_chann=None):
        super().__init__()

        self.doubleConv= DoubleConv(in_chann=in_chann,out_chann=out_chann,mid_chann=mid_chann)
        self.maxPool= nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x= self.doubleConv(x)
        return self.maxPool(x)
    

class AttentionBlock(nn.Module):

    def __init__(self, skip_chann, gating_chann, inter_chann):
        super().__init__()


        self.skipConv= nn.Sequential(
            nn.Conv2d(in_channels=skip_chann,out_channels=inter_chann,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(inter_chann)
        )

        self.gatingConv= nn.Sequential(
            nn.Conv2d(in_channels=gating_chann,out_channels=inter_chann,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(inter_chann)
        )

        self.psi= nn.Sequential(
            nn.Conv2d(in_channels=inter_chann,out_channels=1,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu=nn.ReLU(inplace=True)


    def forward(self,x_skip,g):
        x_skip1=self.skipConv(x_skip)
        g1=self.gatingConv(g)

        attention=self.relu(x_skip1+g1)
        attention=self.psi(attention)

        return x_skip*attention
    

class Up(nn.Module):
    def __init__(self, in_chann, out_chann):
        super().__init__()

        self.deConv= nn.ConvTranspose2d(in_channels=in_chann,out_channels=in_chann//2,kernel_size=2,stride=2)

        self.attention=AttentionBlock(skip_chann=in_chann//2,gating_chann=in_chann//2,inter_chann=in_chann//4)

        self.doubleConv= DoubleConv(in_chann=in_chann,out_chann=out_chann)


    def forward(self,x,x_skip):
        x=self.deConv(x)
        x_skip=self.attention(x_skip,x)
        x=torch.cat((x,x_skip),dim=1)
        return self.doubleConv(x)
    



class AttentionUnet(nn.Module):
    def __init__(self, in_chann, num_class):
        super().__init__()

        self.inc= DoubleConv(in_chann,64)
        self.down1=Down(64,128)
        self.down2=Down(128,256)
        self.down3=Down(256,512)
        self.down4=Down(512,1024)


        self.up1= Up(1024,512)
        self.up2= Up(512,256)
        self.up3= Up(256,128)
        self.up4= Up(128,64)


        self.conv = nn.Conv2d(64,num_class,kernel_size=1)


    def forward(self,x):
        inc = self.inc(x)
        d1= self.down1(inc)
        d2= self.down2(d1)
        d3= self.down3(d2)
        d4= self.down4(d3)

        x = self.up1(d4,d3)
        x = self.up2(x,d2)
        x= self.up3(x,d1)
        x= self.up4(x,inc)

        x=self.conv(x)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

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



class Up(nn.Module):
    def __init__(self, in_chann, out_chann,skip_count):

        super().__init__()

        self.deConv= nn.ConvTranspose2d(in_channels=in_chann,out_channels=out_chann,kernel_size=2,stride=2)

        self.doubleConv= DoubleConv(in_chann=out_chann*(skip_count+1),out_chann=out_chann)


    def forward(self,x,*x_skips):
        x=self.deConv(x)
        x=torch.cat((*x_skips,x),dim=1)
        return self.doubleConv(x)



class UnetPlusPlus(nn.Module):
    def __init__(self, in_chann, num_class):
        super().__init__()

        self.inc= DoubleConv(in_chann,64)
        self.down1=Down(64,128)
        self.down2=Down(128,256)
        self.down3=Down(256,512)
        self.down4=Down(512,1024)

        self.up1_1= Up(128,64,1)
        self.up2_1= Up(256,128,1)
        self.up3_1= Up(512,256,1)
        self.up4_1= Up(1024,512,1)

        self.up1_2= Up(128,64,2)
        self.up2_2= Up(256,128,2)
        self.up3_2= Up(512,256,2)

        self.up1_3= Up(128,64,3)
        self.up2_3= Up(256,128,3)

        self.up1_4= Up(128,64,4)


        self.conv = nn.Conv2d(64,num_class,kernel_size=1)


    def forward(self,x):
        x0_0 = self.inc(x)
        x1_0= self.down1(x0_0)
        x2_0= self.down2(x1_0)
        x3_0= self.down3(x2_0)
        x4_0= self.down4(x3_0)

        x0_1 = self.up1_1(x1_0,x0_0)
        x1_1 = self.up2_1(x2_0,x1_0)
        x2_1 = self.up3_1(x3_0,x2_0)
        x3_1 = self.up4_1(x4_0,x3_0)

        x0_2 = self.up1_2(x1_1,x0_0,x0_1)
        x1_2 = self.up2_2(x2_1,x1_0,x1_1)
        x2_2 = self.up3_2(x3_1,x2_0,x2_1)

        x0_3 = self.up1_3(x1_2,x0_0,x0_1,x0_2)
        x1_3 = self.up2_3(x2_2,x1_0,x1_1,x1_2)

        x0_4 = self.up1_4(x1_3,x0_0,x0_1,x0_2,x0_3)

        x=self.conv(x0_4)

        return x

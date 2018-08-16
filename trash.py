data_transform = transforms.Compose([
        #transforms.Resize((64,64)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop((64,64)),
        #transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(),
        transforms.ToTensor(),
        #lambda x: randomInvert(x)
    ])
train_data = datasets.ImageFolder(root='data/train',
                                           transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_data, 
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=12, drop_last=True)
#test_data = datasets.ImageFolder(root='data/test',
#                                           transform=data_transform)
#test_loader = torch.utils.data.DataLoader(test_data, 
#                                             batch_size=batch_size, shuffle=True,
#                                             num_workers=12, drop_last=True)
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(0, self.stddev).cuda()
            return x + noise        
        else:
            return x

def randomInvert(x):
    if not not getrandbits(1):
        return x
    else:
        return 1-x

def addNoise(x):
    t = torch.Tensor(x.size())
    t.uniform_(-noise,noise)
    return x + t

def normTensor(x):
    return (x-torch.mean(x))/torch.std(x)

data_transform = transforms.Compose([
        #transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        #lambda x: randomInvert(x)
    ])
train_data = datasets.ImageFolder(root='data/train',
                                           transform=data_transform)

train_loader = torch.utils.data.DataLoader(train_data, 
                                             batch_size=batch, shuffle=True,
                                             num_workers=12, drop_last=True)

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = Variable(x.data.new(x.size()).normal_(0, self.stddev))
            return x + noise        
        else:
            return x

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.PReLU() #inplace=True
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class same(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(same, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        self.sqx = SqEx(out_channels)
        #self.noise = GaussianNoise(0.01)
        self.residual = False
        if in_channels == out_channels:
            self.residual = True

    def forward(self,x):
        a = self.conv(x)
        #a = self.noise(a)
        a = self.relu(a)
        a = self.lrn(a)
        a = self.sqx(a)
        if self.residual:
            a = (a + x)*0.5
        return a

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.PReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        self.sqx = SqEx(out_channels)

    def forward(self,x):
        a = self.conv(x)
        a = self.relu(a)
        a = self.lrn(a)
        a = self.sqx(a)
        return a

class one(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(one, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.PReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        self.sqx = SqEx(out_channels)

    def forward(self,x):
        a = self.conv(x)
        a = self.relu(a)
        a = self.lrn(a)
        a = self.sqx(a)
        return a

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.conv = nn.Conv2d(
        #    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sqx = SqEx(out_channels)

    def forward(self,x):
        #a = self.upsample(x)
        a = self.conv(x)
        a = self.relu(a)
        a = self.lrn(a)
        a = self.sqx(a)
        return a


class UNet(nn.Module):
    def __init__(self, inC, outC):
        super(UNet, self).__init__()

        """self.relu = nn.ReLU()
        self.lrn1 = nn.BatchNorm2d(32)
        self.prelu = nn.PReLU()
        self.tanh = nn.Hardtanh()
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')"""
        
        #enco
        self.same1 = same(inC,32)

        self.down1 = down(32,64)
        self.same2 = same(64,64)
        self.same3 = same(64,64)

        self.down2 = down(64,128)
        self.same4 = same(128,128)
        self.same5 = same(128,128)

        self.down3 = down(128,256)
        self.same6 = same(256,256)
        self.same7 = same(256,256)
        self.same8 = same(256,256)

        #Z
        self.down4 = down(256,512)
        self.same9 = same(512,512)
        self.same10 = same(512,512)
        self.same11 = same(512,512)

        #deco
        self.up4 = up(512,256)
        self.same12 = same(512,512)
        self.same13 = same(512,512)
        self.same14 = same(512,256)

        self.up3 = up(256,128)
        self.same15 = same(256,256)
        self.same16 = same(256,128)

        self.up2 = up(128,64)
        self.same17 = same(128,128)
        self.same18 = same(128,64)

        self.up1 = up(64,32)
        self.same19 = same(64,64)
        self.same20 = same(64,32)

        self.finalone = self.conv = nn.Conv2d(
            in_channels=32, out_channels=outC, kernel_size=1, stride=1, padding=0)
        
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #enco
        h1 = self.same1(x)
        
        out = self.down1(h1)
        out = self.same2(out)
        h2 = self.same3(out)

        out = self.down2(h2)
        out = self.same4(out)
        h3 = self.same5(out)

        out = self.down3(h3)
        out = self.same6(out)
        out = self.same7(out)
        h4 = self.same8(out)

        #Z
        out = self.down4(h4)
        out = self.same9(out)
        out = self.same10(out)
        h5 = self.same11(out)

        #deco
        out = self.up4(h5)
        out = torch.cat((h4,out),1)
        out = self.same12(out)
        out = self.same13(out)
        out = self.same14(out)

        out = self.up3(out)
        out = torch.cat((h3,out),1)
        out = self.same15(out)
        out = self.same16(out)

        out = self.up2(out)
        out = torch.cat((h2,out),1)
        out = self.same17(out)
        out = self.same18(out)

        out = self.up1(out)
        out = torch.cat((h1,out),1)
        out = self.same19(out)
        out = self.same20(out)

        out = self.finalone(out)

        #segmento
        #out = out.view(batch,2,3,64,64)
        #out = self.softmax(out)
        #out = out.view(batch,6,64,64)[:,0:3,:,:]
        return out


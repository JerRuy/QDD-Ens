
from networks.cenet import *





class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(avg_out)


class br(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, ch):
        super(br, self).__init__()
        self.attention = ChannelAttention(ch)
        # self.ch =
        self.br = nn.Sequential(
            nn.Conv2d( 3 *ch, ch, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.br(x)
        cab = self.attention(x)
        x = x * cab
        return x


class brr(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, ch):
        super(brr, self).__init__()
        self.br = nn.Sequential(
            nn.Conv2d( 2 *ch, ch, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.br(x)
        return x





class QME_Net_cenet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(QME_Net_cenet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


        self.cenet1 = CE_Net_().cuda()
        self.cenet2 = CE_Net_().cuda()
        self.cenet3 = CE_Net_().cuda()

        self.down0br = br(filters[0])
        self.down1br = br(filters[0])
        self.down2br = br(filters[1])
        self.down3br = br(filters[2])
        self.down4br = br(filters[3])

        self.centerbr = br(516)

        self.up1br = br(filters[0])
        self.up2br = br(filters[0])
        self.up3br = br(filters[1])
        self.up4br = br(filters[2])



        self.down0brr = brr(filters[0])
        self.down1brr = brr(filters[0])
        self.down2brr = brr(filters[1])
        self.down3brr = brr(filters[2])
        self.down4brr = brr(filters[3])

        self.centerbrr = brr(516)

        self.up1brr = brr(filters[0])
        self.up2brr = brr(filters[0])
        self.up3brr = brr(filters[1])
        self.up4brr = brr(filters[2])

    def forward(self, x):


        n1x ,n1e1 ,n1e2 ,n1e3 ,n1e4 ,n1e5, n1d4 ,n1d3 ,n1d2 ,n1d1 ,n1final = self.cenet1(x)
        n2x ,n2e1 ,n2e2 ,n2e3 ,n2e4 ,n2e5, n2d4 ,n2d3 ,n2d2 ,n2d1 ,n2final = self.cenet2(x)
        n3x ,n3e1 ,n3e2 ,n3e3 ,n3e4 ,n3e5, n3d4 ,n3d3 ,n3d2 ,n3d1 ,n3final = self.cenet3(x)


        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)


        n0x = torch.cat([n1x, n2x, n3x] ,dim=1)
        n0x = self.down0br(n0x)
        x = torch.cat([n0x, x] ,dim=1)
        x = self.down0brr(x)

        e1 = self.encoder1(x)
        n0e1 = torch.cat([n1e1, n2e1, n3e1] ,dim=1)
        n0e1 = self.down1br(n0e1)
        e1 = torch.cat([n0e1, e1] ,dim=1)
        e1 = self.down1brr(e1)

        e2 = self.encoder2(e1)
        n0e2 = torch.cat([n1e2, n2e2, n3e2] ,dim=1)
        n0e2 = self.down2br(n0e2)
        e2 = torch.cat([n0e2, e2] ,dim=1)
        e2 = self.down2brr(e2)

        e3 = self.encoder3(e2)
        n0e3 = torch.cat([n1e3, n2e3, n3e3] ,dim=1)
        n0e3 = self.down3br(n0e3)
        e3 = torch.cat([n0e3, e3] ,dim=1)
        e3 = self.down3brr(e3)

        e4 = self.encoder4(e3)
        n0e4 = torch.cat([n1e4, n2e4, n3e4] ,dim=1)
        n0e4 = self.down4br(n0e4)
        e4 = torch.cat([n0e4, e4] ,dim=1)
        e4 = self.down4brr(e4)

        # Center
        e5 = self.dblock(e4)
        e5 = self.spp(e5)
        n0e5 = torch.cat([n1e5, n2e5, n3e5] ,dim=1)
        n0e5 = self.centerbr(n0e5)
        e5 = torch.cat([n0e5, e5] ,dim=1)
        e5 = self.centerbrr(e5)

        # Decoder
        d4 = self.decoder4(e5) + e3
        n0d4 = torch.cat([n1d4, n2d4, n3d4] ,dim=1)
        n0d4 = self.up4br(n0d4)
        d4 = torch.cat([n0d4, d4] ,dim=1)
        d4 = self.up4brr(d4)

        d3 = self.decoder3(d4) + e2
        n0d3 = torch.cat([n1d3, n2d3, n3d3] ,dim=1)
        n0d3 = self.up3br(n0d3)
        d3 = torch.cat([n0d3, d3] ,dim=1)
        d3 = self.up3brr(d3)

        d2 = self.decoder2(d3) + e1
        n0d2 = torch.cat([n1d2, n2d2, n3d2] ,dim=1)
        n0d2 = self.up2br(n0d2)
        d2 = torch.cat([n0d2, d2] ,dim=1)
        d2 = self.up2brr(d2)

        d1 = self.decoder1(d2)
        n0d1 = torch.cat([n1d1, n2d1, n3d1] ,dim=1)
        n0d1 = self.up1br(n0d1)
        d1 = torch.cat([n0d1, d1] ,dim=1)
        d1 = self.up1brr(d1)


        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        final = torch.sigmoid(out)
        return n1final, n2final, n3final, final \
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4 \







from torch import nn
import torch
import math
from torch.nn import functional as F
from model.base_function import init_net
import numpy as np

def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(patchsizes=[1,2,4,8])
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)


class Generator(nn.Module):
    def __init__(self, patchsizes, ngf=64, max_ngf=256):
        super().__init__()
        self.down = Encoder(ngf)
        self.up = Decoder(ngf)
        for i, pz in enumerate(patchsizes):
            length = 64 // pz
            dis = lap(length)
            dis = dis.view(length * length, length, length).float()
            block = TransformerEncoder(patchsizes=pz, num_hidden=max_ngf, dis=dis)
            setattr(self, 'transE'+str(i+1), block)
            block = TransformerDecoder(patchsizes=pz, num_hidden=max_ngf, dis=dis)
            setattr(self, 'transD'+str(i+1), block)

    def forward(self, x, mask):
        feature = torch.cat([x, mask], dim=1)
        feature = self.down(feature)
        feature = self.transE1(feature)
        feature = self.transE2(feature)
        feature = self.transE3(feature)
        feature = self.transE4(feature)

        feature = self.transD4(feature, feature, feature)
        feature = self.transD3(feature, feature, feature)
        feature = self.transD2(feature, feature, feature)
        feature = self.transD1(feature, feature, feature)

        out = self.up(feature)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]



class Encoder(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.encoder1 = ResBlock0(in_ch=4, out_ch=ngf, kernel_size=5, stride=1, padding=2)
        self.encoder2 = ResBlock(in_ch=ngf, out_ch=ngf*2, kernel_size=3, stride=2, padding=1)

        self.encoder22 = ResBlock(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.encoder32 = ResBlock(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1)
        self.encoder3 = ResBlock(in_ch=ngf*2, out_ch=ngf*4, kernel_size=3, stride=2, padding=1)



    def forward(self, img_m):
        x = self.encoder1(img_m)
        x = self.encoder2(x)
        x = self.encoder22(x)
        x = self.encoder3(x)
        x = self.encoder32(x)
        return x


class Decoder(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.decoder1 = ResBlock(in_ch=ngf*4, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.decoder12 =ResBlock(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.decoder2 = ResBlock(in_ch=ngf*2, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.decoder22 = ResBlock(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.decoder1(x)
        x = self.decoder12(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.decoder2(x)
        x = self.decoder22(x)
        x = self.decoder3(x)
        return x


class ResBlock0(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)


    def forward(self, x):
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + residual

        return out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act0 = nn.SiLU(inplace=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n0 = nn.InstanceNorm2d(in_ch, track_running_stats=False)

    def forward(self, x):
        residual = self.projection(x)
        out = self.n0(x)
        out = self.act0(out)
        out = self.conv1(out)
        out = self.n1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + residual

        return out

# (H * W) * C -> (H/2 * C/2) * (4C) -> (H/4 * W/4) * 16C -> (H/8 * W/8) * 64C
class TransformerEncoder(nn.Module):
    def __init__(self, patchsizes, num_hidden=256, dis=None):
        super().__init__()
        self.attn = MultiPatchMultiAttention(patchsizes, num_hidden, dis)
        self.feed_forward = FeedForward(num_hidden)

    def forward(self, x, mask=None):
        x = self.attn(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, patchsizes, num_hidden=256, dis=None):
        super().__init__()
        self.cross_attn = MultiPatchMultiAttention(patchsizes, num_hidden, dis=dis)
        self.feed_forward = FeedForward(num_hidden)

    def forward(self, query, key, value):
        x = self.cross_attn(query, key, value)
        x = self.feed_forward(x)
        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    q, k, v  = B * N (h*w) * C
    """

    def forward(self, query, key, value, mask=None, dis=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        if dis is not None:
            scores = scores + dis

        scores = scores / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            #scores = scores * mask
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiAttn(nn.Module):
    """
    Attention Network
    """

    def __init__(self, head=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        key, value, query B*C*H*W
        """
        super().__init__()
        self.h = head

        self.attn = Attention()

    def forward(self, query, key, value, mask=None, dis=None):

        B,N,C = key.size()
        num_hidden_per_attn = C // self.h
        k = key.view(B, N, self.h, num_hidden_per_attn).contiguous()
        v = value.view(B, N, self.h, num_hidden_per_attn).contiguous()
        q = query.view(B, N, self.h, num_hidden_per_attn).contiguous()

        k = k.permute(2,0,1,3).contiguous() # view(-1, N, num_hidden_per_attn)
        v = v.permute(2,0,1,3).contiguous()
        q = q.permute(2,0,1,3).contiguous()

        if mask is not None:
            mask = mask.unsqueeze(0)
            out, attn = self.attn(q, k, v, mask, dis)
        else:
            out, attn = self.attn(q, k, v, dis=dis)
        out = out.view(self.h, B, N, num_hidden_per_attn)
        out = out.permute(1, 2, 0, 3).contiguous().view(B, N, C).contiguous()
        return out, attn


class FeedForward(nn.Module):
    def __init__(self, num_hidden):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(num_hidden, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_hidden, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = x + self.conv(x)
        return x


class MultiPatchMultiAttention(nn.Module):
    def __init__(self, patchsize, num_hidden, dis):
        super().__init__()
        self.ps = patchsize
        num_head = patchsize * 4
        self.query_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)

        self.output_linear = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))


        self.attention = MultiAttn(head=num_head)
        length = 64 // patchsize
        dis = dis.view(length * length, length * length).float()
        self.register_buffer('lap', dis)
        a = nn.Parameter(torch.ones(1))
        setattr(self, 'lap_a', a)

    def forward(self, query, key, value, mask=None):
        residual = query
        B, C, H, W = query.size()
        q = self.query_embedding(query)
        k = self.key_embedding(key)
        v = self.value_embedding(value)

        num_w = W // self.ps
        num_h = H // self.ps
        # 1) embedding and reshape

        q = q.view(B, C, num_h, self.ps, num_w, self.ps).contiguous()    # B * C* h/s * s * w/s * s
        k = k.view(B, C, num_h, self.ps, num_w, self.ps).contiguous()
        v = v.view(B, C, num_h, self.ps, num_w, self.ps).contiguous()
        # B * (h/s * w/s) * (C * s * s)
        q = q.permute(0, 2, 4, 1, 3, 5).contiguous().view(B,  num_h*num_w, C * self.ps * self.ps)
        k = k.permute(0, 2, 4, 1, 3, 5).contiguous().view(B,  num_h*num_w, C * self.ps * self.ps)
        v = v.permute(0, 2, 4, 1, 3, 5).contiguous().view(B,  num_h*num_w, C * self.ps * self.ps)
        dis = F.softplus(self.lap_a) * self.lap
        if mask is not None:
            m = mask.view(B, 1, num_h, self.ps, num_w, self.ps).contiguous()
            m = m.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, num_h * num_w, self.ps * self.ps).contiguous()
            m = (m.mean(-1) < 0.5).unsqueeze(1).repeat(1, num_w * num_h, 1)
            result, _ = self.attention(q, k, v, m, dis)

        else:
            result, _ = self.attention(q, k, v, dis=dis)
        # 3) "Concat" using a view and apply a final linear.
        result = result.view(B, num_h, num_w, C, self.ps,  self.ps).contiguous()
        result = result.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W).contiguous()
        output = self.output_linear(result)
        output = output + residual
        return output


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

# compute the l1 distance between feature patches
def lap(s):
    outk = []

    for i in range(s):
        for k in range(s):

            out = []
            for x in range(s):
                row = []
                for y in range(s):
                    cord_x = i
                    cord_y = k
                    dis_x = abs(x - cord_x)
                    dis_y = abs(y - cord_y)
                    dis_add = -(dis_x + dis_y)
                    row.append(dis_add)
                out.append(row)

            outk.append(out)

    out = np.array(outk)
    return torch.from_numpy(out)

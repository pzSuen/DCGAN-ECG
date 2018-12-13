import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import h5py
import numpy as np


# onehot编码，输入为一维行向量
def one_hot(x, mu=256):  # 1*250
    hot = np.zeros((mu, x.shape[0]))
    for i in np.arange(x.shape[0]):
        hot[x[i], i] = 1
    return hot


# 编码  
def encode_mu_law(x, mu=256):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.long)


# 解码
def decode_mu_law(y, mu=256):
    mu = mu - 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x


# 通道为1
def show_img(x):
    min, max = np.min(x), np.max(x)
    img = 255.0 * (x[:, :].squeeze() - min) / (max - min)
    return img


# 加载数据
def load_data():
    f = h5py.File("ecg_data_256.h5", 'r')
    ecg_train = f['ecg_train']
    ecg_test = f['ecg_test']
    return ecg_train, ecg_test


# 制作dataloader
def make_tensor():
    y = hot_y.reshape(hot_y.shape[0], -1, hot_y.shape[1], hot_y.shape[2])
    # z=np.concatenate((y,y,y),axis=1)              # 单通道到三通道
    data = t.from_numpy(y / 1.)

    torch_data = t.utils.data.TensorDataset(data, data)  # target不重要，没有用
    dataloader = t.utils.data.DataLoader(torch_data, opt.batch_size, shuffle=True, num_workers=opt.workers)
    return dataloader


# torch.Size([32, 3, 64, 64])
# 四维到三维
def demake_tensor(x):
    y = x.detach().cpu().numpy()
    N = y.shape[0]
    mu = y.shape[2]
    num_nodes = y.shape[3]
    #   z = np.sum(y,axis=1)
    z = y.reshape(N, mu, -1)
    assert z.shape == (N, mu, num_nodes), "dimension is wrong."
    #   return z/3.
    return z


# 找到最大值
# (num ,mu , num_nodes) --> (num , num_nodes)
def find_maximal(x):
    N = x.shape[0]
    num_nodes = x.shape[2]
    prob = np.zeros((N, 1, num_nodes))
    prob = np.max(x, axis=1)
    return prob


# 网络参数
class Config:
    lr = 0.0002  # learning rate
    nz = 100  # noise dimension
    image_size = 64
    image_size2 = 64
    nc = 1  # chanel of img
    ngf = 64  # generate channel
    ndf = 64  # discriminative channel
    beta1 = 0.5
    batch_size = 64
    #     max_epoch = 10 # =1 when debug
    max_epoch = 20  # =1 when debug
    workers = 2  # 加载数据的子进程数，0时只有主进程
    gpu = True  # use gpu or not


def train():
    # begin training
    # 调用dataloader
    dataloader = make_tensor()

    print('Start training......')

    loss_D = t.from_numpy(np.zeros(opt.max_epoch))
    loss_G = t.from_numpy(np.zeros(opt.max_epoch))

    for epoch in range(opt.max_epoch):
        step = 1
        for ii, data in enumerate(dataloader, 0):  # ii is step
            real, _ = data
            input = Variable(real)  # batch_size,channels,width,height
            label = Variable(t.ones(input.size(0)))  # 1 for real
            noise = t.randn(input.size(0), opt.nz, 1, 1)  # opt.nz=100
            noise = Variable(noise)

            if opt.gpu:
                noise = noise.cuda()
                input = input.cuda()
                label = label.cuda()

            # ----- train netd -----
            netd.zero_grad()
            ## train netd with real img    
            ## Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
            output = netd(input.float())
            error_real = criterion(output.squeeze(), label)
            error_real.backward()
            D_x = output.data.mean()
            ## train netd with fake img
            fake_pic = netg(noise).detach()
            output2 = netd(fake_pic)
            label.data.fill_(0)  # 0 for fake
            error_fake = criterion(output2.squeeze(), label)
            error_fake.backward()
            D_x2 = output2.data.mean()
            error_D = error_real + error_fake
            optimizerD.step()

            # ------ train netg -------
            netg.zero_grad()
            label.data.fill_(1)
            noise.data.normal_(0, 1)
            fake_pic = netg(noise)
            output = netd(fake_pic)
            error_G = criterion(output.squeeze(), label)
            error_G.backward()
            optimizerG.step()
            D_G_z2 = output.data.mean()

            loss_D[epoch] += error_D.item()
            loss_G[epoch] += error_G.item()
        loss_D[epoch] = np.mean(loss_D[epoch])
        loss_G[epoch] = np.mean(loss_G[epoch])

        if epoch % 2 == 0:
            #         print("epoch=",epoch)
            #         print('{epoch}  lossD:{loss_D},lossG:{loss_G}'.format(
            #                epoch=epoch,loss_D=loss_D,loss_G=loss_G))
            print("Epoch:", epoch)
            fake_u = netg(fix_noise)
            img1 = demake_tensor(fake_u)
            img2 = find_maximal(img1)
            ecg = decode_mu_law(img2, 64)
            print("ecg[0,:5]:", ecg[0, :5])
            # 显示未解码的数据图
            plt.subplot(131)
            plt.title("Data of Fake img-0")
            plt.plot(img1[0])
            # 显示未解码的图像
            plt.subplot(132)
            plt.title("Img of Fake img-0")
            #         img3=show_img(img1[0])
            plt.imshow(img1[0])
            # 显示解码之后的ECG图像
            plt.subplot(133)
            plt.title("Ecg-0 Generated")
            plt.plot(ecg[0, :])
            plt.tight_layout(pad=0.1, w_pad=1.0)
            plt.show()

    # save param
    t.save(netd.state_dict(), "netd.pth")
    t.save(netg.state_dict(), "netg.pth")

    # save loss
    np.savetxt("loss_D.txt", loss_D)
    np.savetxt("loss_G.txt", loss_G)

    # show loss
    plt.plot(loss_D, 'r--', label="Discriminator")
    plt.plot(loss_G, 'b--', label="Generator")
    plt.legend(loc='best', shadow=True, fancybox=True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss of G&D")
    plt.show()


if __name__ == "__main__":
    opt = Config()
    mu = 256

    # 读取数据
    ecg_train, ecg_test = load_data()
    # print(ecg_train[:10,:10])

    # 数据编码
    y = encode_mu_law(ecg_train[:, :64], mu=mu)  # 每个样本只取前64个样本点

    # onehot编码
    hot_y = np.zeros((y.shape[0], mu, y.shape[1]))  # 实例个数*mu*采样点个数
    for i in np.arange(hot_y.shape[0]):
        hot_y[i] = one_hot(y[i][:], mu=mu)

    # 生成器
    netg = nn.Sequential(
        # input size: nz
        nn.ConvTranspose2d(opt.nz, opt.ngf * 16, 4, 1, 0, bias=False),
        nn.BatchNorm2d(opt.ngf * 16),
        nn.ReLU(True),
        # state size: (ngf*16) x 4 x 4
        nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf * 8),
        nn.ReLU(True),
        # state size: (ngf * 8) x 8 x 8
        nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf * 4),
        nn.ReLU(True),
        # state size: (ngf * 4) x 16 x 16
        nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf * 2),
        nn.ReLU(True),
        # state size: (ngf * 2) x 32 x 32
        nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf),
        nn.ReLU(True),
        # state size: ngf x 64 x 64
        nn.ConvTranspose2d(opt.ngf, int(opt.ngf / 2), 4, 2, 1, bias=False),
        nn.BatchNorm2d(int(opt.ngf / 2)),
        nn.ReLU(True),
        # state size: (ngf/2) x 128 x 128
        nn.ConvTranspose2d(int(opt.ngf / 2), opt.nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # output size: nc x 256 x 256 (nc=1)
    )

    # 判别器
    netd = nn.Sequential(
        # input size: nc x 256 x 256 (nc=1)
        nn.Conv2d(opt.nc, int(opt.ndf / 2), 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size: (ndf/2) x 128 x 128
        nn.Conv2d(int(opt.ndf / 2), opt.ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size: ndf x 64 x64
        nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size: (ndf * 2) x 32 x 32
        nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size: (ndf * 4) x 16 x 16
        nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size: (ndf * 8) x 8 x 8
        nn.Conv2d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),
        # state size: (ndf * 16) x 4 x 4
        nn.Conv2d(opt.ndf * 16, 1, 1, 4, 0, bias=False),
        nn.Sigmoid()
        # output size: 1
    )

    optimizerD = Adam(netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))  # 论文建议0.5
    optimizerG = Adam(netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

    # criterion
    criterion = nn.BCELoss()  # compute loss

    fix_noise = Variable(t.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1))
    if opt.gpu:
        fix_noise = fix_noise.cuda()
        netd.cuda()
        netg.cuda()
        criterion.cuda()  # it's a good habit ，使用cuda计算损失

    # 训练
    train()

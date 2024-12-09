import socket
import timeit
from datetime import datetime
import cv2

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from tensorboardX import SummaryWriter

from dataloaders.combine_dbs import CombineDBs as combine_dbs
from dataloaders import sbd, pascal, davis, cityscapes_processed, cityscapes_custom
from dataloaders import custom_transforms as tr
import networks.backend_cnn as backend_cnn
from layers.loss import *
from layers.lse import *
from dataloaders.helpers import *
from evaluation.eval import *

# for gpu config
import os

class DELSE(object):
    def __init__(self, args):
        self.args = args

        # model
        self.model_name = args.model_name
        self.resolution = (args.resolution, args.resolution)

        self.net = backend_cnn.backend_cnn_model(args)

        if self.args.resume_epoch == 0:
            print("Initializing from pretrained Deeplab-v2 model")
        else:
            resume_dir = os.path.join(args.save_dir_root, 'run_%04d' % args.resume_id)
            print("Initializing weights from: {}".format(
                os.path.join(resume_dir, 'models', self.model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth')))
            self.net.load_state_dict(
                torch.load(os.path.join(resume_dir, 'models', self.model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth'),
                           map_location=lambda storage, loc: storage))

        # optimizer
        self.train_params = [{'params': self.net.get_1x_lr_params(), 'lr': args.lr},
                             {'params': self.net.get_10x_lr_params(), 'lr': args.lr * 10}]
        self.optimizer = optim.SGD(self.train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

        # transforms
        self.composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(), # ランダム水平反転
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)), # 上下限ありでスケール変更・角度変化
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=args.relax_crop, zero_pad=args.zero_pad_crop, dummy=True), # mask領域で画像を切り取り(サイズ変わる)
            tr.FixedResize(resolutions={'crop_image': self.resolution, 'crop_gt': self.resolution}), # crop_imageとcrop_gtの解像度変更
            tr.SDT(elem='crop_gt', dt_max=args.dt_max),
            # tr.ExtremePoints(sigma=10, pert=5, elem='crop_gt'),
            # tr.ToImage(norm_elem='extreme_points'),
            # tr.ConcatInputs(elems=('crop_image', 'extreme_points')), # ここでimageとextream pointsを結合
            tr.ToTensor()])
        self.composed_transforms_ts = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=args.relax_crop, zero_pad=args.zero_pad_crop, dummy=True),
            tr.FixedResize(resolutions={'void_pixels': None, 'gt': None, 'crop_image': self.resolution, 'crop_gt': self.resolution}),
            tr.SDT(elem='crop_gt', dt_max=args.dt_max),
            # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
            # tr.ToImage(norm_elem='extreme_points'),
            # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
            tr.ToTensor()])

        # dataset
        if self.args.dataset == 'pascal':
            self.trainset = pascal.VOCSegmentation(split='train', transform=self.composed_transforms_tr)
            self.valset = pascal.VOCSegmentation(split='val', transform=self.composed_transforms_ts, retname=True)
        elif self.args.dataset == 'pascal-sbd':
            self.valset = pascal.VOCSegmentation(split='val', transform=self.composed_transforms_ts, retname=True)
            voc_trainset = pascal.VOCSegmentation(split='train', transform=self.composed_transforms_tr)
            sbd_trainval = sbd.SBDSegmentation(split=['train', 'val'], transform=self.composed_transforms_tr, retname=True)
            self.trainset = combine_dbs([voc_trainset, sbd_trainval], excluded=[self.valset])
        elif self.args.dataset == 'davis2016':
            self.trainset = davis.DAVIS2016(train=True, transform=self.composed_transforms_tr)
            self.valset = davis.DAVIS2016(train=False, transform=self.composed_transforms_ts, retname=True)
        elif self.args.dataset == 'cityscapes-processed': # default
            self.trainset = cityscapes_custom.CityScapesCustom(train=True, split='train', transform=self.composed_transforms_tr)
            self.valset = cityscapes_custom.CityScapesCustom(train=False, split='val', transform=self.composed_transforms_ts, retname=True)
            # self.trainset = cityscapes_processed.CityScapesProcessed(train=True, split='train', transform=self.composed_transforms_tr)
            # self.valset = cityscapes_processed.CityScapesProcessed(train=False, split='val', transform=self.composed_transforms_ts, retname=True)

        # dataloader
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch, shuffle=True, num_workers=8)
        self.testloader = DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)

        self.num_img_tr = len(self.trainloader)
        self.num_img_ts = len(self.testloader)

        # tensorboard
        log_dir = os.path.join(args.save_dir, 'models',
                               args.txt + '_' + datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        # GPU setting
        if self.args.gpu != 'multi_gpu': # not default
            print(f"selected device list: {self.args.gpu}")
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        # torch.nn.DataParallelで囲えばPyTorchが利用可能な全てのGPUを使用するように設定される。
        # TODO: 使うgpuが1台だと余計なoverheadが発生するので、.to(device)に置き換える
        self.net = torch.nn.DataParallel(self.net).cuda()

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()

    def train(self, epoch):
        # init
        torch.cuda.empty_cache()
        self.net.train()

        running_loss_tr = 0.0
        aveGrad = 0
        start_time = timeit.default_timer()

        # main training loop
        for ii, sample_batched in enumerate(self.trainloader):
            # TODO : 100ミニバッチごとの実行タイミングを確認
            # if ii % 100 == 0:
            #     print(f"batch {ii} / {len(self.trainloader)} : {timeit.default_timer() - start_time}")

            if ii == self.num_img_tr - 1:
                break
            # data

            # inputs: crop_imageとextream_pointsを連結した入力画像データ
            # sdt   : signed distance transformのデータ
            # gts   : cropされたグラウンドトゥルースのセグメンテーションマスク
            inputs, sdts = sample_batched['crop_image'], sample_batched['sdt']
            gts = sample_batched['crop_gt']

            inputs.requires_grad_()
            inputs, sdts = inputs.cuda(), sdts.cuda()
            gts = torch.ge(gts, 0.5).float().cuda() # 0.5以上の値を持つピクセルを1, それ以外を0に変換

            # dts   : distance transformのデータ
            # vfs   : dtsに対してSobelフィルタを適用して計算された勾配データ
            dts = sample_batched['dt']
            dts = dts.cuda()  # (batch, 1, H, W)
            vfs = gradient_sobel(dts, split=False)

            # Forward of the mini-batch
            # phi_0 : 初期レベルセット
            # energy: ベクトル場
            # g     : 選択的に曲率による正規化を行う確率
            # phi_0, energy, g = self.net.forward(inputs)    # inputs: (batch, nchannels, 512, 512)
            
            # 可変長タプルで出力を受け取って、分岐させる
            outputs = (*(self.net.forward(inputs)), )
            if len(outputs) == 1:
                phi_0 = outputs[0]
                energy = None
                g = None
            elif len(outputs) == 2:
                phi_0, energy = outputs
                g = None
            else:
                phi_0, energy, g = outputs
            
            # (batch, nclasses, 64, 64) -> (batch, nclasses, size, size) : アップサンプリング
            # モデル出力は低解像度であるため、バイリニア補完を用いて高解像度に変換
            # g(ベクトル場)については出力値域を[0,1]にするためにsigmoid関数を適用
            phi_0 = F.upsample(phi_0, size=self.resolution, mode='bilinear', align_corners=True)
            energy = F.upsample(energy, size=self.resolution, mode='bilinear', align_corners=True)
            if g is not None:
                g = F.sigmoid(F.upsample(g, size=self.resolution, mode='bilinear', align_corners=True))
            
            # loss
            if not self.args.e2e and epoch < self.args.pretrain_epoch:
                # pre-train
                phi_0_loss = level_map_loss(phi_0, sdts, self.args.alpha) # 初期レベルセットと符号付距離のロス(要素差の2乗のMSE)
                rand_shift = 10 * np.random.rand() - 5
                phi_T = levelset_evolution(sdts + rand_shift, energy, g, T=self.args.T, dt_max=self.args.dt_max) # Tステップのレベルセット進化
                phi_T_loss = vector_field_loss(energy, vfs, sdts) \
                             + LSE_output_loss(phi_T, gts, sdts, self.args.epsilon, dt_max=self.args.dt_max) # (エネルギーマップとベクトル場のロス) + (Tステップ後のレベルセットとグラウンドトゥルースのロス)
                loss = phi_0_loss + phi_T_loss # 初期レベルセットとTステップ後のレベルセットで得られるロスを合わせる
                running_loss_tr += loss.item()
            else:
                # joint-train
                rand_shift = 10 * np.random.rand() - 5
                phi_T = levelset_evolution(phi_0 + rand_shift, energy, g, T=self.args.T, dt_max=self.args.dt_max)

                loss = LSE_output_loss(phi_T, gts, sdts, self.args.epsilon, dt_max=self.args.dt_max) # Tステップ後のレベルセットとグラウンドトゥルースのロス
                running_loss_tr += loss.item()

            # Backward the averaged gradient
            loss /= self.args.ave_grad
            loss.backward()
            aveGrad += 1

            # Update the weights once
            if aveGrad % self.args.ave_grad == 0:
                self.writer.add_scalar('data/total_loss_iter', loss.item(), ii + self.num_img_tr * epoch)
                if not self.args.e2e and epoch < self.args.pretrain_epoch:
                    self.writer.add_scalar('data/total_phi_0_loss_iter', phi_0_loss.item(), ii + self.num_img_tr * epoch)
                    self.writer.add_scalar('data/total_phi_T_loss_iter', phi_T_loss.item(), ii + self.num_img_tr * epoch)
                clip_grad_norm(self.net.parameters(), 10) # 勾配爆発を防ぐためにクリッピング
                self.optimizer.step()
                self.optimizer.zero_grad()
                aveGrad = 0

        # print
        running_loss_tr = running_loss_tr / self.num_img_tr
        self.writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
        # print('[Epoch: %d, numImages: %5d]' % (epoch, ii * self.args.batch + inputs.data.shape[0]))
        # print('Loss: %f' % running_loss_tr)
        stop_time = timeit.default_timer()
        # print("Execution time: " + str(stop_time - start_time) + "\n")
        
        return running_loss_tr


    def test(self, epoch, _save=True):
        # init
        torch.cuda.empty_cache()
        self.net.eval()

        save_dir_res = os.path.join(self.args.save_dir, 'results_ep' + str(epoch))
        if _save and (not os.path.exists(save_dir_res)):
            os.makedirs(save_dir_res)

        # main testing loop
        with torch.no_grad():
            IoU = 0.0
            running_loss_ts = 0.0
            start_time = timeit.default_timer()

            for ii, sample_batched in enumerate(self.testloader):
                # TODO : 100ミニバッチごとの実行タイミングを確認
                # if ii % 100 == 0:
                #    print(f"batch {ii} / {len(self.testloader)} : {timeit.default_timer() - start_time}")
                
                inputs, sdts, metas = sample_batched['crop_image'], sample_batched['sdt'], sample_batched['meta']
                gts = sample_batched['crop_gt']

                # Forward of the mini-batch
                inputs, sdts = inputs.cuda(), sdts.cuda()
                gts = torch.ge(gts, 0.5).float().cuda()

                # phi_0, energy, g = self.net.forward(inputs)
                # 可変長タプルで出力を受け取って、分岐させる
                outputs = (*(self.net.forward(inputs)), )
                if len(outputs) == 1:
                    phi_0 = outputs[0]
                    energy = None
                    g = None
                elif len(outputs) == 2:
                    phi_0, energy = outputs
                    g = None
                else:
                    phi_0, energy, g = outputs
                
                phi_0 = F.upsample(phi_0, size=self.resolution, mode='bilinear', align_corners=True)
                energy = F.upsample(energy, size=self.resolution, mode='bilinear', align_corners=True)
                if g is not None:
                    g = F.sigmoid(F.upsample(g, size=self.resolution, mode='bilinear', align_corners=True))

                phi_T = levelset_evolution(phi_0, energy, g, T=self.args.T, dt_max=self.args.dt_max, _test=True)
                loss = LSE_output_loss(phi_T, gts, sdts, self.args.epsilon, dt_max=self.args.dt_max)
                running_loss_ts += loss.item()

                # cal & save!
                format = lambda x: np.squeeze(np.transpose(x.cpu().data.numpy()[0, :, :, :], (1, 2, 0)))
                phi_T = format(phi_T)

                # TODO; ここどうにかする → phi_Tで終わりで良い。
                result = phi_T
                
                gt = tens2image(sample_batched['gt'][0, :, :, :])
                # gtのbboxを取得
                # bbox = get_bbox(gt, pad=self.args.relax_crop, zero_pad=self.args.zero_pad_crop)
                # phi_Tをbboxサイズにリサイズして、画像マスクのbbox領域に埋め込む
                # 背景値20の、phi_Tがクラス数の分だけマスクを持っている
                # resultのうち、あるクラスマスクに対して、<= 0でフィルタをかければ、マスクを取得できる
                result = crop2fullmask(phi_T, None, gt, zero_pad=self.args.zero_pad_crop, relax=self.args.relax_crop, bg_value=20)

                # IoU
                if 'pascal' in self.args.dataset:
                    void_pixels = tens2image(sample_batched['void_pixels'][0, :, :, :])
                    void_pixels = (void_pixels >= 0.5)
                else:
                    void_pixels = None
                # self.args.mask_threshold以下でresult(phi_T)をフィルターして、gtと比べて、IoUを出す
                # TODO: gtとresult <= -2のshapeが異なるらしい
                IoU += jaccard(gt, (result <= self.args.mask_threshold), void_pixels)

                # save outputs
                # 出力の保存部分(別途、diplay.pyを使って可視化)
                if _save:
                    # round() is used to save space
                    np.save(os.path.join(save_dir_res, metas['image'][0] + '-' + metas['object'][0] + '.npy'),
                            result.round().astype(np.int8))

            # Print stuff at end of testing
            running_loss_ts = running_loss_ts / self.num_img_ts
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii + inputs.data.shape[0]))
            self.writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
            print('Loss: %f' % running_loss_ts)

            mIoU = IoU / self.num_img_ts
            self.writer.add_scalar('data/test_mean_IoU', mIoU, epoch)
            print('Mean IoU: %f' % mIoU)

            stop_time = timeit.default_timer()
            print("Test time: " + str(stop_time - start_time) + "\n")
            
            return running_loss_ts ,mIoU

    def save_ckpt(self, epoch):
        torch.save(self.net.module.state_dict(),
                   os.path.join(self.args.save_dir, 'models', self.model_name + '_epoch-' + str(epoch) + '.pth'))

    def adjust_learning_rate(self, epoch, adjust_epoch=None, ratio=0.1):
        """Sets the learning rate to the initial LR decayed at selected epochs"""
        if epoch in adjust_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * ratio


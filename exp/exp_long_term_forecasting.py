from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class WeightedL1Loss:
    def __init__(self, alpha, loss_mode):
        self.alpha = alpha
        self.loss_mode = loss_mode
        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')
        elif self.loss_mode == 'L1L2':
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')

    def __call__(self, pred, gt):
        # [b,l,n]
        if pred.ndim == 1:
            # imputation
            mask = torch.isnan(gt)
            if torch.any(mask):
                # pred, gt = pred.masked_fill(mask, 0), gt.masked_fill(mask, 0)
                pred, gt = pred[~mask], gt[~mask]

            loss_fun = nn.L1Loss(reduction='mean')
            weightedLoss = loss_fun(pred, gt)
        else:
            L = pred.shape[1]
            weights = (torch.tensor([(i + 1) ** (-self.alpha) for i in range(L)]).unsqueeze(dim=0).unsqueeze(dim=-1)
                       .to(pred.device))
            if self.loss_mode in ['L1', 'L2']:
                loss_vec = self.loss_fun(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights)
            elif self.loss_mode == 'L1L2':
                loss_vec = self.loss_fun1(pred, gt)
                loss_vec2 = self.loss_fun2(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights + loss_vec2 * weights)
            else:
                raise NotImplementedError
        return weightedLoss

def spectral_entropy(x, eps=1e-8):
    # x: (batch, seq_len)
    fft = torch.fft.fft(x, dim=-1)
    psd = fft.real**2 + fft.imag**2
    psd = psd[:, :x.shape[-1]//2]  # 取前半段频谱
    psd_sum = psd.sum(dim=-1, keepdim=True) + eps
    p = psd / psd_sum
    entropy = -torch.sum(p * torch.log(p + eps), dim=-1)
    max_entropy = torch.log2(torch.tensor(psd.shape[-1], dtype=psd.dtype, device=psd.device))
    entropy = entropy / max_entropy
    return entropy

# class SELoss(nn.Module):
#     def __init__(self):
#         super().__init__()

def SELoss(y_pred, y_true):
    
    # return spectral_entropy(y_pred, y_true)
    y_pred_fft = torch.fft.fft(y_pred.permute(0, 2, 1))
    y_true_fft = torch.fft.fft(y_true.permute(0, 2, 1))
    y_pred_power = torch.abs(y_pred_fft)
    y_true_power = torch.abs(y_true_fft)
    y_pred_se = spectral_entropy(y_pred_power)
    y_true_se = spectral_entropy(y_true_power)
    
    return ((y_pred_se - y_true_se) ** 2).mean()

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        print(f'=======total parameters=======: {total_params} .')
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs, _ = self.model(batch_x, batch_x_mark, is_training=False)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print(len(train_data), len(vali_data), len(test_data))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # criterion = WeightedL1Loss(alpha=0.5, loss_mode='L1')
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, my_tuple = self.model(batch_x, batch_x_mark, is_training=True)
                if type(my_tuple) is tuple or type(my_tuple) is list:
                    moe_loss = my_tuple[0]
                    # pred_mean = my_tuple[1]
                    # pred_std = my_tuple[2]
                else:
                    moe_loss = my_tuple
                
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # true_mean = batch_y.mean(dim=1, keepdim=True)
                # true_std = batch_y.std(dim=1, keepdim=True)
                
                ext_loss = SELoss(outputs, batch_y) # criterion(pred_mean, true_mean) + criterion(pred_std, true_std)
                # loss = criterion(outputs, batch_y) + ext_loss
                
                # loss_feq = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean() 
                # alpha_feq = 0.85
                alpha = 0.1
                loss = criterion(outputs, batch_y) + alpha * moe_loss + 0.05 * ext_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            criterion = self._select_criterion()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, _ = self.model(batch_x, batch_x_mark, is_training=False)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                input_ = batch_x
                pred = outputs
                true = batch_y

                inputs.append(input_)
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # result save
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'input.npy', inputs)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

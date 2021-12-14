import numpy as np
from tqdm import tqdm
import wandb
import time

import torch
import torch.optim as optim

from model import MultiExit
from utils import Distillation

import warnings

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiExit().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = Distillation()

        self.best_val_loss = float('inf')
        self.args = args

    def train(self, train_iter, val_iter, length):
        train_size, val_size = length
        if self.args.wandb:
            wandb.init(project="ESW-Final-Project", entity="nadenny")
            wandb.config = {}
            print('\n')
            time.sleep(1)

        for epoch in range(self.args.epochs):
            print('\n')
            print("=============================== Epoch: ", epoch + 1, " of ", self.args.epochs,
                  "===============================")
            b_acc1, b_acc2, b_acc3, b_acc4 = 0, 0, 0, 0
            b_l1, b_l2, b_l3, b_l4 = 0, 0, 0, 0

            self.model.train()
            for source, target in tqdm(train_iter):
                source = source.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                o1, o2, o3, o4 = self.model(source)

                loss1, loss2, loss3, loss4 = self.criterion(o1, o2, o3, o4, target)
                loss = loss1 + loss2 + loss3 + loss4
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                b_acc1 += self._acc(target, o1)
                b_acc2 += self._acc(target, o2)
                b_acc3 += self._acc(target, o3)
                b_acc4 += self._acc(target, o4)

                b_l1 += loss1
                b_l2 += loss2
                b_l3 += loss3
                b_l4 += loss4

            t_acc1 = b_acc1 / train_size
            t_acc2 = b_acc2 / train_size
            t_acc3 = b_acc3 / train_size
            t_acc4 = b_acc4 / train_size

            t_loss1 = b_l1 / len(train_iter)
            t_loss2 = b_l2 / len(train_iter)
            t_loss3 = b_l3 / len(train_iter)
            t_loss4 = b_l4 / len(train_iter)

            v_loss1, v_loss2, v_loss3, v_loss4, v_acc1, v_acc2, v_acc3, v_acc4 = self.validate(val_iter)

            v_acc1 = v_acc1 / val_size
            v_acc2 = v_acc2 / val_size
            v_acc3 = v_acc3 / val_size
            v_acc4 = v_acc4 / val_size
            v_loss = np.average((v_loss1.to('cpu'), v_loss2.to('cpu'), v_loss3.to('cpu'), v_loss4.to('cpu'),))
            if v_loss < self.best_val_loss:
                self.save_param()
                self.best_val_loss = v_loss

            print(f'\rTrain Loss: {t_loss1:.3f}, {t_loss2:.3f}, {t_loss3:.3f}, {t_loss4: .3f} | '
                  f' Train Acc.: {t_acc1:.3f}, {t_acc2:.3f}, {t_acc3:.3f}, {t_acc4: .3f} | '
                  f'Val. Loss: {v_loss1:.3f}, {v_loss2:.3f}, {v_loss3:.3f}, {v_loss4: .3f} | '
                  f'Val. Acc: {v_acc1:.3f}, {v_acc2:.3f}, {v_acc3:.3f}, {v_acc4: .3f}'
                  , end='', flush=True)
            print('\n ')

            if self.args.wandb:
                wandb.log({
                    'train_loss1': t_loss1,
                    'train_loss2': t_loss2,
                    'train_loss3': t_loss3,
                    'train_loss4': t_loss4,

                    'train_acc1': t_acc1,
                    'train_acc2': t_acc2,
                    'train_acc3': t_acc3,
                    'train_acc4': t_acc4,

                    'val_loss1': v_loss1,
                    'val_loss2': v_loss2,
                    'val_loss3': v_loss3,
                    'val_loss4': v_loss4,

                    'val_acc1': v_acc1,
                    'val_acc2': v_acc2,
                    'val_acc3': v_acc3,
                    'val_acc4': v_acc4,
                })

    @torch.no_grad()
    def validate(self, valid_iter):
        self.model.eval()
        b_acc1, b_acc2, b_acc3, b_acc4 = 0, 0, 0, 0
        b_l1, b_l2, b_l3, b_l4 = 0, 0, 0, 0
        for source, target in tqdm(valid_iter):
            source = source.to(self.device)
            target = target.to(self.device)

            o1, o2, o3, o4 = self.model(source)
            loss1, loss2, loss3, loss4 = self.criterion(o1, o2, o3, o4, target)

            b_acc1 += self._acc(target, o1)
            b_acc2 += self._acc(target, o2)
            b_acc3 += self._acc(target, o3)
            b_acc4 += self._acc(target, o4)

            b_l1 += loss1
            b_l2 += loss2
            b_l3 += loss3
            b_l4 += loss4

        b_l1 = b_l1 / len(valid_iter)
        b_l2 = b_l2 / len(valid_iter)
        b_l3 = b_l3 / len(valid_iter)
        b_l4 = b_l4 / len(valid_iter)
        return b_l1, b_l2, b_l3, b_l4, b_acc1, b_acc2, b_acc3, b_acc4

    def _acc(self, target, output):
        acc = 0
        out = torch.argmax(output, dim=1)
        for i in range(target.shape[0]):
            if target[i] == out[i]:
                acc += 1
        return acc

    def save_param(self):
        param = {"model_state": self.model.state_dict(),
                 "optim_state": self.optimizer.state_dict(),
                 "best_loss": self.best_val_loss}
        torch.save(param, 'my_model.pth')

    def load_param(self):
        param = torch.load('my_model.pth')
        self.model.load_state_dict(param['model_state'])
        self.optimizer.load_state_dict(param['optim_state'])
        self.best_val_loss = param['best_loss']

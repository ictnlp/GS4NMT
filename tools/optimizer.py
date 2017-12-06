import torch.optim as opt
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from utils import wlog
import math
import wargs

class Optim(object):

    def __init__(self, opt_mode, learning_rate, max_grad_norm, learning_rate_decay=1,
                 start_decay_from=None, last_valid_bleu=None, warmup_steps=4000):

        self.opt_mode = opt_mode
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.lr_decay = learning_rate_decay
        self.start_decay_from = start_decay_from
        self.last_valid_bleu = last_valid_bleu
        self.start_decay = False

        #self.step_num = 0
        #self.warmup_steps = warmup_steps

    def __repr__(self):

        return '\nMode: {}\nLearning rate: {}\nGrad norm: {}\nlearning rate decay: {}\nstart '\
                    'decay from: {}\nlast valid bleu: {}'.format(
                    self.opt_mode, self.learning_rate, self.max_grad_norm, self.lr_decay,
                        self.start_decay_from, self.last_valid_bleu)

    def zero_grad(self):

        self.optimizer.zero_grad()

    def init_optimizer(self, params):

        # careful: params may be a generator
        # self.params = params
        self.params = list(params)
        self.params = filter(lambda p: p.requires_grad, self.params)

        if self.opt_mode == 'sgd':
            self.optimizer = opt.SGD(self.params, lr=self.learning_rate)
        elif self.opt_mode == 'adagrad':
            self.optimizer = opt.Adagrad(self.params, lr=self.learning_rate)
        elif self.opt_mode == 'adadelta':
            self.optimizer = opt.Adadelta(self.params, lr=self.learning_rate, rho=0.95)
            #self.optimizer = opt.Adadelta(self.params, lr=self.learning_rate, rho=0.95, eps=10e-06)
            #self.optimizer = opt.Adadelta(self.params, lr=self.learning_rate, rho=0.95, weight_decay=10e-5)
        elif self.opt_mode == 'adam':
            self.optimizer = opt.Adam(self.params,
                                      lr=self.learning_rate, betas=[0.9, 0.98], eps=10e-9)
        else:
            wlog('Do not support this opt_mode {}'.format(self.opt_mode))

    def step(self):

        #self.step_num += 1

        # attention is all you need
        #lr_vary = math.pow(wargs.enc_hid_size, -0.5) * min(
        #    math.pow(self.step_num, -0.5), self.step_num * math.pow(self.warmup_steps, -1.5))
        #self.learning_rate = wargs.learning_rate * lr_vary
        #self.optimizer.param_groups[0]['lr'] = self.learning_rate

        # clip by the gradients norm
        if self.max_grad_norm:
            clip_grad_norm(self.params, max_norm=self.max_grad_norm)

        self.optimizer.step()

    def update_learning_rate(self, bleu, epoch):

        if self.start_decay_from is not None and epoch >= self.start_decay_from:

            self.start_decay = True

        # comparing last epoch, it becomes worse
        if self.start_decay_from is not None and bleu < self.last_valid_bleu:

            self.start_decay = True

        if self.start_decay:

            self.learning_rate = self.learning_rate * self.lr_decay
            wlog('Decaying learning rate to {}'.format(self.learning_rate))

        self.last_valid_bleu = bleu
        self.optimizer.param_groups[0]['lr'] = self.learning_rate

        '''
        param_groups:
        [{'betas': (0.9, 0.999),
          'eps': 1e-08,
          'lr': 0.0001,
          'params': [Variable containing:
           -0.7941 -0.9056 -0.1569
           -0.7084  1.7447 -0.6319
           [torch.FloatTensor of size 2x3], Variable containing:
           -1.0234 -0.2506 -0.3016  0.7835
            0.1354 -1.1608 -0.7858  0.2127
           -0.6725 -0.8482 -0.6999  1.5561
           [torch.FloatTensor of size 3x4]],
          'weight_decay': 0}]
        '''


from warmup_scheduler import GradualWarmupScheduler
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, StepLR, LambdaLR
from torch import optim
from adamp import AdamP

# define your params



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def fetch_optimizer(args, params, hparams):
    
    if args.optimizer == 'Adam':
        optimizer = Adam(params, lr=hparams.learning_rate, weight_decay=params.weight_decay)
    elif args.optimizer == 'RAdam':
        optimizer = radam(params, lr=hparams.learning_rate, betas=(0.9,0.999), 
                        eps=1e-3, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(params, lr=hparams.learning_rate)  
    elif args.optimizer == 'SGD':
        optimizer = SGD(params, momentum=0.9, lr=hparams.learning_rate, weight_decay=args.weight_decay)   
    elif args.optimizer == 'AdamP':
        optimizer = AdamP(params, lr=hparams.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
    else:
        NotImplementedError
    
    return optimizer


def fetch_scheduler(args, optimizer):

    if args.scheduler == 'CosineAnnealingLR':
        T_max = 10
        eta_min = 1e-6
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.T_max, eta_min=args.eta_min)
    elif args.scheduler == 'StepLR':
        step_size = 1
        gamma = 0.95
        scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif args.scheduler == 'LambdaLR':
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 / (epoch+1))
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=args.plateau_factor, patience=args.patience)
    elif args.scheduler == 'WarmupV2':
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.T_max)
        scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)
    elif args.scheduler == "CosWarm":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=args.eta_min, last_epoch=-1)
    else:
        NotImplementedError

    return scheduler

def fetch_combined_scheduler(args, sch_name, optimizer):
    if sch_name == 'Cosine':
        T_max = 10
        eta_min = 1e-5
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min)
    elif sch_name == 'Steplr':
        step_size = 1
        gamma = 0.95
        scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif sch_name == 'Lambda':
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 / (epoch+1))
    elif sch_name == 'Plateau':
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=args.plateau_factor, patience=args.patience)
    elif sch_name == 'WarmupV2':
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=9)
        scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)
    elif sch_name == "CosWarm":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    else:
        NotImplementedError

    return scheduler


def radam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
            if isinstance(betas, str):
                betas = eval(betas)
            return optim.RAdam(parameters,
                              lr=lr,
                              betas=betas,
                              eps=eps,
                              weight_decay=weight_decay)

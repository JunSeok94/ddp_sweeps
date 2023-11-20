import wandb
import socket
import json
import setproctitle
import argparse
import os
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import psutil
num_cpus = psutil.cpu_count(logical=True)
world_size = torch.cuda.device_count() # num of available gpu
torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--temporal_kernel_size', default=11, type=int)
parser.add_argument("--residual", default=False, action='store_true')

parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--ray", default=False, action='store_true')
parser.add_argument('--num_process', default=1, type=int)
parser.add_argument('--save_dir', type=str)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--fold_num', default=5, type=int)
parser.add_argument('--test_ratio', default=0.2, type=float)
parser.add_argument('--val_ratio', default=0.2, type=float)
parser.add_argument('--epochs', default=5, type=int)

parser.add_argument('--project_name', type=str)
parser.add_argument('--group_name', type=str)
parser.add_argument('--description', type=str)
args = parser.parse_args()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features=400, out_features=100)
        self.output_linear = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        output1 = self.linear1(x)
        output2 = self.output_linear(output1)

        return output2


class CustomDataset(Dataset): 
    def __init__(self):
        self.x_data = torch.randn(size=(30000, 400))
        self.y_data = torch.empty(30000, dtype=torch.float).random_(3)

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

    def make_kfold_splits(self, args):
        self.kfolds = []
        kf = KFold(n_splits=int(1/args.test_ratio), shuffle=True, random_state=9404)

        for idx, (a, b) in enumerate(kf.split(dataset.x_data)):
            train_index, valid_index = train_test_split(a, test_size=int(1/args.val_ratio), random_state=9404)
            test_index = b
            self.kfolds.append((train_index, valid_index, test_index))


class Trainer:
    def __init__(self, args, model, criterion, optimizer, rank=None, wandb_run=None):
        self.args = args
        self.wandb = wandb_run

        if rank is None:
            self.rank = args.device 

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.wandb_run = wandb_run

    def train_valid(self, dataset, ddp_batch_size, train_sampler, valid_sampler):
        self.model.train()
        for epoch in range(1, self.args.epochs+1):
            self.epoch = epoch
            train_sampler.set_epoch(self.epoch)
            valid_sampler.set_epoch(self.epoch)
            train_loader = DataLoader(dataset, batch_size=ddp_batch_size, pin_memory=True, sampler=train_sampler)
            valid_loader = DataLoader(dataset, batch_size=ddp_batch_size, pin_memory=True, sampler=valid_sampler)

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.rank), target.to(self.rank)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.squeeze())
                loss.backward()
                self.optimizer.step()

            if self.wandb_run is not None:
                #self.wandb_run.log({'epoch': self.epoch})
                wandb_dict = {'epoch': self.epoch, 'train/loss': loss.item()}
                self.wandb_run.log(wandb_dict)

            # validation
            self.test(dataset, ddp_batch_size, valid_sampler, 'valid')
    
    @torch.no_grad()
    def test(self, dataset, ddp_batch_size, test_sampler, wandb_label='test'):
        test_sampler.set_epoch(self.epoch)
        test_loader = DataLoader(dataset, batch_size=ddp_batch_size, pin_memory=True, sampler=test_sampler)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.rank), target.to(self.rank)
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)

            if self.wandb_run is not None:
                #self.wandb_run.log({'epoch': self.epoch}
                wandb_dict = {'epoch': self.epoch, f'{wandb_label}/loss': loss.item()}
                self.wandb_run.log(wandb_dict)
            

def reset_wandb_env():
    exclude = {
        'WANDB_PROJECT',
        'WANDB_ENTITY',
        'WANDB_API_KEY',
    }
    
    for k, v in os.environ.items():
        if k.startswith('WANDB_') and k not in exclude:
            del os.environ[k]

def setup_wandb(args):
    # id for local name (Do not touch if using sweeps)!!!!!!!!!
    # dir for local dir (Do not touch if using sweeps)!!!!!!!!!
    wandb_init = dict()
    wandb_init['project'] = args.project_name
    wandb_init['group'] = args.group_name
    wandb_init['notes'] = args.description
    wandb_init['name'] = f'fold_{args.fold_num}' 
    #wandb_init['dir'] = f'wandb/{args.group_name}/' ##### do not edit in sweeps
    wandb_init['allow_val_change'] = True # allow to config update
    wandb_run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))

    return wandb_run 


def debug(args, dataset):
    setproctitle.setproctitle(f'debug')

    train_sampler, valid_sampler, test_sampler = [RandomSampler(x) for x in dataset.kfolds[args.fold_num-1]]

    # Load the model after DDP init
    print(f"Run the model on dataset fold #{args.fold_num}")
    model = Model().to(args.device)
    criterion = torch.nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # save the model and loss
    save_name = f'fold_{args.fold_num}'
    args.save_name = os.path.join(args.save_dir, f'{save_name}')

    #edited_args = save_args(args)
    trainer = Trainer(args, model, criterion, optimizer)
    trainer.train_valid(dataset, ddp_batch_size, train_sampler, valid_sampler)
    trainer.test(dataset, ddp_batch_size, test_sampler, wandb_label='test')

def get_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('',0))
    addr = s.getsockname()
    port = addr[-1]
    return port

def run_single_fold(rank, world_size, args, dataset):
    setproctitle.setproctitle(f'fold_{args.fold_num}_gpu_{rank}')

    ###################################### DDP init, do not touch ######################################
    # rank is gpu number
    # world_size: ngpus_per_node * world_size is all available gpu number
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{args.port}',
        rank=rank,
        world_size=world_size
    )
    ###################################### DDP init, do not touch ######################################

    # Make dataloaders with DistributedSampler after DDP init
    samplers = [DistributedSampler(x) for x in dataset.kfolds[args.fold_num-1]]
    train_sampler, valid_sampler, test_sampler = [DistributedSampler(x) for x in dataset.kfolds[args.fold_num-1]]
    ddp_batch_size = int(args.batch_size/world_size)
    print(f'Batch size per each gpu: {ddp_batch_size}')

    # Load the model after DDP init
    if rank == 0: print(f"Run the model on dataset fold #{args.fold_num} on gpu {rank}")
    model = Model().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = torch.nn.MSELoss().to(rank)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    # save the model and loss
    save_name = f'fold_{args.fold_num}_gpu_{rank}'
    args.save_name = os.path.join(args.save_dir, f'{save_name}')

    # WANDB setup
    if not args.debug and rank == 0:
        reset_wandb_env()
        wandb_run = setup_wandb(args)
        wandb_run.define_metric('epoch')
        wandb_run.define_metric('loss/*', step_metric='epoch')
        wandb_run.define_metric('train/*', step_metric='epoch')
        wandb_run.define_metric('valid/*', step_metric='epoch')
        wandb_run.define_metric('test/*', step_metric='epoch')
        wandb_run.name = f'fold_{args.fold_num}_gpu_{rank}'
        wandb_run.notes = f'{wandb_run.get_project_url()}/groups/{wandb_run.sweep_id}'
        wandb_run.watch(model, log="all")
    else: wandb_run = None

    if wandb_run is not None:
        edited_args = save_args(args)
        wandb_run.config.update(edited_args) ### after editing args for readability

    #edited_args = save_args(args)
    trainer = Trainer(args, model, criterion, optimizer)
    trainer.train_valid(dataset, ddp_batch_size, train_sampler, valid_sampler)
    trainer.test(dataset, ddp_batch_size, test_sampler, wandb_label='test')

    '''
    print(loss.item())
    wandb_run.log(loss.item())    # eval_result is dict, please log first in the Trainer
    '''
    wandb.join()

def save_args(args):
    import copy
    args_dict = vars(copy.deepcopy(args))

    del args_dict['debug']
    del args_dict['ray']
    del args_dict['num_process']
    del args_dict['port']
    del args_dict['device']
    json.dump(args_dict, open(args.save_dir + '/debug_args.json', 'w'), indent=4)

    del args_dict['save_dir']
    del args_dict['project_name']
    del args_dict['group_name']
    del args_dict['description']
    #del args_dict['']

    return args_dict

'''
@ray.remote(num_gpus=world_size)
def multigpu(world_size, args, dataset):
    loss = torch.multiprocessing.spawn(
        run_single_fold, 
        args=(world_size, args, dataset),
        nprocs=world_size,
        join=True,
    ) # spawn does not return

    return loss
'''


if __name__ == '__main__': 
    if torch.cuda.is_available():
        args.device = torch.device('cuda')                  # all gpu
        gpu_available = os.environ['CUDA_VISIBLE_DEVICES']
        device = f'cuda: {gpu_available}'
    else:
        device = 'cpu'
        args.device = torch.device(device)
    args.port = get_port()
    
    print('################################################')
    print(f'Run fold {args.fold_num}')
    print(f'CPU\t {psutil.cpu_count(logical=True)}')
    print(f'Device\t {device}')
    print('################################################')
    
    # Create dataset
    dataset = CustomDataset()
    dataset.make_kfold_splits(args)

    if args.debug: 
        print('########################## Debug Mode ##########################')
        debug(args, dataset)
    else:
        loss = torch.multiprocessing.spawn(
            run_single_fold, 
            args=(world_size, args, dataset),
            nprocs=world_size,
            join=True,
        ) # spawn does not return

import os, wandb, argparse
from datetime import datetime
now = datetime.now().strftime('%Y%m%d_%H%M%S')

sweep_config = {
    'project': 'test',
    'method': 'bayes',
    'name': now,
    'metric': {'goal': 'minimize', 'name': 'Eval_loss'},
    'parameters':
    {
        'dropout_rate': {'min': 0.0, 'max': 0.5},
        'temporal_kernel_size': {'values': [3, 5, 7, 9]},
        'residual': {'values': ['false', 'true']},
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--temporal_kernel_size', default=11, type=int)
parser.add_argument("--residual", default=False, type=lambda s: s.lower() == 'true')

# wandb off and multigpu off for debuging
parser.add_argument('--debug', '-d',  default=False, action='store_true')
args = parser.parse_args()

################################### edit below ###################################

num_process = 5       # number of folds run concurrently

gpu = '0'
batch_size = 65536
test_ratio = 1/5      
val_ratio = 1/5       # except test set
epochs = 500

# wandb setting
project_name = 'sweeps_multiprocessing_multigpu'
#group_name = f'dr_{args.dropout_rate}_temp_{args.temporal_kernel_size}_res_{str(args.residual)}_'
group_name = now
online_name = 'main' # the name of main record in the group
description = 'sweeps_example'

if args.debug: gpu = args.gpu.split(',')[0]

################# Keep 1 THREADS for avoiding cpu problem ###############
SCRIPT_LINE = f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu}' \
+f' OMP_NUM_THREADS=1' \
+f' MKL_NUM_THREADS=1' \
+f' NUMEXPR_NUM_THREADS=1' \
+f' OPENBLAS_NUM_THREADS=1' \
+f' python main.py' \
+f' --num_process {num_process}' \
+f' --batch_size {batch_size}' \
+f' --test_ratio {test_ratio}' \
+f' --val_ratio {val_ratio}' \
+f' --epochs {epochs}' \
+f' --project_name {project_name}' \
+f' --group_name {group_name}' \
+f' --description {description}'

#+f' --dropout_rate {args.dropout_rate}' \
#+f' --temporal_kernel_size {args.temporal_kernel_size}' \
#if args.residual: SCRIPT_LINE += ' --residual'
if args.debug: SCRIPT_LINE += ' --debug'

#################################################################################
#################################################################################

assert test_ratio < 1
assert val_ratio < 1
save_dir = os.path.join(os.getcwd(), 'saved')
os.makedirs(save_dir, exist_ok=True)
SCRIPT_LINE += f' --save_dir {save_dir}'

def setup_sweeps():
    '''
    id for local name (Do not touch if using sweeps)!!!!!!!!!
    dir for local dir (Do not touch if using sweeps)!!!!!!!!!
    '''

    wandb_init = dict()
    wandb_init['project'] = project_name
    wandb_init['group'] = group_name
    wandb_init['notes'] = description
    wandb_init['name'] = online_name
    wandb_init['allow_val_change'] = True # allow to config update
    #os.environ['WANDB_START_METHOD'] = 'thread'

    sweeps = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
    assert sweeps is wandb.run
    #sweeps.dir = f'wandb/{args.group_name}/'

    #sweeps.config.update(args)
    #sweep.save('main.h5') # save nothing

    return sweeps

def run_process(SCRIPT_LINE, fold_num):
    SCRIPT_LINE += f' --fold_num {fold_num}'
    SCRIPT_LINE += f' --dropout_rate {wandb.config.dropout_rate}' 
    SCRIPT_LINE += f' --temporal_kernel_size {wandb.config.temporal_kernel_size}'
    if wandb.config.residual: SCRIPT_LINE += ' --residual'

    os.system(f'{SCRIPT_LINE}')
    return 1

def multiprocess(SCRIPT_LINE):
    from multiprocessing import Pool
    #pool = Pool(num_process)
    pool = Pool()

    last_fold_num = int(1/test_ratio)
    all_folds = [*range(1, last_fold_num+1)]
    run_folds_list = [all_folds[start_fold:(start_fold+num_process)]
                      for start_fold in range(0, last_fold_num, num_process)]

    eval_loss = 0
    for folds in run_folds_list:
        args_list = [(SCRIPT_LINE, fold_idx)
                     for fold_idx in folds]
        for eval_result in pool.starmap(run_process, args_list):
            eval_loss += eval_result
    pool.close()
    pool.join()

    mean_eval_loss = eval_loss / last_fold_num
    return mean_eval_loss

def main():
    if args.debug: # wandb off for debuging
        run_process(SCRIPT_LINE, fold_num=0)
    else:
        # sweeps = setup_sweeps()
        sweeps = wandb.init()
        multiprocess(SCRIPT_LINE)

    '''
    save_dir = sweeps.dir
    os.makedirs(save_dir, exist_ok=True)

    train_loss = pickle.load(open(args.save_dir + f'/{save_name}_train.loss', 'rb'))
    print(f'Train loss on fold {args.fold_num}: {train_loss}')

    test_loss = pickle.load(open(args.save_dir + f'/{save_name}_test.loss', 'rb'))
    print(f'Test loss on fold {args.fold_num}: {test_loss}')
    '''

    '''
    train_loss = np.mean(train_losses)
    test_loss = np.mean(test_losses)
    print('')
    print(f'Train loss of {fold_num} cross validation: {train_loss}')
    print(f'Test loss of {fold_num} cross validation: {test_loss}')

    if sweeps is not None:
        sweep_run.log(dict(
            Eval_loss=eval_loss,
            Eval_accuracy=eval_acc
        ))
        sweeps.join() # wait to finish sweeps

    print(f'Sweep URL:       {sweeps.get_sweep_url()}')
    print(f'Sweep Group URL: {sweeps.notes}\n')
    '''


if __name__ == "__main__":
    wandb.agent(wandb.sweep(sweep_config), function=main)

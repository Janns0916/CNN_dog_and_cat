# encoding:utf-8
import copy
import math
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
import transformers
import wandb
from torch.autograd import Variable
import numpy as np
from CNN_Dataset import load_cnn_data
import argparse
from loguru import logger
import sys
import time
from CNN_Model import CNNet
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='CNN/model', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', default=530, type=int)
    parser.add_argument('--fp16', action='store_true')
    # wandb
    parser.add_argument("--use_wandb", type=bool, default=False, help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = vars(args)
    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # if args.output_dir is not None:
    #     os.makedirs(args.output_dir, exist_ok=True)

    # 模型的加载
    modules = [CNNet().to(device)]
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 数据集的加载
    train_dataloader, valid_dataloader, test_dataloader = load_cnn_data()

    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    criterion = nn.CrossEntropyLoss()
    model = CNNet().to(device)
    # train(args, train_dataloader, valid_dataloader, optimizer, model, criterion, device)
    test(test_dataloader, r'E:\Jann\Algorithm\Basic_Network_Architecture\CNN\model\cnn.pkl', model, device)


def train(args, train_dataloader, valid_dataloader, optimizer, model, criterion, device):
    min_val_loss = 5
    min_epoch = 5
    best_model = None
    for epoch in tqdm(range(args.num_train_epochs), ascii=True):
        train_loss = []
        for batch_idx, (data, target) in enumerate(train_dataloader, 0):
            data, target = Variable(data).to(device), Variable(target).to(device)
            optimizer.zero_grad()
            output = model(data)
            training_loss = criterion(output, target)
            training_loss.backward()
            train_loss.append(training_loss.cpu().item())

        # validation
        val_loss = valid(valid_dataloader, model, criterion, device)

        model.train()
        if epoch + 1 > min_epoch and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))

    torch.save(best_model.state_dict(), "model/cnn.pkl")


def valid(valid_dataloader, model, criterion, device):
    model.eval()
    valid_loss = []
    for batch_idx, (data, target) in enumerate(valid_dataloader):
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        validing_loss = criterion(output, target)
        valid_loss.append(validing_loss.cpu().item())
    return np.mean(valid_loss)


def test(test_dataloader, path, model, device):
    total = 0
    current = 0
    model.load_state_dict(torch.load(path), False)
    for (data, target) in test_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        predict = torch.max(output.data, 1)[1].data
        total += target.size(0)
        current += (predict == target).sum()
    print('\n')
    print('Accuracy:%d%%' % (100 * current / total))


if __name__ == '__main__':
    main()
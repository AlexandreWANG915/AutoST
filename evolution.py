import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from lib.datasets import build_dataset
from lib import utils
from model.AutoST import AutoST
import argparse
import os
import yaml
from lib.config import cfg, update_config_from_file
from lib.training_free import *
from timm.utils.model import unwrap_model
import copy
from timm.utils import accuracy
import wandb
import os
import yaml
from collections import deque
from lib.flops import count_flops
import csv
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
from timm.models import create_model
from syops import get_model_complexity_info
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.utils.data
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class OnesDataset(Dataset):
    def __init__(self, num_samples, img_shape):
        self.num_samples = num_samples
        self.img_shape = img_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.ones(self.img_shape, dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)  # You can set this to any fixed value
        return img, label


def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth + 1]), list(cand_tuple[depth + 1: 2 * depth + 1]), cand_tuple[-1]


class EvolutionSearcher(object):

    def __init__(self, args, device, choices, val_loader, test_loader, output_dir):
        self.device = device
        self.indicator_name = args.indicator_name
        self.args = args
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.s_prob = args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices
        self.top = {}
        self.all_res = []
        self.ld = args.ld

    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            print('visited')
            return False
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
        # print('depth', depth, 'mlp_ratio', mlp_ratio, 'num_heads', num_heads, 'embed_dim', embed_dim)
        model = create_model(
            'AutoST',
            pretrained=False,
            drop_rate=0.,
            drop_path_rate=0.,
            drop_block_rate=None,
            img_size_h=32, img_size_w=32,
            patch_size=4, embed_dims=embed_dim, num_heads=num_heads, mlp_ratios=mlp_ratio,
            in_channels=3, num_classes=100, qkv_bias=False,
            depths=depth, sr_ratios=1,
            T=4
        )
        model.to(self.device)
        model_without_ddp = copy.deepcopy(model)
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

        info['params'] = n_parameters / 10 ** 6

        if info['params'] > self.parameters_limits:
            print('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            print('under minimum parameters limit')
            return False

        # Define the dataset and DataLoader
        num_samples = 32
        img_shape = (3, 32, 32)
        batch_size = 32

        ones_dataset = OnesDataset(num_samples, img_shape)
        dataloader = DataLoader(ones_dataset, batch_size=batch_size, drop_last=True, num_workers=8, pin_memory=True)

        ops, n_parameters = get_model_complexity_info(model_without_ddp, (3, 32, 32), dataloader, as_strings=False,
                                                      print_per_layer_stat=False, verbose=False)

        power = compute_indicators.find_indicators(model_without_ddp,
                                                   self.val_loader,
                                                   ('random', 1, 1000),
                                                   self.device)
        print(power)
        # print(n_parameters / 10**6)
        n_parameters = n_parameters / 10 ** 6
        flops = ops[0] / 10 ** 9
        print("rank:", utils.get_rank(), cand, info['params'])

        indicators = {self.indicator_name: flops + self.ld/ power['NASWOT']}
        print(indicators)
        del model
        del model_without_ddp
        if self.top == {}:
            self.top['cand'] = cand
            self.top[self.indicator_name] = indicators[self.indicator_name]
        else:
            if self.top[self.indicator_name] < indicators[self.indicator_name]:
                self.top['cand'] = cand
                self.top[self.indicator_name] = indicators[self.indicator_name]
        res = {'config': cand}
        res.update({'params': n_parameters})
        res.update({'flops': flops})
        res.update({'indicator': indicators})

        self.all_res.append(res)
        print(len(self.all_res))
        print(res)

        info['visited'] = True
        info.update(indicators)
        info['params'] = n_parameters
        info['flops'] = flops
        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        cand_tuple = list()
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(self.choices['depth'])
        cand_tuple.append(depth)
        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.choices[dimension]))

        cand_tuple.append(random.choice(self.choices['embed_dim']))
        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            random_s = random.random()

            # depth
            if random_s < s_prob:
                new_depth = random.choice(self.choices['depth'])

                if new_depth > depth:
                    mlp_ratio = mlp_ratio + [random.choice(self.choices['mlp_ratio']) for _ in range(new_depth - depth)]
                    num_heads = num_heads + [random.choice(self.choices['num_heads']) for _ in range(new_depth - depth)]
                else:
                    mlp_ratio = mlp_ratio[:new_depth]
                    num_heads = num_heads[:new_depth]

                depth = new_depth
            # mlp_ratio
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    mlp_ratio[i] = random.choice(self.choices['mlp_ratio'])

            # num_heads

            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    num_heads[i] = random.choice(self.choices['num_heads'])

            # embed_dim
            random_s = random.random()
            if random_s < s_prob:
                embed_dim = random.choice(self.choices['embed_dim'])

            result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]

            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            # Update top_k selection criteria
            print(self.vis_dict)
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x][self.indicator_name])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.select_num])))
            tmp_indicator_values = []
            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                print('No.{} {} Indicator = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand][self.indicator_name], self.vis_dict[cand]['params']))
                tmp_indicator_values.append(self.vis_dict[cand][self.indicator_name])
            self.top_accuracies.append(tmp_indicator_values)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            # self.save_checkpoint()
        # Find top 100 candidates
        sorted_candidates = sorted(self.all_res, key=lambda x: x["indicator"][self.indicator_name], reverse=True)
        print(sorted_candidates[0])

        num_candidates_to_save = 100
        interval = len(sorted_candidates) // num_candidates_to_save

        output_dir = "Cifar10_evolution_small"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for index in tqdm(range(len(sorted_candidates))):
            # for index in range(0, len(sorted_candidates), interval):
            candidate = sorted_candidates[index]['config']
            indicator_value = sorted_candidates[index]['indicator'][self.indicator_name]
            params = sorted_candidates[index]['params']
            flops = sorted_candidates[index]['flops']

            file_name = os.path.join(output_dir, f"config_{index + 1}.yaml")
            self.save_candidate_to_yaml(candidate, indicator_value, params, flops, ACs, MACs, Power, file_name)
        print('Searched Architecture: ', self.top['cand'])

    def save_candidate_to_yaml(self, candidate, indicator, params, flops, ACs, MACs, Power, filename):
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(candidate)
        data = {
            "RETRAIN": {
                "MLP_RATIO": mlp_ratio,
                "NUM_HEADS": num_heads,
                "DEPTH": depth,
                "EMBED_DIM": embed_dim,
            },
            "INDICATOR": float(indicator),
            "PARAMETER": float(params),
            "FLOPS": float(flops),
        }
        merged_data = {}
        merged_data.update(data)

        with open(filename, 'w') as outfile:
            yaml.dump(merged_data, outfile)


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # evolution search parameters
    parser.add_argument('--indicator-name', default='AutoST', type=str)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=12)
    parser.add_argument('--min-param-limits', type=float, default=11)
    parser.add_argument('--ld', type=float, default=0.)

    # config file
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14,
                        help='max distance in relative position embedding')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # custom model argument
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='CIFAR100', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'EVO_IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    return parser


def main(args):
    update_config_from_file(args.cfg)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args, folder_name="subImageNet")
    dataset_test, _ = build_dataset(is_train=False, args=args, folder_name="val")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(2 * args.batch_size),
        sampler=sampler_test, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    print(cfg)

    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM, 'depth': cfg.SEARCH_SPACE.DEPTH}

    t = time.time()
    searcher = EvolutionSearcher(args, device, choices, data_loader_val, data_loader_test, args.output_dir)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoST evolution search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

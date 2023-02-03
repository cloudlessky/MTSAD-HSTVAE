
import pandas as pd
import numpy as np
import torch

import sys
sys.path.insert(0,"")
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.preprocessing import MinMaxScaler
from util.utils import *
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data, construct_data_SMD
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep
from TimeDataset import TimeDataset
import HSTVAE
import torch.nn as nn
from train import *
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
import json
import random
import datetime

print(torch.cuda.is_available())
print(torch.cuda.device_count())
torch.cuda.current_device()
torch.cuda._initialized = True
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(linewidth=50)

def dist(a, b):
    return np.sqrt(sum((a - b) ** 2))

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        NUM_TASKS = self.train_config['num_tasks']
        LATENT_CODE_SIZE = self.train_config['LATENT_CODE_SIZE']
        T = self.train_config['slide_win']
        Graph_learner_n_hid = self.train_config['Graph_learner_n_hid']
        Graph_learner_n_head_dim = self.train_config['Graph_learner_n_head_dim']
        Graph_learner_head = self.train_config['Graph_learner_head']
        do_prob = self.train_config['do_prob']
        dec_in = self.train_config['dec_in']
        d_model = self.train_config['dim']
        alpha = self.train_config['alpha']
        beta = self.train_config['beta']
        dataset = self.env_config['dataset']
        if dataset == 'SMD':
            group_index = self.train_config['group_index']
            index = self.train_config['index']
            (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=True)
            train = torch.from_numpy(x_train).float()
            test = torch.from_numpy(x_test).float()
            labels = y_test if y_test is not None else None
            labels_list = labels.tolist()
            NUM_TASKS = train.shape[1]
            set_device(env_config['device'])
            self.device = get_device()
            train_dataset_indata = construct_data_SMD(train, labels=0)
            test_dataset_indata = construct_data_SMD(test, labels=labels_list)
            label1 = labels_list
        else:
            train = pd.read_csv(f'../data/{dataset}/train.csv', sep=',', index_col=0)
            test = pd.read_csv(f'../data/{dataset}/test.csv', sep=',', index_col=0)
            print('train.shape',train.shape)
            print('test.shape',test.shape)
            if 'attack' in train.columns:
                train = train.drop(columns=['attack'])
            print('train.shape=', train.shape)
            if 'P603' in train.columns:
                train = train.drop(columns=['P603'])
            print('train.shape=', train.shape)
            if 'P603' in test.columns:
                test = test.drop(columns=['P603'])
            print('test.shape=', test.shape)
            set_device(env_config['device'])
            self.device = get_device()
            train_dataset_indata = construct_data(train, labels=0)
            test_dataset_indata = construct_data(test, labels=test.attack.tolist())
        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, mode='test', config=cfg)
        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                          shuffle=False, num_workers=0)
        batch_num = 0
        for x, labels, attack_labels in train_dataloader:
            if batch_num == 0:
                print(x.shape, labels.shape, attack_labels.shape)
            batch_num += 1
        batch_num = 0
        for x, labels, attack_labels in self.test_dataloader:
            if batch_num == 0:
                print(x.shape, labels.shape, attack_labels.shape)
            batch_num += 1
        enc_spatio = HSTVAE.Encoder_Spatio(T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head,
                      do_prob, device=self.device)
        enc_temporal = HSTVAE.Encoder_Temporal(NUM_TASKS, LATENT_CODE_SIZE, factor=5, d_model=d_model, n_heads=8, e_layers=1, d_ff=d_model,
                 dropout=0, attn='full', activation='gelu',output_attention=False, distil=True,device=self.device)
        dec_rec = HSTVAE.Decoder_Rec(NUM_TASKS, LATENT_CODE_SIZE, factor=5, d_model=d_model, n_heads=8, d_layers=1, d_ff=d_model,
                dropout=0, attn='full',activation='gelu',mix=True, c_out=1, device=self.device)
        dec_pre = HSTVAE.Decoder_Pre(NUM_TASKS, LATENT_CODE_SIZE,factor=5, d_model=d_model, n_heads=8, d_layers=1, d_ff=d_model,
                 dropout=0, attn='full',activation='gelu', mix=True, c_out=1, device=self.device)
        sampling_z = HSTVAE.Sampling_Temporal(NUM_TASKS, LATENT_CODE_SIZE, factor=5, d_model=d_model)
        gat = HSTVAE.GAT(input_size=train_config['dim'] * train_config['slide_win'],
                        hidden_size=train_config['dim'] * train_config['slide_win'],output_size=train_config['dim'] * train_config['slide_win'],
                        num_of_task=NUM_TASKS,dropout=0,nheads=1,alpha=0.2)
        self.model = HSTVAE.MTVAE(enc_spatio, enc_temporal, sampling_z, gat, dec_rec, dec_pre, T, dec_in, d_model=d_model,topk_indices_ji=None)
        self.model.apply(init_weights)
        self.model.to(self.device)

    def run(self):
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
            print('model_save_path:', model_save_path)
        else:
            model_save_path = self.get_save_path()[0]
            print('model_save_path:', model_save_path)

            self.train_log = train(self.model,self.device,model_save_path,
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                group_index=self.train_config['group_index'],
                index=self.train_config['index'],
            )

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)
        _, self.test_result,self.test_neighbor_result,self.degree_anamoly_label = test(best_model, self.test_dataloader, train_config, self.device)
        self.get_score(self.test_result,self.test_neighbor_result,self.degree_anamoly_label,self.train_config['alpha'],self.train_config['beta'])

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.2):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)
        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)
        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)
        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                      shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                    shuffle=False)
        return train_dataloader, val_dataloader

    def get_score(self, test_result,test_neighbor_result,degree_anamoly_label,alpha,beta):
        np_test_result = np.array(test_result)
        test_labels = np_test_result[2, :, 0].tolist()
        test_scores,test_scores_abs,test_future_scores,test_history_scores = get_full_err_scores(test_result,test_neighbor_result)
        top1_best_info = get_best_performance_data(test_result,test_scores,test_scores_abs,test_future_scores,test_history_scores, test_labels,degree_anamoly_label,
                                                   topk=50,alpha=alpha,beta=beta,config=eval_config,group_index=1,index=1)

    def get_save_path(self, feature_name=''):
        dir_path = self.env_config['save_path']
        group_index = self.train_config['group_index']
        index = self.train_config['index']
        dataset = self.train_config['dataset']
        if self.datestr is None:
            now = datetime.datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr
        paths = [
            f'./pretrained/{dir_path}/best_{datestr}-data{dataset}-group{group_index}-index{index}.pt',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return paths


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-epoch', help='train epoch', type=int, default=50)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=40)
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('-num_tasks', help='num_tasks', type=int, default=50)
    parser.add_argument('-LATENT_CODE_SIZE', help='LATENT_CODE_SIZE', type=int, default=8)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument('-down', help='down', type=int, default=10)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=1)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='swat')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=64)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.2)
    parser.add_argument('-topk', help='topk', type=int, default=49)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-n_hid', help='n_hid', type=int, default=64)
    parser.add_argument('-do_prob', help='do_prob', type=int, default=0)
    parser.add_argument('-Graph_learner_n_hid', help='Graph_learner_n_hid', type=int, default=64)
    parser.add_argument('-Graph_learner_n_head_dim', help='Graph_learner_n_head_dim', type=int, default=32)
    parser.add_argument('-Graph_learner_head', help='Graph_learner_head', type=int, default=1)
    parser.add_argument('-prior', help='prior', type=np.array, default=np.array([0.1]))
    parser.add_argument('-temperature', help='temperature', type=int, default=0.5)
    parser.add_argument('-GRU_n_dim', help='GRU_n_dim', type=int, default=64)
    parser.add_argument('-max_diffusion_step', help='max_diffusion_step', type=int, default=2)
    parser.add_argument('-num_rnn_layers', help='num_rnn_layers', type=int, default=1)
    parser.add_argument('-dec_in', help='dec_in', type=int, default=1)
    parser.add_argument('-d_model', help='d_model', type=int, default=64)
    parser.add_argument('-alpha', help='alpha', type=float, default=0.74)
    parser.add_argument('-beta', help='beta', type=float, default=0.08)
    parser.add_argument('-group_index', help='group_index', type=int, default=1)
    parser.add_argument('-index', help='index', type=int, default=1)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'dataset': args.dataset,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'num_tasks': args.num_tasks,
        'LATENT_CODE_SIZE': args.LATENT_CODE_SIZE,
        'n_hid': args.n_hid,
        'do_prob': args.do_prob,
        'Graph_learner_n_hid': args.Graph_learner_n_hid,
        'Graph_learner_n_head_dim': args.Graph_learner_n_head_dim,
        'Graph_learner_head': args.Graph_learner_head,
        'prior': args.prior,
        'temperature': args.temperature,
        'GRU_n_dim': args.GRU_n_dim,
        'max_diffusion_step': args.max_diffusion_step,
        'num_rnn_layers': args.num_rnn_layers,
        'dec_in':args.dec_in,
        'd_model':args.d_model,
        'alpha': args.alpha,
        'beta': args.beta,
        'down': args.down,
        'group_index': args.group_index,
        'index': args.index,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'group_index': args.group_index,
        'index': args.index,
    }

    eval_config = {
        'slide_win': args.slide_win,
        'down': args.down,
        'num_tasks': args.num_tasks,
    }

    main = Main(train_config, env_config, debug=False)
    main.run()
    endtime = datetime.datetime.now()
    t = (endtime - starttime).seconds







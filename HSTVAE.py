import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
import random
from lib.utils import *
import matplotlib.pyplot as plt
from util.masking import TriangularCausalMask, ProbMask
from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, Sampling, SampLayer,RecLinear,LinearLayer
from decoder import DecoderR, DecoderP, DecoderRecLayer, DecoderPreLayer
from attn import FullAttention, ProbAttention, AttentionLayer_Rec, AttentionLayer_Pre_Cross, AttentionLayer
from embed import DataEmbedding
from sklearn import manifold

SEED = 2020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def calculate_kl_loss(z_mean, z_log_sigma):
    temp = 1.0 + 2 * z_log_sigma - z_mean ** 2 - torch.exp(2 * z_log_sigma)
    return -0.5 * torch.sum(temp, -1)

criterion = nn.MSELoss()

class Enc_linear(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(Enc_linear, self).__init__()
        super(Enc_linear, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.sigmoid(self.fc1(inputs))
        x = F.sigmoid(self.fc2(x))
        return self.batch_norm(x)

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class MLP2(nn.Module):
    def __init__(self, n_in, n_out, do_prob=0.):
        super(MLP2, self).__init__()
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = self.fc1(inputs)
        return x

def norm(t):
    return t / t.norm(dim=1, keepdim=True)

def cos_sim(v1, v2):
    v1 = norm(v1)
    v2 = norm(v2)
    return v1 @ v2.t()

class Encoder_Spatio(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, do_prob=0., device=None):
        super(Encoder_Spatio, self).__init__()
        self.n_hid = n_hid
        self.head = head
        self.n_in = n_in
        self.n_head_dim = n_head_dim
        self.device = device
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.Wq = nn.Linear(n_hid, n_head_dim * head)
        self.Wk = nn.Linear(n_hid, n_head_dim * head)
        self.n_hid2 = n_head_dim
        self.mlp2 = MLP2(n_head_dim*2, 1, do_prob)
        for m in [self.Wq, self.Wk]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):
        X = self.mlp1(inputs)
        Xq = self.Wq(X)
        Xk = self.Wk(X)
        B, N, n_hid = Xq.shape
        Xq = Xq.view(B, N, self.head, self.n_head_dim)
        Xk = Xk.view(B, N, self.head, self.n_head_dim)
        Xq = Xq.permute(0, 2, 1, 3).squeeze()
        Xk = Xk.permute(0, 2, 1, 3).squeeze()
        XQ = Xq.unsqueeze(2)
        XQ = XQ.expand(-1,-1,Xk.shape[1],-1)
        NQ = XQ.reshape(Xk.shape[0],Xk.shape[1]*Xk.shape[1],Xk.shape[2])
        NK = Xk.repeat(1,Xk.shape[1],1)

        NN_qk = torch.cat((NK,NQ),-1)
        probs = self.mlp2(NN_qk)
        probs = probs.reshape(Xk.shape[0],Xk.shape[1],Xk.shape[1])
        return probs

class Encoder_Temporal(nn.Module):
    def __init__(self, num_tasks, latent_dim, factor=5, d_model=64, n_heads=8, e_layers=1, d_ff=64,
                 dropout=0.0, attn='full', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(Encoder_Temporal, self).__init__()
        self.attn = attn
        self.output_attention = output_attention
        self.Nz = latent_dim
        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder_temporal = nn.ModuleList([Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for i in range(num_tasks)])

    def forward(self, num_of_task, inp_enc, enc_self_mask=None):
        enc_out,_ = self.encoder_temporal[num_of_task](inp_enc, enc_self_mask)
        return enc_out

class GATLayer(nn.Module):
    def __init__(self, input_feature, output_feature, num_of_task, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(num_of_task, 2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
        self.beta = 0.3

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj,batch_i):
        N, C = h.size()
        Wh = torch.matmul(h, self.w)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, C), Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.output_feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.mm(attention, Wh)
        out = F.elu(h_prime + self.beta * Wh)
        return out

class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_of_task, dropout, alpha, nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, num_of_task, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * nheads, output_size, num_of_task, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj,batch_i):
        x = torch.cat([att(x, adj,batch_i) for att in self.attention], dim=1)
        return x

class Sampling_Temporal(nn.Module):
    def __init__(self, num_tasks, latent_dim, factor=5, d_model=64,
                 device=torch.device('cuda:0')):
        super(Sampling_Temporal, self).__init__()
        self.Nz = latent_dim
        self.win_len = 40
        self.mu = nn.ModuleList([Enc_linear(d_model, int(d_model/2), self.Nz) for i in range(num_tasks)])
        self.sigma = nn.ModuleList([Enc_linear(d_model, int(d_model/2), self.Nz) for i in range(num_tasks)])

    def forward(self, num_of_task, enc_out):
        mu = self.mu[num_of_task](enc_out).cuda()
        sigma_hat = self.sigma[num_of_task](enc_out).cuda()
        sigma = sigma_hat.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(sigma.size()).normal_()
        z = eps.mul(sigma).add_(mu)
        return z, mu, sigma_hat

class Decoder_Rec(nn.Module):
    def __init__(self, num_tasks, latent_dim, factor=5, d_model=64, n_heads=8, d_layers=1, d_ff=64,
                dropout=0.0, attn='full',activation='gelu',mix=True, c_out=1,
                device=torch.device('cuda:0')):
        super(Decoder_Rec, self).__init__()

        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.decoder_rec = nn.ModuleList([DecoderR(
            [
                DecoderRecLayer(
                    AttentionLayer_Rec(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, latent_dim, n_heads, mix=mix),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for i in range(num_tasks)])

        self.fc_out = nn.ModuleList([nn.Linear(d_model, c_out, bias=True) for i in range(num_tasks)])

    def forward(self, num_of_task, enc_out, dec_self_mask=None):
        dec_rec_out = self.decoder_rec[num_of_task](enc_out, x_mask=dec_self_mask)
        recon_out = self.fc_out[num_of_task](dec_rec_out)
        return recon_out

class Decoder_Pre(nn.Module):
    def __init__(self, num_tasks, latent_dim, factor=5, d_model=64, n_heads=8, d_layers=1, d_ff=64,
                 dropout=0.0, attn='full',activation='gelu', mix=True, c_out=1,
                 device=torch.device('cuda:0')):
        super(Decoder_Pre, self).__init__()
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.decoder_pre = nn.ModuleList([DecoderP(
            [
                DecoderPreLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer_Pre_Cross(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, latent_dim, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for i in range(num_tasks)])

        self.fc_out = nn.ModuleList([nn.Linear(d_model, c_out, bias=True) for i in range(num_tasks)])

    def forward(self, num_of_task, dec_out, enc_out, dec_self_mask=None, dec_enc_mask=None):
        dec_rec_out = self.decoder_pre[num_of_task](dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        pred_out = self.fc_out[num_of_task](dec_rec_out)
        return pred_out

class MTVAE(nn.Module):
    def __init__(self, enc_spatio,enc_temporal,sampling_z, GAT, dec_rec, dec_pre, enc_in, dec_in, d_model=64, topk_indices_ji=None, dropout=0.0, embed='fixed', freq='t'):
        super(MTVAE, self).__init__()
        self.enc_spatio = enc_spatio
        self.enc_temporal = enc_temporal
        self.dec_rec = dec_rec
        self.dec_pre = dec_pre
        self.sampling_z = sampling_z
        self.topk_indices_ji = topk_indices_ji
        self.GAT = GAT
        self.num_nodes = 50
        self.head = 1
        self.temperature = 0.01
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

    def forward(self, all_input, all_target, batch_i, config, epoch, device):
        losses = 0
        loss_pred = 0

        NUM_OF_TASK = all_input.shape[1]
        batch_size = all_input.shape[0]
        all_input = all_input.to(device).float()
        all_target = all_target.to(device).float()
        probs = self.enc_spatio(all_input).to(device)
        probs = probs.reshape(probs.shape[0], probs.shape[1] * probs.shape[1], 1)
        full_1 = torch.ones(probs.shape).to(device)
        probs_0 = full_1 - probs
        prob_cat = torch.cat((probs,probs_0),2).to(device)
        prob = F.softmax(prob_cat, -1)
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True).to(device)
        adj=edges[:,:,0].reshape(probs.shape[0], all_input.shape[1], all_input.shape[1])

        log_prior = torch.FloatTensor(np.log(config['prior']))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)
        log_prior = log_prior.to(device)
        for num_of_task in range(NUM_OF_TASK):
            src = all_input[:, num_of_task, :]
            src = src.unsqueeze(2).to(device)
            src = src.float()
            enc_inp = self.enc_embedding(src)
            enc_out = self.enc_temporal(num_of_task, enc_inp, enc_self_mask=None)
            enc_out = enc_out.unsqueeze(1)
            if num_of_task == 0:
                enc_out_cat = enc_out
            else:
                enc_out_cat = torch.cat((enc_out_cat, enc_out), dim=1)

        enc_out_cat1 = enc_out_cat.reshape(enc_out_cat.shape[0], enc_out_cat.shape[1],
                                           enc_out_cat.shape[2] * enc_out_cat.shape[3]).to(device)
        enc_out_update = torch.zeros_like(enc_out_cat1)
        for i in range(enc_out_cat1.shape[0]):
            enc_out_update[i, :] = self.GAT(enc_out_cat1[i, :], adj[i, :],batch_i).float().to(device)
        enc_out_update = enc_out_update.reshape(enc_out_cat.shape[0], enc_out_cat.shape[1],
                                                enc_out_cat.shape[2], enc_out_cat.shape[3]).to(device)
        each_seq_loss_list = []
        for num_of_task in range(NUM_OF_TASK):
            src = all_input[:, num_of_task, :]
            trg_now = all_target[:, num_of_task,0]
            trg_future = all_target[:, num_of_task, 1:]
            src = src.to(device)

            enc_out1 = enc_out_update[:, num_of_task, :]
            z,mu,sigma = self.sampling_z(num_of_task, enc_out1)
            rec_output = self.dec_rec(num_of_task, z, dec_self_mask=None)

            kl_loss1 = calculate_kl_loss(mu.float(), sigma.float())
            kl_loss2 = kl_loss1.mean(dim=-1)
            kl_loss = kl_loss2.mean(dim=-1)
            rec_output = rec_output.squeeze()
            rec_loss = criterion(rec_output.float(), src.float())
            recon_loss = 1.0 * rec_loss

            batch_size = src.shape[0]
            token = torch.rand(batch_size, 1,1).cuda()
            dec_pre_inp = torch.zeros_like(trg_future.unsqueeze(2))
            dec_inp_init = torch.cat([token, dec_pre_inp], dim=1)
            dec_inp = self.dec_embedding(dec_inp_init)
            pred_output = self.dec_pre(num_of_task, dec_inp, z, dec_self_mask=None, dec_enc_mask=None)

            pred_now_loss = criterion(pred_output[:, 0, :].float(), trg_now.unsqueeze(1).float())
            pred_loss = pred_now_loss
            pred_future_loss = criterion(pred_output[:,1:,:].squeeze().float(), trg_future.float())
            kl_coeff = 0.001
            loss = recon_loss + pred_now_loss + pred_future_loss + kl_coeff * kl_loss
            losses += loss
            loss_pred += pred_loss + kl_coeff * kl_loss

            each_seq_loss_list.append(loss.item())
            rec_output_return = rec_output.unsqueeze(1)
            pred_output_return = pred_output.squeeze(2).unsqueeze(1)
            if num_of_task == 0:
                rec_out_cat = rec_output_return
                pre_out_cat = pred_output_return
            else:
                rec_out_cat = torch.cat((rec_out_cat, rec_output_return), dim=1)
                pre_out_cat = torch.cat((pre_out_cat, pred_output_return), dim=1)

        spatio_kl_loss = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(device)
        spatio_error = spatio_kl_loss.repeat(all_input.shape[0])
        losses += spatio_kl_loss
        return rec_out_cat, pre_out_cat, losses, prob, spatio_error, each_seq_loss_list



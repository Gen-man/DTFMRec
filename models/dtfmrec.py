# coding: utf-8
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.transformer import TransformerEncoder, TransformerEncoderLayer
from utils.crossAttention import CrossAttention


class DTFMRec(GeneralRecommender):
    def __init__(self, config, dataset, latent_dim=64, nhead=1, transformer_layers=1):
        super(DTFMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.n_mm_layer = config['n_mm_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_nodes = self.n_users + self.n_items
        self.src_len = 10
        self.reg_weight = config['reg_weight']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.cosine = nn.CosineEmbeddingLoss()
        self.users, self.user_item, self.mask = self.load_interaction_matrix(config)
        self.user_exp = nn.Parameter(torch.rand(self.n_users, latent_dim))
        self.mm_image_weight = config['mm_image_weight']
        self.knn_k = 10
        v_len, t_len = 1, 1

        if self.v_feat is not None and self.t_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.v_linear = nn.Linear(self.v_feat.size(1), latent_dim)
            self.t_linear = nn.Linear(self.t_feat.size(1), latent_dim)

            self.sh_mlp = nn.Linear(latent_dim, latent_dim)
            self.sh_encoder_layer = TransformerEncoderLayer(dim=latent_dim, nhead=nhead)
            self.sh_encoder = TransformerEncoder(encoder_layer=self.sh_encoder_layer, num_layers=transformer_layers)
            self.sh_dense = nn.Linear(latent_dim, latent_dim)

            self.sh_proj1 = nn.Linear(latent_dim * 2, latent_dim * 2)
            self.sh_proj2 = nn.Linear(latent_dim * 2, latent_dim)

            self.proj1_v = nn.Linear(latent_dim * 2, latent_dim * 2)
            self.proj2_v = nn.Linear(latent_dim * 2, latent_dim)
            self.proj1_t = nn.Linear(latent_dim * 3, latent_dim * 3)
            self.proj2_t = nn.Linear(latent_dim * 3, latent_dim)

            self.proj1_t_high = nn.Linear(latent_dim, latent_dim)
            self.proj2_t_high = nn.Linear(latent_dim, latent_dim)
            self.proj1_v_high = nn.Linear(latent_dim, latent_dim)
            self.proj2_v_high = nn.Linear(latent_dim, latent_dim)

        if self.v_feat is not None:
            self.v_encoder_layer = TransformerEncoderLayer(dim=latent_dim, nhead=nhead)
            self.v_encoder = TransformerEncoder(self.v_encoder_layer, num_layers=transformer_layers)
            self.v_dense = nn.Linear(latent_dim, latent_dim)

            self.v_decoder = nn.Conv1d(v_len * 2, v_len, kernel_size=1, padding=0, bias=False)  # 特征重建
            self.v_cosine = nn.Linear(latent_dim * (v_len - 1 + 1), latent_dim)
            self.v_align = nn.Linear(latent_dim * (t_len - 1 + 1), latent_dim)
            self.v_cross_att = CrossAttention(latent_dim)

        if self.t_feat is not None:
            self.t_encoder_layer = TransformerEncoderLayer(dim=latent_dim, nhead=nhead)
            self.t_encoder = TransformerEncoder(self.t_encoder_layer, num_layers=transformer_layers)
            self.t_dense = nn.Linear(latent_dim, latent_dim)

            self.t_decoder = nn.Conv1d(t_len * 2, t_len, kernel_size=1, padding=0, bias=False)
            self.t_cosine = nn.Linear(latent_dim * (t_len - 1 + 1), latent_dim)
            self.t_align = nn.Linear(latent_dim * (t_len - 1 + 1), latent_dim)
            self.t_cross_att = CrossAttention(latent_dim)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path,
                                   'mm_adj_{}_{}原始.pt'.format(self.knn_k, int(10 * self.mm_image_weight)))
        self.mm_adj = None
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device, weights_only=False)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

    def load_interaction_matrix(self, config):
        df = pd.read_csv(r'./data/{}/{}.inter'.format(config['dataset'], config['dataset']), delimiter='\t')
        users = df['userID'].to_list()
        items = df['itemID'].to_list()
        user_item_dict = {}

        for user, item in zip(users, items):
            user_id = user
            item_id = item
            if user_id not in user_item_dict:
                user_item_dict[user_id] = []
            user_item_dict[user_id].append(item_id)

        user_item_list = []
        mask_list = []
        for i in user_item_dict.keys():
            temp = user_item_dict[i]
            if len(temp) > self.src_len:
                mask = torch.ones(self.src_len + 1) == 0
                temp = temp[:self.src_len]
            else:
                mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
                temp.extend([0 for i in range(self.src_len - len(temp))])
            user_item = torch.cat([torch.tensor([-1])] + [torch.tensor(temp)])
            user_item_list.append(user_item)
            mask_list.append(mask)

        return list(user_item_dict.keys()), torch.stack(user_item_list), torch.stack(mask_list).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))

        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return torch.tensor(sumArr).to(self.device), self.scipy_matrix_to_sparse_tenser(L, torch.Size(
            (self.n_nodes, self.n_nodes)))

    def cge(self, ):
        h = self.item_id_embedding.weight
        h_all = [h]
        for i in range(self.n_mm_layer):
            h = torch.sparse.mm(self.mm_adj, h)
            h_all.append(h)
        h = torch.mean(torch.stack(h_all, dim=0), dim=0)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        cge_embs = [ego_embeddings]
        cge_embs = torch.stack(cge_embs, dim=1).mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(cge_embs, [self.n_users, self.n_items], dim=0)
        i_g_embeddings = i_g_embeddings + self.item_id_embedding.weight + h
        cge_embs_h = torch.cat((u_g_embeddings, i_g_embeddings), dim=0)

        return cge_embs, cge_embs_h

    def mge(self):
        h_v, h_t = self.image_embedding.weight, self.text_embedding.weight
        h_v_all, h_t_all = [h_v], [h_t]
        for i in range(self.n_mm_layer):
            h_v = torch.sparse.mm(self.mm_adj, h_v)
            h_t = torch.sparse.mm(self.mm_adj, h_t)
            h_v_all.append(h_v)
            h_t_all.append(h_t)

        h_v, h_t = torch.mean(torch.stack(h_v_all, dim=0), dim=0), torch.mean(torch.stack(h_t_all, dim=0), dim=0)

        v, t, v_out, t_out = None, None, None, None
        v_sp, v_sh, t_sp, t_sh = None, None, None, None
        v_sim, v_sim_recon, t_sim, t_sim_recon = None, None, None, None
        v_recon, t_recon = None, None

        if self.v_feat is not None:
            v = self.v_linear(self.image_embedding.weight + h_v)
            v_in = v[self.user_item]
            v_in[:, 0] = self.user_exp[self.users]

            v_sp = self.v_encoder(v_in.transpose(0, 1), key_padding_mask=self.mask).transpose(0, 1)[:, 0]
            v_sp = F.leaky_relu(self.v_dense(v_sp))

            v_sh = self.sh_encoder(v_in.transpose(0, 1), key_padding_mask=self.mask).transpose(0, 1)[:, 0]
            v_sh = F.leaky_relu(self.sh_dense(v_sh))

            v_sim = self.v_align(v_sh.contiguous().view(v_in.size(0), -1))  # 特征对齐
            v_recon = self.v_decoder(torch.cat([v_sp.unsqueeze(1), v_sh.unsqueeze(1)], dim=1)).view(v_in.size(0), -1)
            v_in_recon = v[self.user_item]
            v_in_recon[:, 0] = v_recon
            v_sim_recon = self.v_encoder(v_in_recon.transpose(0, 1), self.mask).transpose(0, 1)[:, 0]  # 特征编码

        if self.t_feat is not None:
            t = self.t_linear(self.text_embedding.weight + h_t)
            t_in = t[self.user_item]
            t_in[:, 0] = self.user_exp[self.users]

            t_sp = self.t_encoder(t_in.transpose(0, 1), key_padding_mask=self.mask).transpose(0, 1)[:, 0]
            t_sp = F.leaky_relu(self.t_dense(t_sp))

            t_sh = self.sh_encoder(t_in.transpose(0, 1), key_padding_mask=self.mask).transpose(0, 1)[:, 0]
            t_sh = F.leaky_relu(self.sh_dense(t_sh))

            t_sim = self.t_align(t_sh.contiguous().view(t_in.size(0), -1))
            t_recon = self.t_decoder(torch.cat([t_sp.unsqueeze(1), t_sh.unsqueeze(1)], dim=1)).view(t_in.size(0), -1)
            t_in_recon = t[self.user_item]
            t_in_recon[:, 0] = t_recon
            t_sim_recon = self.t_encoder(t_in_recon.transpose(0, 1), self.mask).transpose(0, 1)[:, 0]

        # MFA
        sh_v = self.v_cross_att(v_sh, v_sh, v_sh) + v_sh
        sh_t = self.t_cross_att(t_sh, t_sh, t_sh) + t_sh
        sh_fusion = torch.cat([sh_v, sh_t], dim=1)
        sh_proj = self.sh_proj2(F.leaky_relu(self.sh_proj1(sh_fusion)) + sh_fusion)

        t_t = self.v_cross_att(t_sp, t_sp, t_sp)
        v_t = self.v_cross_att(v_sp, t_sp, t_sp)
        t_v = self.t_cross_att(t_sp, v_sp, v_sp)
        v_t = self.proj2_v_high(F.leaky_relu(self.proj1_v_high(v_t))) + v_sp
        t_v = self.proj2_t_high(F.leaky_relu(self.proj1_t_high(t_v))) + t_sp

        v_out = torch.cat([sh_proj, v_t], dim=1)
        v_out = self.proj2_v(F.relu(self.proj1_v(v_out)) + v_out)

        t_out = torch.cat([sh_proj, t_t, t_v], dim=1)
        t_out = self.proj2_t(F.relu(self.proj1_t(t_out)) + t_out)

        return v, t, v_out, t_out, [v_sh, t_sh, v_sp, t_sp, v_recon, t_recon, v_sim, t_sim, v_sim_recon, t_sim_recon]

    def forward(self):
        lge_embs, mge_embs, dlf_emb = None, None, None
        cge_embs, cge_embs_h = self.cge()
        if self.v_feat is not None and self.t_feat is not None:
            v, t, v_feats, t_feats, dlf_emb = self.mge()

            v_feats = torch.concat([v_feats, v], dim=0)
            t_feats = torch.concat([t_feats, t], dim=0)
            mge_embs = F.normalize(v_feats) + F.normalize(t_feats)

            lge_embs = cge_embs_h + mge_embs

        u_embs, i_embs = torch.split(lge_embs, [self.n_users, self.n_items], dim=0)
        return u_embs, i_embs, dlf_emb

    def mse_loss(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def triplet_loss(self, S, P, N, margin=0.1):

        pos_sim = F.cosine_similarity(S, P, dim=1)
        neg_sim = F.cosine_similarity(S, N, dim=1)
        loss = torch.relu(margin - pos_sim + neg_sim).mean()

        return loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, dlf_emb = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        mf_v_loss, mf_t_loss = 0.0, 0.0
        text_feats = self.t_linear(self.text_embedding.weight)
        image_feats = self.v_linear(self.image_embedding.weight)
        if self.t_feat is not None:
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])

        # # 额外loss
        v_sh, t_sh, v_sp, t_sp, v_recon, t_recon, v_sim, t_sim, v_sim_recon, t_sim_recon = dlf_emb

        target = torch.tensor([-1]).to(self.device)
        v_cosine = self.cosine(v_sp, v_sh, target)
        t_cosine = self.cosine(t_sp, t_sh, target)
        loss_cosine = v_cosine + t_cosine

        v_sp_mse = self.mse_loss(v_sp, v_sim_recon)
        t_sp_mse = self.mse_loss(t_sp, t_sim_recon)
        loss_sp_mse = v_sp_mse + t_sp_mse

        v_recon_mse = self.mse_loss(v_recon, v_sp) + self.mse_loss(v_recon, v_sh)
        t_recon_mse = self.mse_loss(t_recon, t_sp) + self.mse_loss(t_recon, t_sh)
        loss_recon_mse = 0.5 * v_recon_mse + 0.5 * t_recon_mse

        v_sh_triplet, t_sh_triplet, v_sh_bpr, t_sh_bpr = None, None, None, None
        if self.v_feat is not None:
            u_g = v_sim[users]
            t_pos_i = text_feats[pos_items]
            v_neg_i = image_feats[neg_items]
            v_sh_triplet = self.triplet_loss(u_g, t_pos_i, v_neg_i)
        if self.t_feat is not None:
            u_g = t_sim[users]
            v_pos_i = image_feats[pos_items]
            t_neg_i = text_feats[neg_items]
            t_sh_triplet = self.triplet_loss(u_g, v_pos_i, t_neg_i)
        loss_sim = v_sh_triplet + t_sh_triplet

        w_o = nn.Parameter(torch.tensor(0.08))  # L_o
        w_m = nn.Parameter(torch.tensor(0.15))  # L_m
        w_s = nn.Parameter(torch.tensor(0.3))  # L_s
        w_r = nn.Parameter(torch.tensor(0.3))  # L_r
        w = nn.Parameter(torch.tensor(0.1))

        loss_dlf = w_o * loss_cosine + w_m * loss_sim + w_s * loss_sp_mse + w_r * loss_recon_mse
        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss + self.reg_weight * (mf_t_loss + mf_v_loss)
        loss += w * loss_dlf
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores

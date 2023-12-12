import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from scipy.sparse import coo_matrix
import scipy.sparse as sp


from attention import Attention
from domain_disentanglement import Domain_Disen



class P2M2_CDR(nn.Module):
    def __init__(self,**params):
        super(P2M2_CDR, self).__init__()
        self.user_num = params['num_users']
        self.source_item_num = params['source_num_items']
        self.target_item_num = params['target_num_items']
        self.embed_id_dim = params['embed_id_dim']
        self.domain_disen_dim = params['domain_disen_dim']
        self.device = params['device']
        self.text_embed_dim = params['text_embed_dim']
        self.visual_embed_dim = params['visual_embed_dim']
        self.review_embed_dim = params['review_embed_dim']
        self.source_ui_inter_lists_train = params['source_ui_inter_lists_train']
        self.target_ui_inter_lists_train = params['target_ui_inter_lists_train']
        self.source_text_feat = params['source_text_feat']
        self.source_visual_feat = params['source_visual_feat']
        self.source_review_feat = params['source_review_feat']
        self.target_text_feat = params['target_text_feat']
        self.target_visual_feat = params['target_visual_feat']
        self.target_review_feat = params['target_review_feat']
        self.n_layers = 3
        self.ssl_temp = 0.1
        self.alpha = 0.001
        self.beta = 0.001
        self.noise_control = 0.01
        self.sensitivity = 1.0
        self.epsilon = 1.0
        self.source_n_nodes = self.user_num + self.source_item_num
        self.target_n_nodes = self.user_num + self.target_item_num
        self.source_u_emb = nn.Embedding(params['num_users'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.source_u_emb.weight)
        self.target_u_emb = nn.Embedding(params['num_users'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.target_u_emb.weight)
        self.source_v_emb = nn.Embedding(params['source_num_items'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.source_v_emb.weight)
        self.target_v_emb = nn.Embedding(params['target_num_items'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.target_v_emb.weight)

        self.source_text_feat_emb = nn.Embedding(self.source_item_num, self.text_embed_dim).to(self.device)
        self.source_text_feat_emb.weight.data.copy_(self.source_text_feat)
        self.source_text_feat_emb.weight.requires_grad = False

        self.source_visual_feat_emb = nn.Embedding(self.source_item_num, self.visual_embed_dim).to(self.device)
        self.source_visual_feat_emb.weight.data.copy_(self.source_visual_feat)
        self.source_visual_feat_emb.weight.requires_grad = False

        self.source_review_feat_emb = nn.Embedding(self.user_num, self.review_embed_dim).to(self.device)
        self.source_review_feat_emb.weight.data.copy_(self.source_review_feat)
        self.source_review_feat_emb.weight.requires_grad = False

        self.target_text_feat_emb = nn.Embedding(self.target_item_num, self.text_embed_dim).to(self.device)
        self.target_text_feat_emb.weight.data.copy_(self.target_text_feat)
        self.target_text_feat_emb.weight.requires_grad = False

        self.target_visual_feat_emb = nn.Embedding(self.target_item_num, self.visual_embed_dim).to(self.device)
        self.target_visual_feat_emb.weight.data.copy_(self.target_visual_feat)
        self.target_visual_feat_emb.weight.requires_grad = False

        self.target_review_feat_emb = nn.Embedding(self.user_num, self.review_embed_dim).to(self.device)
        self.target_review_feat_emb.weight.data.copy_(self.target_review_feat)
        self.target_review_feat_emb.weight.requires_grad = False

        self.source_mat = self.create_sparse_matrix(params['source_train_data'],self.user_num,self.source_item_num)
        self.target_mat = self.create_sparse_matrix(params['target_train_data'],self.user_num,self.target_item_num)
        self.source_norm_adj = self.get_norm_adj_mat(self.source_mat.astype(np.float32),self.user_num,self.source_item_num,self.source_n_nodes).to(self.device)
        self.target_norm_adj = self.get_norm_adj_mat(self.target_mat.astype(np.float32),self.user_num,self.target_item_num,self.target_n_nodes).to(self.device)
        self.source_u_g_embeddings, self.source_v_g_embeddings = self.get_user_item_id_emb(self.source_u_emb,self.source_v_emb,self.user_num,self.source_item_num,self.source_norm_adj)
        self.target_u_g_embeddings, self.target_v_g_embeddings = self.get_user_item_id_emb(self.target_u_emb,
                                                                                           self.target_v_emb,
                                                                                           self.user_num,
                                                                                           self.target_item_num,
                                                                                           self.target_norm_adj)
        self.att = Attention(self.embed_id_dim,self.device)
        self.domain_disen = Domain_Disen(self.device,self.embed_id_dim,self.domain_disen_dim)

        self.source_text_feat_layer = nn.Linear(self.text_embed_dim,self.embed_id_dim).to(self.device)
        self.target_text_feat_layer = nn.Linear(self.text_embed_dim,self.embed_id_dim).to(self.device)
        self.source_visual_feat_layer = nn.Linear(self.visual_embed_dim,self.embed_id_dim).to(self.device)
        self.target_visual_feat_layer = nn.Linear(self.visual_embed_dim,self.embed_id_dim).to(self.device)
        self.source_review_feat_layer = nn.Linear(self.review_embed_dim, self.embed_id_dim).to(self.device)
        self.target_review_feat_layer = nn.Linear(self.review_embed_dim, self.embed_id_dim).to(self.device)

        self.source_v_feat_layer = nn.Linear(self.embed_id_dim,self.domain_disen_dim).to(self.device)
        self.target_v_feat_layer = nn.Linear(self.embed_id_dim, self.domain_disen_dim).to(self.device)

        self.source_pred_1 = nn.Linear(self.domain_disen_dim*2,self.domain_disen_dim).to(self.device)
        self.source_norm_1 = nn.BatchNorm1d(self.domain_disen_dim).to(self.device)
        self.source_pred_2 = nn.Linear(self.domain_disen_dim, self.domain_disen_dim//2).to(self.device)
        self.source_norm_2 = nn.BatchNorm1d(self.domain_disen_dim//2).to(self.device)
        self.source_pred_3 = nn.Linear(self.domain_disen_dim // 2, 1).to(self.device)

        self.target_pred_1 = nn.Linear(self.domain_disen_dim * 2, self.domain_disen_dim).to(self.device)
        self.target_norm_1 = nn.BatchNorm1d(self.domain_disen_dim).to(self.device)
        self.target_pred_2 = nn.Linear(self.domain_disen_dim, self.domain_disen_dim // 2).to(self.device)
        self.target_norm_2 = nn.BatchNorm1d(self.domain_disen_dim // 2).to(self.device)
        self.target_pred_3 = nn.Linear(self.domain_disen_dim // 2, 1).to(self.device)

        self.criterion = nn.BCELoss()


    def create_sparse_matrix(self, df_feat, user_num,item_num,form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat['userID'].values
        tgt = df_feat['itemID'].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(user_num, item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    def get_norm_adj_mat(self, interaction_matrix,user_num,item_num,n_nodes):
        A = sp.dok_matrix((user_num + item_num,
                           user_num + item_num), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + user_num),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + user_num, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
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
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((n_nodes, n_nodes)))

    def get_user_item_id_emb(self,u_emb,v_emb,user_num,item_num,norm_adj):

        h = v_emb.weight

        ego_embeddings = torch.cat((u_emb.weight, v_emb.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [user_num, item_num], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def generate_user_emb(self,nodes_u,u_g_embeddings,v_g_embeddings,ui_inter_lists,domain_flag):
        embed_matrix = torch.empty(len(nodes_u), self.embed_id_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes_u)):
            e_u = u_g_embeddings[int(nodes_u[i])]
            ui_inters = ui_inter_lists[int(nodes_u[i])]
            id_feat = v_g_embeddings[ui_inters]
            if domain_flag == 'source':
                text_feat = self.source_text_feat_emb(torch.tensor(ui_inters).to(self.device))
                text_feat = F.relu(self.source_text_feat_layer(text_feat))
                visual_feat = self.source_visual_feat_emb(torch.tensor(ui_inters).to(self.device))
                visual_feat = F.relu(self.source_visual_feat_layer(visual_feat))
            else:
                text_feat = self.target_text_feat_emb(torch.tensor(ui_inters).to(self.device))
                text_feat = F.relu(self.target_text_feat_layer(text_feat))
                visual_feat = self.target_visual_feat_emb(torch.tensor(ui_inters).to(self.device))
                visual_feat = F.relu(self.target_visual_feat_layer(visual_feat))
            e_v = id_feat+text_feat+visual_feat
            att_w = self.att.forward(e_v, e_u, len(ui_inters))
            att_history = torch.mm(e_v.t(), att_w).t()
            embed_matrix[i] = att_history
        return embed_matrix

    def forward(self, source_nodes_u, source_nodes_v,target_nodes_u,target_nodes_v):
        # logger.info('get user embeddings')
        source_u_review_feats = self.source_review_feat_emb(source_nodes_u)
        source_u_review_feats = F.relu(self.source_review_feat_layer(source_u_review_feats))
        target_u_review_feats = self.target_review_feat_emb(target_nodes_u)
        target_u_review_feats = F.relu(self.target_review_feat_layer(target_u_review_feats))
        source_u_id_feats = self.source_u_g_embeddings[source_nodes_u]
        target_u_id_feats = self.target_u_g_embeddings[target_nodes_u]
        source_v_id_feats = self.source_v_g_embeddings[source_nodes_v]
        target_v_id_feats = self.target_v_g_embeddings[target_nodes_v]
        source_u_feats = source_u_id_feats+source_u_review_feats
        target_u_feats = target_u_id_feats + target_u_review_feats
        # logger.info('get item disen features')
        source_v_text_feat = self.source_text_feat_emb(source_nodes_v)
        source_v_text_feat = F.relu(self.source_text_feat_layer(source_v_text_feat))
        source_v_visual_feat = self.source_visual_feat_emb(source_nodes_v)
        source_v_visual_feat = F.relu(self.source_visual_feat_layer(source_v_visual_feat))
        target_v_text_feat = self.target_text_feat_emb(target_nodes_v)
        target_v_text_feat = F.relu(self.target_text_feat_layer(target_v_text_feat))
        target_v_visual_feat = self.target_visual_feat_emb(target_nodes_v)
        target_v_visual_feat = F.relu(self.target_visual_feat_layer(target_v_visual_feat))
        source_v_feats = source_v_id_feats + source_v_text_feat + source_v_visual_feat
        target_v_feats = target_v_id_feats + target_v_text_feat + target_v_visual_feat
        # logger.info('conduct domain disen')
        source_disen_feats_c, source_disen_feats_c_aug, source_disen_feats_s, source_disen_feats_s_aug, \
        target_disen_feats_c, target_disen_feats_c_aug, target_disen_feats_s, target_disen_feats_s_aug\
            = self.domain_disen.forward(source_u_feats,target_u_feats)
        #add noise
        source_noise = self.add_laplace_noise(source_disen_feats_s)
        source_disen_feats_c = source_disen_feats_c+source_noise
        source_disen_feats_c_aug = source_disen_feats_c_aug+source_noise
        source_disen_feats_s = source_disen_feats_s+source_noise
        source_disen_feats_s_aug = source_disen_feats_s_aug+source_noise
        target_noise = self.add_laplace_noise(target_disen_feats_c)
        target_disen_feats_c = target_disen_feats_c+target_noise
        target_disen_feats_c_aug = target_disen_feats_c_aug+target_noise
        target_disen_feats_s = target_disen_feats_s+target_noise
        target_disen_feats_s_aug = target_disen_feats_s_aug+target_noise

        source_u_feats = source_disen_feats_c+source_disen_feats_s
        target_u_feats = target_disen_feats_c+target_disen_feats_s
        # logger.info('start predict')
        source_v_feats = F.relu(self.source_v_feat_layer(source_v_feats))
        source_pred = torch.concat([source_u_feats,source_v_feats],dim=1)
        source_pred = self.source_norm_1(F.relu(self.source_pred_1(source_pred)))
        source_pred = self.source_norm_2(F.relu(self.source_pred_2(source_pred)))
        source_pred = F.sigmoid(self.source_pred_3(source_pred))

        target_v_feats = F.relu(self.target_v_feat_layer(target_v_feats))
        target_pred = torch.concat([target_u_feats, target_v_feats],dim=1)
        target_pred = self.target_norm_1(F.relu(self.target_pred_1(target_pred)))
        target_pred = self.target_norm_2(F.relu(self.target_pred_2(target_pred)))
        target_pred = F.sigmoid(self.target_pred_3(target_pred))
        return source_disen_feats_c,source_disen_feats_c_aug,source_disen_feats_s,source_disen_feats_s_aug,\
               target_disen_feats_c,target_disen_feats_c_aug,target_disen_feats_s,target_disen_feats_s_aug,\
               source_v_feats,target_v_feats,source_pred.squeeze(),target_pred.squeeze()

    def intra_domain_loss(self,source_disen_feats_c,source_disen_feats_c_aug,source_disen_feats_s,source_disen_feats_s_aug,\
               target_disen_feats_c,target_disen_feats_c_aug,target_disen_feats_s,target_disen_feats_s_aug):
        normalize_source_disen_feats_c = torch.nn.functional.normalize(source_disen_feats_c, p=2, dim=1)
        normalize_source_disen_feats_c_aug = torch.nn.functional.normalize(source_disen_feats_c_aug, p=2, dim=1)
        normalize_source_disen_feats_s = torch.nn.functional.normalize(source_disen_feats_s, p=2, dim=1)
        normalize_source_disen_feats_s_aug = torch.nn.functional.normalize(source_disen_feats_s_aug, p=2, dim=1)

        normalize_target_disen_feats_c = torch.nn.functional.normalize(target_disen_feats_c, p=2, dim=1)
        normalize_target_disen_feats_c_aug = torch.nn.functional.normalize(target_disen_feats_c_aug, p=2, dim=1)
        normalize_target_disen_feats_s = torch.nn.functional.normalize(target_disen_feats_s, p=2, dim=1)
        normalize_target_disen_feats_s_aug = torch.nn.functional.normalize(target_disen_feats_s_aug, p=2, dim=1)
        #
        # source_pos_score_1 = torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_source_disen_feats_c_aug), dim=1)
        # source_pos_score_1 = source_pos_score_1/self.ssl_temp
        # source_neg_score_1 = torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_source_disen_feats_s), dim=1) \
        #                    + torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_source_disen_feats_s_aug),
        #                                dim=1)
        # source_neg_score_1 = source_neg_score_1/self.ssl_temp
        # source_L_ssl_intra_1 = -torch.sum(torch.log(source_pos_score_1 / (source_pos_score_1+source_neg_score_1)))
        #
        # source_pos_score_2 = torch.sum(torch.mul(normalize_source_disen_feats_s, normalize_source_disen_feats_s_aug),dim=1)
        # source_pos_score_2 = source_pos_score_2 / self.ssl_temp
        # source_neg_score_2 = torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_source_disen_feats_s), dim=1) \
        #                      + torch.sum(torch.mul(normalize_source_disen_feats_s, normalize_source_disen_feats_c_aug),dim=1)
        # source_neg_score_2 = source_neg_score_2 / self.ssl_temp
        # source_L_ssl_intra_2 = -torch.sum(torch.log(source_pos_score_2 / (source_pos_score_2 + source_neg_score_2)))
        #
        # source_L_ssl_intra = source_L_ssl_intra_1+source_L_ssl_intra_2
        #
        # target_pos_score_1 = torch.sum(torch.mul(normalize_target_disen_feats_c, normalize_target_disen_feats_c_aug),
        #                                dim=1)
        # target_pos_score_1 = target_pos_score_1 / self.ssl_temp
        # target_neg_score_1 = torch.sum(torch.mul(normalize_target_disen_feats_c, normalize_target_disen_feats_s), dim=1) \
        #                      + torch.sum(torch.mul(normalize_target_disen_feats_c, normalize_target_disen_feats_s_aug),
        #                                  dim=1)
        # target_neg_score_1 = target_neg_score_1 / self.ssl_temp
        # target_L_ssl_intra_1 = -torch.sum(torch.log(target_pos_score_1 / (target_pos_score_1 + target_neg_score_1)))
        #
        # target_pos_score_2 = torch.sum(torch.mul(normalize_target_disen_feats_s, normalize_target_disen_feats_s_aug),
        #                                dim=1)
        # target_pos_score_2 = target_pos_score_2 / self.ssl_temp
        # target_neg_score_2 = torch.sum(torch.mul(normalize_target_disen_feats_c, normalize_target_disen_feats_s), dim=1) \
        #                      + torch.sum(torch.mul(normalize_target_disen_feats_s, normalize_target_disen_feats_c_aug),
        #                                  dim=1)
        # target_neg_score_2 = target_neg_score_2 / self.ssl_temp
        # target_L_ssl_intra_2 = -torch.sum(torch.log(target_pos_score_2 / (target_pos_score_2 + target_neg_score_2)))
        #
        # target_L_ssl_intra = target_L_ssl_intra_1 + target_L_ssl_intra_2




        source_pos_score = torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_source_disen_feats_c_aug), dim=1)+ \
                                torch.sum(torch.mul(normalize_source_disen_feats_s, normalize_source_disen_feats_s_aug),
                                          dim=1)
        source_pos_score = source_pos_score/self.ssl_temp
        source_neg_score =  torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_source_disen_feats_s), dim=1)\
                           +torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_source_disen_feats_s_aug),dim=1)\
                           +torch.sum(torch.mul(normalize_source_disen_feats_s, normalize_source_disen_feats_c_aug), dim=1)\
                           +torch.sum(torch.mul(normalize_source_disen_feats_s_aug, normalize_source_disen_feats_c_aug), dim=1)
        source_neg_score = source_neg_score/self.ssl_temp
        L_ssl_source = -torch.sum(torch.log(source_pos_score / (source_pos_score+source_neg_score)))

        target_pos_score = torch.sum(torch.mul(normalize_target_disen_feats_c, normalize_target_disen_feats_c_aug),
                                     dim=1) + \
                           torch.sum(torch.mul(normalize_target_disen_feats_s, normalize_target_disen_feats_s_aug),
                                     dim=1)
        target_pos_score = torch.exp(target_pos_score / self.ssl_temp)
        target_neg_score = torch.exp(torch.sum(torch.mul(normalize_target_disen_feats_c, normalize_target_disen_feats_s), dim=1)
                          + torch.sum(torch.mul(normalize_target_disen_feats_c, normalize_target_disen_feats_s_aug),dim=1)
                          + torch.sum(torch.mul(normalize_target_disen_feats_s, normalize_target_disen_feats_c_aug), dim=1)
                          + torch.sum(torch.mul(normalize_target_disen_feats_s_aug, normalize_target_disen_feats_c_aug), dim=1))
        target_neg_score = target_neg_score / self.ssl_temp
        L_ssl_target = -torch.sum(torch.log(target_pos_score / (target_pos_score+target_neg_score)))

        L_ssl_intra = L_ssl_source+L_ssl_target
        return L_ssl_intra


    def add_laplace_noise(self,data):

        # self.beta = self.sensitivity/self.epsilon
        noise = torch.tensor(np.random.laplace(0,self.noise_control,data.shape)).to(self.device)
        noise = noise.to(torch.float32)
        # noisy_data = data+noise
        return noise

    def inter_domain_loss(self, source_disen_feats_c, source_disen_feats_c_aug, source_disen_feats_s,source_disen_feats_s_aug, \
                          target_disen_feats_c, target_disen_feats_c_aug, target_disen_feats_s,target_disen_feats_s_aug):
        # source_disen_feats_c = self.add_laplace_noise(source_disen_feats_c)
        normalize_source_disen_feats_c = torch.nn.functional.normalize(source_disen_feats_c, p=2, dim=1)

        # source_disen_feats_s = self.add_laplace_noise(source_disen_feats_s)
        normalize_source_disen_feats_s = torch.nn.functional.normalize(source_disen_feats_s, p=2, dim=1)

        # target_disen_feats_c = self.add_laplace_noise(target_disen_feats_c)
        normalize_target_disen_feats_c = torch.nn.functional.normalize(target_disen_feats_c, p=2, dim=1)

        # target_disen_feats_s = self.add_laplace_noise(target_disen_feats_s)
        normalize_target_disen_feats_s = torch.nn.functional.normalize(target_disen_feats_s, p=2, dim=1)


        pos_score = torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_target_disen_feats_c), dim=1)
        neg_score = torch.sum(torch.mul(normalize_source_disen_feats_c, normalize_target_disen_feats_s), dim=1)+\
                    torch.sum(torch.mul(normalize_source_disen_feats_s, normalize_target_disen_feats_s), dim=1)
        pos_score = torch.exp(pos_score/self.ssl_temp)
        neg_score = torch.exp(neg_score/self.ssl_temp)
        L_ssl_inter = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        return L_ssl_inter

    def loss(self,source_disen_feats_c,source_disen_feats_c_aug,source_disen_feats_s,source_disen_feats_s_aug,\
               target_disen_feats_c,target_disen_feats_c_aug,target_disen_feats_s,target_disen_feats_s_aug,
             source_pred,source_label,target_pred,target_label):
        L_ssl_intra = self.intra_domain_loss(source_disen_feats_c,source_disen_feats_c_aug,source_disen_feats_s,source_disen_feats_s_aug,\
               target_disen_feats_c,target_disen_feats_c_aug,target_disen_feats_s,target_disen_feats_s_aug)
        L_ssl_inter = self.inter_domain_loss(source_disen_feats_c,source_disen_feats_c_aug,source_disen_feats_s,source_disen_feats_s_aug,\
               target_disen_feats_c,target_disen_feats_c_aug,target_disen_feats_s,target_disen_feats_s_aug)
        source_pred_loss = self.criterion(source_pred,source_label.to(torch.float32))
        target_pred_loss = self.criterion(target_pred,target_label.to(torch.float32))
        loss = source_pred_loss+target_pred_loss+self.alpha*(L_ssl_intra+L_ssl_inter)
        return loss,source_pred_loss,target_pred_loss,L_ssl_intra,L_ssl_inter






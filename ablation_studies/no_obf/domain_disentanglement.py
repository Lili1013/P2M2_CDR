
import torch.nn as nn
import torch.nn.functional as F


class Domain_Disen(nn.Module):
    def __init__(self,device,feat_emb_size,feat_disen_dim):
        super(Domain_Disen, self).__init__()
        self.device = device
        self.feat_emb_size = feat_emb_size
        self.feat_disen_dim = feat_disen_dim
        self.d_1_disen_layer_c = nn.Linear(self.feat_emb_size,self.feat_disen_dim).to(self.device)
        self.d_1_disen_layer_s = nn.Linear(self.feat_emb_size, self.feat_disen_dim).to(self.device)
        self.d_2_disen_layer_c = nn.Linear(self.feat_emb_size, self.feat_disen_dim).to(self.device)
        self.d_2_disen_layer_s = nn.Linear(self.feat_emb_size, self.feat_disen_dim).to(self.device)

    def forward(self,source_feats,target_feats):
        source_disen_feats_c = F.dropout(F.relu(self.d_1_disen_layer_c(source_feats)),p=0.2,training=True)
        source_disen_feats_c_aug = F.dropout(F.relu(self.d_1_disen_layer_c(source_feats)),p=0.2,training=True)
        source_disen_feats_s = F.dropout(F.relu(self.d_1_disen_layer_s(source_feats)),p=0.2,training=True)
        source_disen_feats_s_aug = F.dropout(F.relu(self.d_1_disen_layer_s(source_feats)), p=0.2, training=True)

        target_disen_feats_c = F.dropout(F.relu(self.d_2_disen_layer_c(target_feats)),p=0.2,training=True)
        target_disen_feats_c_aug = F.dropout(F.relu(self.d_2_disen_layer_c(target_feats)), p=0.2, training=True)
        target_disen_feats_s = F.dropout(F.relu(self.d_2_disen_layer_s(target_feats)),p=0.2,training=True)
        target_disen_feats_s_aug = F.dropout(F.relu(self.d_2_disen_layer_s(target_feats)), p=0.2, training=True)

        return source_disen_feats_c,source_disen_feats_c_aug,source_disen_feats_s,source_disen_feats_s_aug,\
               target_disen_feats_c,target_disen_feats_c_aug,target_disen_feats_s,target_disen_feats_s_aug
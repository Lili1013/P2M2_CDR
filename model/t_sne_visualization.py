import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pandas as pd
import numpy as np
from loguru import logger
import os
import random as rd

# from data_process import Data_Process
from modal_model import DMMD_CDR
from utils.data_load import load_test_pos_neg
from utils.evaluation import calculate_hr_ndcg
from utils.para_parser import parse
# from data_load import load_test_pos_neg
# from evaluation import calculate_hr_ndcg
import multiprocessing
from functools import partial
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.path_params import phone_elec,phone_sport,sport_cloth,elec_cloth

import warnings
warnings.filterwarnings("ignore")

from dcdr import DCDR

def t_sne_plot(x,sample_num):
    tsneData = TSNE(n_components=2,n_iter=2000,random_state=2022).fit_transform(x)
    # tsneData_1 = TSNE(n_components=2,n_iter=2000,random_state=2022).fit_transform(x_1)
    plt.scatter(tsneData[:sample_num, 0], tsneData[:sample_num, 1],c='purple')
    plt.scatter(tsneData[sample_num:, 0], tsneData[sample_num:, 1],c='orange')
    plt.axis('off')
    plt.show()
    # plt.savefig('initialization.eps',format='eps')
    logger.info('draw complete')
def calculate_vector_sim(vector1,vector2):
    sim = cosine_similarity(vector1,vector2)
    return sim

def generate_user_inter_lists(df):
    ui_interaction = {}
    for x in df.groupby(by='userID'):
        ui_interaction[x[0]] = list(x[1]['itemID'])
    return ui_interaction


def load_data(path_inter,path_text_feat,path_visual_feat,path_review_feat):
    df = pd.read_csv(path_inter)
    num_users = len(df['userID'].unique().tolist())
    num_items = len(df['itemID'].unique().tolist())
    ui_inter_lists = generate_user_inter_lists(df)
    with open(path_text_feat, 'rb') as f:
        text_feat = torch.from_numpy(np.load(f)).to(device)
        # text_feat = pickle.load(f)
    with open(path_visual_feat, 'rb') as f:
        visual_feat = torch.from_numpy(np.load(f)).to(device)
    with open(path_review_feat, 'rb') as f:
        review_feat = torch.from_numpy(np.load(f)).to(device)
        # visual_feat = pickle.load(f)
    train_data = df[df['x_label'] == 0][['userID', 'itemID', 'rating']]
    test_data = df[df['x_label'] == 1][['userID', 'itemID', 'rating']]
    ui_inter_lists_train = generate_user_inter_lists(train_data)
    ui_inter_lists_test = generate_user_inter_lists(test_data)
    return num_users,num_items,ui_inter_lists,text_feat,visual_feat,review_feat,train_data,test_data,df,ui_inter_lists_train,ui_inter_lists_test

def generate_train_batch_for_all_overlap(source_user_inters, source_user_inters_test, source_num_items, target_user_inters, target_user_inters_test,
                                         target_num_items, batch_size):
    t_source = []
    t_target = []
    for b in range(batch_size//2):
        u = rd.sample(source_user_inters.keys(), 1)[0]
        i_source = rd.sample(source_user_inters[u], 1)[0]
        i_target = rd.sample(target_user_inters[u], 1)[0]
        t_source.append([u,i_source,1])
        t_target.append([u,i_target,1])
        while i_source == source_user_inters_test[u]:
            i_source = rd.sample(source_user_inters[u], 1)[0]
        while i_target == target_user_inters_test[u]:
            i_target = rd.sample(target_user_inters[u], 1)[0]
        j_source = rd.randint(0, source_num_items - 1)
        j_target = rd.randint(0, target_num_items - 1)
        while j_source in source_user_inters[u]:
            j_source = rd.randint(0, source_num_items - 1)
        while j_target in target_user_inters[u]:
            j_target = rd.randint(0, target_num_items - 1)
        t_source.append([u, j_source,0])
        t_target.append([u, j_target,0])
    train_batch_source = np.asarray(t_source)
    train_batch_target = np.asarray(t_target)
    return train_batch_source, train_batch_target

def generate_test_batch_for_all_overlap(source_user_inters, source_user_inters_test, source_num_items,
                                        target_user_inters, target_user_inters_test, target_num_items,test_neg_num):
    source_pos_samples = []
    source_neg_samples = []
    target_pos_samples = []
    target_neg_samples = []
    for u in source_user_inters.keys():
        source_test_i = source_user_inters_test[u][0]
        source_i_lists = source_user_inters[u]
        source_pos_samples.append([u,source_test_i])
        source_each_neg_samples = []
        for j in range(test_neg_num):
            k = np.random.randint(0, source_num_items - 1)
            while k in source_i_lists:
                k = np.random.randint(0, source_num_items - 1)
            source_each_neg_samples.append(k)
        source_neg_samples.append(source_each_neg_samples)
        target_test_i = target_user_inters_test[u][0]
        target_i_lists = target_user_inters[u]
        target_pos_samples.append([u, target_test_i])
        target_each_neg_samples = []
        for j in range(test_neg_num):
            k = np.random.randint(0, target_num_items - 1)
            while k in target_i_lists:
                k = np.random.randint(0, target_num_items - 1)
            target_each_neg_samples.append(k)
        target_neg_samples.append(target_each_neg_samples)
    return source_pos_samples,source_neg_samples,target_pos_samples,target_neg_samples

def train(model, device, optimizer, source_ui_inter_lists, source_ui_inter_lists_test, source_num_items,
          target_ui_inter_lists, target_ui_inter_lists_test, target_num_items, batch_size,bar_length):
    '''
    the process of training model
    :param model:
    :return:
    '''
    logger.info('start train')
    model.train()
    running_loss = 0.0

    total_loss = []
    for idx in range(bar_length):

        uij_source, uij_target = generate_train_batch_for_all_overlap(source_ui_inter_lists, source_ui_inter_lists_test,
                                                                      source_num_items, target_ui_inter_lists,
                                                                      target_ui_inter_lists_test,
                                                                      target_num_items, batch_size)
        source_batch_users, source_batch_items, source_batch_r = uij_source[:, 0], uij_source[:,1], uij_source[:, 2]
        target_batch_users, target_batch_items, target_batch_r = uij_target[:, 0], uij_target[:,1], uij_target[:, 2]

        optimizer.zero_grad()
        source_disen_feats_c, source_disen_feats_c_aug, source_disen_feats_s, source_disen_feats_s_aug, \
        target_disen_feats_c, target_disen_feats_c_aug, target_disen_feats_s, target_disen_feats_s_aug, \
        source_v_feats, target_v_feats,source_pred,target_pred\
            = model.forward(torch.tensor(source_batch_users).to(device), torch.tensor(source_batch_items).to(device),
                      torch.tensor(target_batch_users).to(device), torch.tensor(target_batch_items).to(device))
        source_disen_feats_c = source_disen_feats_c.data.cpu().numpy()
        target_disen_feats_c = target_disen_feats_c.data.cpu().numpy()
        source_disen_feats_s = source_disen_feats_s.data.cpu().numpy()
        target_disen_feats_s = target_disen_feats_s.data.cpu().numpy()
        logger.info('start draw')

        c_sim = calculate_vector_sim(source_disen_feats_c, target_disen_feats_c)
        source_c_s_sim = calculate_vector_sim(source_disen_feats_c, source_disen_feats_s)
        target_c_s_sim = calculate_vector_sim(target_disen_feats_c, target_disen_feats_s)
        source_target_s_sim = calculate_vector_sim(source_disen_feats_s,target_disen_feats_s)
        logger.info(c_sim)
        # data_array = np.concatenate((target_disen_feats_c, target_disen_feats_s), axis=0)
        # t_sne_plot(data_array, batch_size)
        data_array1 = np.concatenate((source_disen_feats_c, target_disen_feats_c), axis=0)
        t_sne_plot(data_array1, batch_size)


if __name__ == '__main__':
    # select device: GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    args = parse()

    # initialize parameters
    logger.info('load source data')
    source_num_users, source_num_items, source_ui_inter_lists, source_text_feat, source_visual_feat, source_review_feat,source_train_data, source_test_data,source_df,source_ui_inter_lists_train,source_ui_inter_lists_test \
        = load_data(sport_cloth['source_path_inter'],sport_cloth['source_path_text_feat'],
                    sport_cloth['source_path_visual_feat'],sport_cloth['source_path_review_feat'])
    logger.info('load target data')
    target_num_users, target_num_items, target_ui_inter_lists, target_text_feat, target_visual_feat, target_review_feat,target_train_data, target_test_data, target_df,target_ui_inter_lists_train,target_ui_inter_lists_test \
        = load_data(sport_cloth['target_path_inter'], sport_cloth['target_path_text_feat'],
                    sport_cloth['target_path_visual_feat'],sport_cloth['target_path_review_feat'])
    params={
        'num_users':source_num_users,
        'source_num_items':source_num_items,
        'source_train_data': source_train_data,
        'source_ui_inter_lists_train':source_ui_inter_lists_train,
        'source_text_feat':source_text_feat,
        'source_visual_feat':source_visual_feat,
        'source_review_feat':source_review_feat,
        'target_num_items':target_num_items,
        'target_train_data':target_train_data,
        'target_ui_inter_lists_train':target_ui_inter_lists_train,
        'target_text_feat':target_text_feat,
        'target_visual_feat':target_visual_feat,
        'target_review_feat':target_review_feat,
        'embed_id_dim':args.embed_id_dim,
        'text_embed_dim':args.text_embed_dim,
        'visual_embed_dim':args.visual_embed_dim,
        'review_embed_dim':args.review_embed_dim,
        'domain_disen_dim':args.domain_disen_dim,
        'device':device
    }
    dcdr = DCDR(**params)


    pretrain_dict = torch.load('../data/sport_cloth/model.pth')  # update net parameters
    para_dict = dcdr.state_dict()
    same_para_dict = {k: v for k, v in pretrain_dict.items() if k in para_dict}
    para_dict.update(same_para_dict)
    dcdr.load_state_dict(para_dict)
    optimizer = torch.optim.RMSprop(dcdr.parameters(), lr=args.lr, alpha=0.9)
    logger.info('start training and testing')

    bar_length = 3000

    for epoch in range(1, args.epochs + 1):
        train(dcdr,device,optimizer,source_ui_inter_lists, source_ui_inter_lists_test,source_num_items,
          target_ui_inter_lists,target_ui_inter_lists_test,target_num_items, 500,bar_length)

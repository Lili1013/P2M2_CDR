import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pandas as pd
import numpy as np
from loguru import logger
import os
import random as rd

from utils.evaluation_1 import calculate_hr_ndcg
from utils.para_parser import parse
from utils.path_params import phone_elec,phone_sport,sport_cloth,elec_cloth

import multiprocessing
from functools import partial

import warnings
warnings.filterwarnings("ignore")

from p2m2_cdr import P2M2_CDR


def generate_user_inter_lists(df):
    ui_interaction = {}
    for x in df.groupby(by='userID'):
        ui_interaction[x[0]] = list(x[1]['itemID'])
    return ui_interaction


def load_data(path_inter,path_text_feat,path_review_feat):
    df = pd.read_csv(path_inter)
    num_users = len(df['userID'].unique().tolist())
    num_items = len(df['itemID'].unique().tolist())
    ui_inter_lists = generate_user_inter_lists(df)
    with open(path_text_feat, 'rb') as f:
        text_feat = torch.from_numpy(np.load(f)).to(device)
        # text_feat = pickle.load(f)
    # with open(path_visual_feat, 'rb') as f:
    #     visual_feat = torch.from_numpy(np.load(f)).to(device)
    with open(path_review_feat, 'rb') as f:
        review_feat = torch.from_numpy(np.load(f)).to(device)
        # visual_feat = pickle.load(f)
    train_data = df[df['x_label'] == 0][['userID', 'itemID', 'rating']]
    test_data = df[df['x_label'] == 1][['userID', 'itemID', 'rating']]
    ui_inter_lists_train = generate_user_inter_lists(train_data)
    ui_inter_lists_test = generate_user_inter_lists(test_data)
    return num_users,num_items,ui_inter_lists,text_feat,review_feat,train_data,test_data,df,ui_inter_lists_train,ui_inter_lists_test

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
        loss,source_pred_loss,target_pred_loss,L_ssl_intra,L_ssl_inter = model.loss(source_disen_feats_c,source_disen_feats_c_aug,source_disen_feats_s,source_disen_feats_s_aug,\
               target_disen_feats_c,target_disen_feats_c_aug,target_disen_feats_s,target_disen_feats_s_aug,
                          source_pred,torch.tensor(source_batch_r).to(device),target_pred,torch.tensor(target_batch_r).to(device))
        total_loss.append(loss.item())
        # print(f'{loss.item()},{source_pred_loss.item()},{target_pred_loss.item()},{L_ssl_intra.item()},{L_ssl_inter.item()}')
        running_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        if idx % 100 == 0:
            logger.info('[%d, %5d] loss: %.5f'% (epoch, idx, running_loss / 100))
            running_loss = 0.0
    return total_loss

def test(model, device, top_k,epoch,source_user_inters, source_user_inters_test, source_num_items,target_user_inters, target_user_inters_test, target_num_items,test_neg_num):
    logger.info('start test')
    model.eval()
    with torch.no_grad():
        logger.info('load test pos and neg samples')
        source_test_pos,source_test_neg,target_test_pos,target_test_neg = generate_test_batch_for_all_overlap(source_user_inters, source_user_inters_test, source_num_items,target_user_inters, target_user_inters_test, target_num_items,test_neg_num)
        source_hr, source_ndcg, source_precision, source_recall, source_mrr, \
        target_hr, target_ndcg, target_precision, target_recall, target_mrr, \
        source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1, \
        target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1, \
        source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5, \
        target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5 = calculate_hr_ndcg(model,source_test_pos,source_test_neg,target_test_pos,target_test_neg,top_k,device,epoch)
        # source_hr = sum(source_hits)/len(source_hits)
        # source_ndcg = sum(source_ndcgs)/len(source_ndcgs)
        # target_hr = sum(target_hits) / len(target_hits)
        # target_ndcg = sum(target_ndcgs) / len(target_ndcgs)
    return source_hr, source_ndcg, source_precision, source_recall, source_mrr, \
           target_hr, target_ndcg, target_precision, target_recall, target_mrr, \
           source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1, \
           target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1, \
           source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5, \
           target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5

if __name__ == '__main__':
    # select device: GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    args = parse()
    dataset_name = elec_cloth
    # initialize parameters
    logger.info('load source data')
    source_num_users, source_num_items, source_ui_inter_lists, source_text_feat, source_review_feat,source_train_data, source_test_data,source_df,source_ui_inter_lists_train,source_ui_inter_lists_test \
        = load_data(dataset_name['source_path_inter'],dataset_name['source_path_text_feat'],
                    dataset_name['source_path_review_feat'])
    logger.info('load target data')
    target_num_users, target_num_items, target_ui_inter_lists, target_text_feat, target_review_feat,target_train_data, target_test_data, target_df,target_ui_inter_lists_train,target_ui_inter_lists_test \
        = load_data(dataset_name['target_path_inter'], dataset_name['target_path_text_feat'],
                    dataset_name['target_path_review_feat'])
    logger.info(f'source num users: {source_num_users}, source num items: {source_num_items}, '
                f'source training data: {len(source_train_data)}')
    logger.info(
        f'target num users: {target_num_users}, target num items: {target_num_items}, '
        f'target training data: {len(target_train_data)}')
    logger.info(f'datasets:{args.datasets},emb_dim:{args.embed_id_dim},disen_dim:{args.domain_disen_dim},alpha:{args.alpha},'
                f'ldp_noise:{args.ldp_noise}, lr:{args.lr},batch_size:{args.batch_size},disen_agg_way:{args.agg_way}')
    params={
        'num_users':source_num_users,
        'source_num_items':source_num_items,
        'source_train_data': source_train_data,
        'source_ui_inter_lists_train':source_ui_inter_lists_train,
        'source_text_feat':source_text_feat,
        # 'source_visual_feat':source_visual_feat,
        'source_review_feat':source_review_feat,
        'target_num_items':target_num_items,
        'target_train_data':target_train_data,
        'target_ui_inter_lists_train':target_ui_inter_lists_train,
        'target_text_feat':target_text_feat,
        # 'target_visual_feat':target_visual_feat,
        'target_review_feat':target_review_feat,
        'embed_id_dim':args.embed_id_dim,
        'text_embed_dim':args.text_embed_dim,
        # 'visual_embed_dim':args.visual_embed_dim,
        'review_embed_dim':args.review_embed_dim,
        'domain_disen_dim':args.domain_disen_dim,
        'alpha': args.alpha,
        'ssl_temp': args.ssl_temp,
        'ldp_noise': args.ldp_noise,
        'field': args.field,
        'n_layers':args.n_layers,
        'agg_way': args.agg_way,
        'dropout_rate': args.dropout_rate,
        'device':device
    }
    dcdr = P2M2_CDR(**params,args=args)
    # optimizer = torch.optim.RMSprop(dcdr.parameters(), lr=args.lr, alpha=0.9)
    optimizer = torch.optim.Adam(dcdr.parameters(),lr=args.lr)
    logger.info('start training and testing')
    best_hr = 0.0
    best_ndcg = 0.0
    # endure_count = 1
    bar_length = dataset_name['bar_length']
    save_model = dataset_name['save_model_path']
    # save_model = 'sport_cloth_emb_16.pth'
    source_best_hr,source_best_ndcg,target_best_hr,target_best_ndcg = 0.0,0.0,0.0,0.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = train(dcdr,device,optimizer,source_ui_inter_lists, source_ui_inter_lists_test,source_num_items,
          target_ui_inter_lists,target_ui_inter_lists_test,target_num_items, args.batch_size,bar_length)
        epoch_loss = sum(total_loss)/len(total_loss)
        logger.info('epoch {}, training loss is {}'.format(epoch,epoch_loss))
        source_hr, source_ndcg, source_precision, source_recall, source_mrr, \
        target_hr, target_ndcg, target_precision, target_recall, target_mrr, \
        source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1, \
        target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1, \
        source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5, \
        target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5 = test(dcdr, device, args.top_k,epoch,source_ui_inter_lists, source_ui_inter_lists_test, source_num_items,
                                        target_ui_inter_lists, target_ui_inter_lists_test, target_num_items,args.test_neg_num)
        logger.info('[%d] s_hr_1: %.4f, s_NDCG_1: %.4f, s_mrr_1:%.4f, s_prec_1: %.4f,'
                    's_hr_5: %.4f, s_NDCG_5: %.4f, s_mrr_5:%.4f, s_prec_5: %.4f,'
                    's_hr_10: %.4f, s_NDCG_10: %.4f, s_mrr_10:%.4f, s_prec_10: %.4f,' %
                    (epoch, source_hr_1,source_ndcg_1,source_mrr_1,source_precision_1,
                     source_hr_5,source_ndcg_5,source_mrr_5,source_precision_5,
                     source_hr,source_ndcg,source_mrr,source_precision))
        logger.info('[%d] t_hr_1: %.4f, t_NDCG_1: %.4f, t_mrr_1:%.4f, t_prec_1: %.4f,'
                    't_hr_5: %.4f, t_NDCG_5: %.4f, t_mrr_5:%.4f, t_prec_5: %.4f,'
                    't_hr_10: %.4f, t_NDCG_10: %.4f, t_mrr_10:%.4f, t_prec_10: %.4f,' %
                    (epoch, target_hr_1, target_ndcg_1, target_mrr_1, target_precision_1,
                     target_hr_5, target_ndcg_5, target_mrr_5, target_precision_5,
                     target_hr, target_ndcg, target_mrr, target_precision))

        if source_hr > source_best_hr:
            source_best_hr = source_hr
            source_best_ndcg = source_ndcg
            target_best_hr = target_hr
            target_best_ndcg = target_ndcg
            endure_count = 0
            # torch.save(dcdr.state_dict(), '{}'.format(save_model))
        else:
            endure_count += 1
        if endure_count > 5:
            break
        # logger.info('[%d] Best source HR: %.7f, Best source NDCG: %.7f, Best target HR: %.7f, Best target NDCG: %.7f' %
        #             (epoch, source_best_hr, source_best_ndcg, target_best_hr,target_best_ndcg))

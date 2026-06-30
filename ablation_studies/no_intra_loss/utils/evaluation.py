import math
import heapq
import numpy as np
import torch
import heapq
from loguru import logger

_model = None
_test_ratings = None
_test_negatives = None
_k = None
_device = None
def calculate_hr_ndcg(model,source_test_ratings,source_test_negatives,target_test_ratings,target_test_negatives,k,device,epoch):
    global _model
    global _source_test_negatives
    global _source_test_ratings
    global _target_test_negatives
    global _target_test_ratings
    global _k
    global _device
    global _epoch
    _model = model
    _source_test_ratings = source_test_ratings
    _source_test_negatives = source_test_negatives
    _target_test_ratings = target_test_ratings
    _target_test_negatives = target_test_negatives
    _k = k
    _device = device
    _epoch = epoch

    source_hits,source_ndcgs = [],[]
    target_hits, target_ndcgs = [], []
    for idx in range(len(_source_test_ratings)):
        # if idx % 1000 == 0:
        #     logger.info('{}'.format(idx))
        # logger.info('predict user {}'.format(idx))
        # logger.info('calculate each user metrics')
        (source_hr,source_ndcg,target_hr,target_ndcg) = eval_one_user(idx)
        source_hits.append(source_hr)
        target_hits.append(target_hr)
        source_ndcgs.append(source_ndcg)
        target_ndcgs.append(target_ndcg)
    return (source_hits,source_ndcgs,target_hits,target_ndcgs)

def eval_one_user(idx):
    source_rating = _source_test_ratings[idx]
    source_items = _source_test_negatives[idx]
    target_rating = _target_test_ratings[idx]
    target_items = _target_test_negatives[idx]
    source_u_id = int(source_rating[0])
    source_i_id = int(source_rating[1])
    target_u_id = int(target_rating[0])
    target_i_id = int(target_rating[1])
    source_items.append(source_i_id)
    target_items.append(target_i_id)
    source_user_id_input,source_item_id_input = [],[]
    target_user_id_input, target_item_id_input = [], []
    for i in range(len(source_items)):
        source_user_id_input.append(source_u_id)
        source_item_id_input.append(source_items[i])
    for i in range(len(target_items)):
        target_user_id_input.append(target_u_id)
        target_item_id_input.append(target_items[i])
    source_user_id_input = torch.tensor(source_user_id_input).to(_device)
    source_item_id_input = torch.tensor(source_item_id_input).to(_device)
    target_user_id_input = torch.tensor(target_user_id_input).to(_device)
    target_item_id_input = torch.tensor(target_item_id_input).to(_device)
    source_disen_feats_c, source_disen_feats_c_aug, source_disen_feats_s, source_disen_feats_s_aug, \
    target_disen_feats_c, target_disen_feats_c_aug, target_disen_feats_s, target_disen_feats_s_aug, \
    source_v_feats, target_v_feats, source_pred, target_pred \
        = _model.forward(source_user_id_input,source_item_id_input,target_user_id_input,target_item_id_input)
    source_map_item_score = {}
    target_map_item_score = {}
    for i in range(len(source_items)):
        item = source_items[i]
        source_map_item_score[item] = source_pred[i]
    for i in range(len(target_items)):
        item = target_items[i]
        target_map_item_score[item] = target_pred[i]
    source_ranklist = heapq.nlargest(_k,source_map_item_score,key=source_map_item_score.get)
    target_ranklist = heapq.nlargest(_k, target_map_item_score, key=target_map_item_score.get)
    source_hr = get_hit_ratio(source_ranklist,source_i_id)
    target_hr = get_hit_ratio(target_ranklist, target_i_id)
    source_ndcg = get_ndcg(source_ranklist,source_i_id)
    target_ndcg = get_ndcg(target_ranklist, target_i_id)
    return (source_hr,source_ndcg,target_hr,target_ndcg)

def get_hit_ratio(ranklist,item):
    for each_item in ranklist:
        if each_item == item:
            return 1
    return 0

def get_ndcg(ranklist,item):
    for i in range(len(ranklist)):
        each_item = ranklist[i]
        if each_item == item:
            return math.log(2)/math.log(i+2)
    return 0


def calculate_precision(pred_logits,true_lables):
    # _,pred_leables = torch.max(pred_logits,dim=1)
    pred_lists = pred_logits.tolist()
    pred_labels = []
    for each in pred_lists:
        if each > 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    pred_labels = torch.tensor(pred_labels)
    true_lables = true_lables.int()
    correct_pred = (pred_labels==true_lables).sum().item()
    total_pred = len(true_lables)
    precison = correct_pred/total_pred
    return precison


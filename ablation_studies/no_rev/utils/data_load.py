import pandas as pd


def load_test_pos_neg(path):
    neg_lists = []
    pos_list = []
    user_ids = []
    item_ids = []
    labels = []
    with open(path,'r') as f:
        line = f.readline()
        while line != None and line != '':
            arr = line.split(' ')
            pos_list.append([int(arr[0]),int(arr[1])])
            user_ids.append(int(arr[0]))
            item_ids.append(int(arr[1]))
            labels.append(1)
            negatives = []
            for x in arr[2:]:
                if x == '\n':
                    continue
                user_ids.append(int(arr[0]))
                item_ids.append(int(x))
                labels.append(0)
                negatives.append(int(x))
            neg_lists.append(negatives)
            line = f.readline()
    return pos_list,neg_lists

import math
import heapq
import numpy as np
import torch
import heapq
from loguru import logger
import multiprocessing
from functools import partial
torch.multiprocessing.set_start_method('spawn')

_model = None
_test_ratings = None
_test_negatives = None
_k = None
_device = None
def calculate_hr_ndcg(source_test_ratings,source_test_negatives,target_test_ratings,target_test_negatives,k):
    global _model
    global _source_test_negatives
    global _source_test_ratings
    global _target_test_negatives
    global _target_test_ratings
    global _k
    global _device
    global _epoch

    _source_test_ratings = source_test_ratings
    _source_test_negatives = source_test_negatives
    _target_test_ratings = target_test_ratings
    _target_test_negatives = target_test_negatives
    _k = k


    source_hits,source_ndcgs = [],[]
    target_hits, target_ndcgs = [], []
    u_batch_size = 64
    n_batch = len(_source_test_ratings)//u_batch_size+1
    for u_batch_id in range(n_batch):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        source_user_pos_batch = _source_test_ratings[start: end]
        source_user_neg_batch = _source_test_negatives[start: end]
        target_user_pos_batch = _target_test_ratings[start: end]
        target_user_neg_batch = _target_test_negatives[start: end]
        source_uids,source_iids,target_uids,target_iids = [],[],[],[]
        for idx in range(len(source_user_pos_batch)):
            uid = source_user_pos_batch[idx][0]
            iid = source_user_pos_batch[idx][1]
            source_uids.append(uid)
            source_iids.append(iid)
            items = source_user_neg_batch[idx]
            length_items = len(items)
            users = [uid] * length_items
            source_uids.extend(users)
            source_iids.extend(items)
        for idx in range(len(target_user_pos_batch)):
            uid = target_user_pos_batch[idx][0]
            iid = target_user_pos_batch[idx][1]
            target_uids.append(uid)
            target_iids.append(iid)
            items = target_user_neg_batch[idx]
            length_items = len(items)
            users = [uid] * length_items
            target_uids.extend(users)
            target_iids.extend(items)
        source_uids = torch.tensor(source_uids).to(_device)
        source_iids = torch.tensor(source_iids).to(_device)
        target_uids = torch.tensor(target_uids).to(_device)
        target_iids = torch.tensor(target_iids).to(_device)
        source_disen_feats_c, source_disen_feats_c_aug, source_disen_feats_s, source_disen_feats_s_aug, \
        target_disen_feats_c, target_disen_feats_c_aug, target_disen_feats_s, target_disen_feats_s_aug, \
        source_v_feats, target_v_feats, source_pred, target_pred \
            = _model.forward(source_uids, source_iids, target_uids, target_iids)






    test_num_samples = len(_source_test_ratings)
    chunk_size = test_num_samples // multiprocessing.cpu_count()
    index_ranges = [(i * chunk_size, min((i + 1) * chunk_size, test_num_samples)) for i in
                    range(multiprocessing.cpu_count())]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    func = partial(eval_one_user)
    results = pool.map(func, index_ranges)
    pool.close()
    pool.join()
    if index_ranges[-1][1] < test_num_samples:
        index_ranges = (index_ranges[-1][1], test_num_samples)
    source_hr,source_ndcg,target_hr,target_ndcg = eval_one_user(index_ranges)
    print(results)
    for result in results:
        source_hits.extend(result[0])
        source_ndcgs.extend(result[1])
        target_hits.extend(result[2])
        target_ndcgs.extend(result[3])
    source_hits.extend(source_hr)
    source_ndcgs.extend(source_ndcg)
    target_hits.extend(target_hr)
    target_ndcgs.extend(target_ndcg)

    return (source_hits,source_ndcgs,target_hits,target_ndcgs)

def eval_one_user(idxes):
    source_hrs, source_ndcgs, target_hrs, target_ndcgs = [],[],[],[]
    start_idx = idxes[0]
    end_idx = idxes[1]
    each_idx = start_idx
    while each_idx < end_idx:
        source_rating = _source_test_ratings[each_idx]
        source_items = _source_test_negatives[each_idx]
        target_rating = _target_test_ratings[each_idx]
        target_items = _target_test_negatives[each_idx]
        source_u_id,source_i_id,target_u_id,target_i_id = int(source_rating[0]),int(source_rating[1]),int(target_rating[0]),int(target_rating[1])
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
        source_hrs.append(source_hr)
        source_ndcgs.append(source_ndcg)
        target_hrs.append(target_hr)
        target_ndcgs.append(target_ndcg)
    return source_hrs,source_ndcgs,target_hrs,target_ndcgs

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


if __name__ == '__main__':
    print('load data')
    pos_list,neg_lists = load_test_pos_neg(path='../../data/phone_elec/phones/test_pos_neg.txt')
    calculate_hr_ndcg(pos_list,neg_lists,pos_list,neg_lists,10)



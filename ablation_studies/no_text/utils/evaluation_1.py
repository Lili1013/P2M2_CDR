import math
import heapq
import numpy as np
import torch
import heapq
from loguru import logger
import multiprocessing
import heapq
import random as rd


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    rd.seed(seed)


set_seed(2021)
cores = multiprocessing.cpu_count() // 2

_model = None
_test_ratings = None
_test_negatives = None
_k = None
_device = None


def calculate_hr_ndcg(model, source_test_ratings, source_test_negatives, target_test_ratings, target_test_negatives, k,
                      device, epoch):
    global _model
    global _source_test_negatives
    global _source_test_ratings
    global _target_test_negatives
    global _target_test_ratings
    global _k
    global _device
    global _epoch
    global _source_test_user_records
    global _target_test_user_records
    _model = model
    _source_test_ratings = source_test_ratings
    _source_test_negatives = source_test_negatives
    _target_test_ratings = target_test_ratings
    _target_test_negatives = target_test_negatives
    _k = k
    _device = device
    _epoch = epoch
    pool = multiprocessing.Pool(cores)
    source_hits, source_ndcgs, source_precisions, source_recalls = [], [], [], []
    target_hits, target_ndcgs, target_precisions, target_recalls = [], [], [], []
    test_user_num = len(_source_test_ratings)
    source_pred_ratings = np.zeros(shape=(test_user_num, (len(source_test_negatives[0]) + 1)))
    target_pred_ratings = np.zeros(shape=(test_user_num, (len(source_test_negatives[0]) + 1)))

    source_test_records = np.zeros(shape=(test_user_num, (len(source_test_negatives[0]) + 1)))
    target_test_records = np.zeros(shape=(test_user_num, (len(source_test_negatives[0]) + 1)))

    # source_test_users = list(np.array(_source_test_ratings)[:,0])
    # target_test_users = list(np.array(_target_test_ratings)[:, 0])
    source_test_users = []
    target_test_users = []
    source_test_pos_items_list = []
    target_test_pos_items_list = []

    for idx in range(len(_source_test_ratings)):
        source_u_id = _source_test_ratings[idx][0]
        source_test_users.append(source_u_id)
        source_test_pos_items = _source_test_ratings[idx][1:]
        source_test_pos_items_list.append(source_test_pos_items)
        source_test_neg_items = _source_test_negatives[idx]
        source_items = source_test_pos_items + source_test_neg_items
        source_test_records[idx, :] = source_items

        target_u_id = _target_test_ratings[idx][0]
        target_test_users.append(target_u_id)
        target_test_pos_items = _target_test_ratings[idx][1:]
        target_test_pos_items_list.append(target_test_pos_items)
        target_test_neg_items = _target_test_negatives[idx]
        target_items = target_test_pos_items + target_test_neg_items
        target_test_records[idx, :] = target_items
        # _target_test_user_records[target_u_id].append([target_test_pos_item])
        # _target_test_user_records[target_u_id].append(target_test_neg_items)

        source_pred, target_pred = pred_one_user(source_u_id, source_items, target_u_id, target_items)
        source_pred_ratings[idx, :] = source_pred.detach().cpu()
        target_pred_ratings[idx, :] = target_pred.detach().cpu()
    #top_k=10----------------
    top_k_10 = [10 for i in range(len(source_test_users))]
    source_rating_uid = zip(source_pred_ratings, source_test_users, source_test_records, source_test_pos_items_list,top_k_10)
    target_rating_uid = zip(target_pred_ratings, target_test_users, target_test_records, target_test_pos_items_list,top_k_10)
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # func = partial(test_one_user, x=source_rating_uid,test_user_records=_source_test_user_records,flag='source')
    # source_result = pool.map(func)
    source_result = pool.map(test_one_user, source_rating_uid)
    source_hr, source_ndcg, source_precision, source_recall, source_mrr = obtain_final_result(source_result)
    target_result = pool.map(test_one_user, target_rating_uid)
    target_hr, target_ndcg, target_precision, target_recall, target_mrr = obtain_final_result(target_result)
    # top_k=1----------------
    top_k_1 = [1 for i in range(len(source_test_users))]
    source_rating_uid_1 = zip(source_pred_ratings, source_test_users, source_test_records, source_test_pos_items_list,
                            top_k_1)
    target_rating_uid_1 = zip(target_pred_ratings, target_test_users, target_test_records, target_test_pos_items_list,
                            top_k_1)
    source_result_1 = pool.map(test_one_user, source_rating_uid_1)
    source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1 = obtain_final_result(source_result_1)
    target_result_1 = pool.map(test_one_user, target_rating_uid_1)
    target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1 = obtain_final_result(target_result_1)

    # top_k=5----------------
    top_k_5 = [5 for i in range(len(source_test_users))]
    source_rating_uid_5 = zip(source_pred_ratings, source_test_users, source_test_records, source_test_pos_items_list,
                              top_k_5)
    target_rating_uid_5 = zip(target_pred_ratings, target_test_users, target_test_records, target_test_pos_items_list,
                              top_k_5)
    source_result_5 = pool.map(test_one_user, source_rating_uid_5)
    source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5 = obtain_final_result(source_result_5)
    target_result_5 = pool.map(test_one_user, target_rating_uid_5)
    target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5 = obtain_final_result(target_result_5)



    pool.close()

    return source_hr, source_ndcg, source_precision, source_recall, source_mrr, \
           target_hr, target_ndcg, target_precision, target_recall, target_mrr, \
           source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1, \
           target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1, \
           source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5, \
           target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5

def obtain_final_result(source_result):
    source_result = np.array(source_result)
    source_hr = np.mean(source_result[:, 0])
    source_ndcg = np.mean(source_result[:, 1])
    source_precision = np.mean(source_result[:, 2])
    source_recall = np.mean(source_result[:, 3])
    source_mrr = np.mean(source_result[:, 4])
    return source_hr,source_ndcg,source_precision,source_recall,source_mrr


def test_one_user(x):
    rating = x[0]
    test_user_records = x[2]
    user_pos_test = x[3]
    top_k = x[4]
    test_items = test_user_records
    r = ranklist_by_heapq(test_items, rating,top_k)
    return eval_one_user(r, user_pos_test)


def ranklist_by_heapq(test_items, rating,top_k):
    item_score = {}
    for i in range(len(test_items)):
        item_score[int(test_items[i])] = rating[i]

    K_item_score = heapq.nlargest(top_k, item_score, key=item_score.get)
    return K_item_score


def pred_one_user(source_u_id, source_items, target_u_id, target_items):
    source_user_id_input, source_item_id_input = [], []
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
        = _model.forward(source_user_id_input, source_item_id_input, target_user_id_input, target_item_id_input)
    return source_pred, target_pred


def eval_one_user(r, user_pos_test):
    hr = get_hit_ratio(r, user_pos_test)
    precision = get_precision(r, user_pos_test)
    recall = get_recall(r, user_pos_test)
    ndcg = get_ndcg(r, user_pos_test)
    mrr = get_mrr(r, user_pos_test)
    # target_ndcg = get_ndcg(target_ranklist, target_i_id)
    # target_precision = get_precision(target_ranklist,_k)
    # target_recall = get_recall(target_ranklist, _k, 1)
    return [hr, ndcg, precision, recall, mrr]


def get_hit_ratio(ranklist, user_pos_test):
    for each_item in ranklist:
        if each_item in user_pos_test:
            return 1
    return 0


def get_ndcg(ranklist, user_pos_test):
    for i in range(len(ranklist)):
        each_item = ranklist[i]
        if each_item in user_pos_test:
            return math.log(2) / math.log(i + 2)
    return 0


def get_precision(ranklist, user_pos_test):
    precision_items = []
    for each in ranklist:
        if each in user_pos_test:
            precision_items.append(1)
    return sum(precision_items) / len(ranklist)


def get_recall(ranklist, user_pos_test):
    recall_items = []
    for each in ranklist:
        if each in user_pos_test:
            recall_items.append(1)
    return sum(recall_items) / len(user_pos_test)


def get_mrr(ranklist, user_pos_test):
    for i in range(len(ranklist)):
        if ranklist[i] in user_pos_test:
            return 1 / (i + 1)
    return 0



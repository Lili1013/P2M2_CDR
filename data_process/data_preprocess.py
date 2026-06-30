import gzip
import json
import pandas as pd
from loguru import logger

def preprocess(json_file_path,to_path):
    ''''
    extract all users and items from the original files
    '''
    # json_gz_file_path = 'C:\work\Amazon_datasets\\Electronics_5.json'

    fin = open(json_file_path, 'r')
    review_list = []
    i = 0# 存储筛选出来的字段，如果数据量过大可以尝试用dict而不是list
    for line in fin:
        # 顺序读取json文件的每一行
        try:
            if i % 10000 == 0:
                logger.info(i)
            d = eval(line, {"true":True,"false":False,"null":None})
            review_list.append([d['reviewerID'],d['asin']])
        except:
            continue
        i += 1
    df = pd.DataFrame(review_list, columns =['user_id', 'item_id']) # 转换为dataframe
    df.to_csv(to_path,index=False)

def find_common_users(df_elec,df_phone):
    '''
    find the common users from cross domain data
    :return:
    '''
    users_elec = df_elec['user_id'].unique().tolist()
    users_phone = df_phone['user_id'].unique().tolist()
    inter_users = list(set(users_phone)&set(users_elec))
    return inter_users
def select_data(inter_users,df,to_path):
    '''
    select the records with common users from cross domain data
    :param inter_users:
    :param json_file_path:
    :param to_path:
    :return:
    '''
    df_new = df[df['user_id'].isin(inter_users)]
    # df_new = df_new[df_new['rating']>=3.0]
    # df_new.loc[:,['item_id','user_id']] = df_new.loc[:,['user_id','item_id']].values
    # df_new.columns = ['user_id','item_id','rating','timestamp']
    df_new.to_csv(to_path, index=False)

def filter_g_k_one(data,k_user=5,k_item=10,u_name='user_id',i_name='item_id',y_name='rating'):
    '''
    delete the records that user and item interactions lower than k
    '''
    item_group = data.groupby(i_name).agg({y_name:'count'}) #every item has the number of ratings
    item_g10 = item_group[item_group[y_name]>=k_item].index
    data_new = data[data[i_name].isin(item_g10)]

    user_group = data_new.groupby(u_name).agg({y_name: 'count'})  # every user has the number of ratings
    user_g10 = user_group[user_group[y_name] >= k_user].index
    data_new = data_new[data_new[u_name].isin(user_g10)]
    return data_new

def id_map(source_path,user_to_path,item_to_path,to_path):
    df = pd.read_csv(source_path)
    df_users = df.sort_values(by=['user_id'])
    df_items = df.sort_values(by=['item_id'])
    uni_users = df_users['user_id'].unique().tolist()
    uni_items = df_items['item_id'].unique().tolist()

    # start from 0
    u_id_map = {k: i for i, k in enumerate(uni_users)}
    i_id_map = {k: i for i, k in enumerate(uni_items)}

    u_df = pd.DataFrame(list(u_id_map.items()), columns=['user_id', 'userID'])
    i_df = pd.DataFrame(list(i_id_map.items()), columns=['item_id', 'itemID'])
    df['userID'] = df['user_id'].map(u_id_map)
    df['itemID'] = df['item_id'].map(i_id_map)
    df['userID'] = df['userID'].astype(int)
    df['itemID'] = df['itemID'].astype(int)
    u_df.to_csv(user_to_path,index=False)
    i_df.to_csv(item_to_path,index=False)
    df.to_csv(to_path,index=False)

if __name__ == '__main__':
    #----------first step---------------
    #source domain
    df_source = pd.read_csv('../datasets/ratings_Cell_Phones_and_Accessories.csv')
    columns_source = ['user_id','item_id','rating','timestamp']
    df_source.columns = columns_source
    print(len(df_source))
    df_source_new = filter_g_k_one(data=df_source,k_user=10,k_item=10,u_name='user_id',i_name='item_id',y_name='rating')
    print(len(df_source_new))
    #target domain
    df_target = pd.read_csv('/home/lwang9/Data/amazon_data_process/datasets/ratings_Sports_and_Outdoors.csv')
    columns_target = ['user_id','item_id','rating','timestamp']
    df_target.columns = columns_target
    print(len(df_target))
    df_target_new = filter_g_k_one(data=df_target,k_user=10,k_item=10,u_name='user_id',i_name='item_id',y_name='rating')
    print(len(df_target_new))
    # find common users
    inter_users = find_common_users(df_target_new,df_source_new)
    print(len(inter_users))
    select_data(inter_users, df_source_new, to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/data_phone.csv')
    select_data(inter_users,df_target_new,to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/sport/data_sport.csv')
    # # select_data(inter_users, df_source_new, to_path='../datasets/sport_cloth/cloth/data_cloth.csv')
    # # select_data(inter_users, df_target_new, to_path='../datasets/sport_cloth/sport/data_sport.csv')
    # select_data(inter_users, df_source_new, to_path='../datasets/music_movie/music/data_music.csv')
    # select_data(inter_users, df_target_new, to_path='../datasets/music_movie/movie/data_movie.csv')
    #-----------second step-------------
    #ID map
    id_map(source_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/data_phone.csv',
           user_to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/user_id_map.csv',
           item_to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/item_id_map.csv',
           to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_inter.csv')
    id_map(source_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/sport/data_sport.csv',
           user_to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/sport/user_id_map.csv',
           item_to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/sport/item_id_map.csv',
           to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/sport/sport_inter.csv')





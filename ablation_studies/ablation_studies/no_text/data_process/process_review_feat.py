import gzip
import pandas as pd
from sentence_transformers	import SentenceTransformer
import numpy as np
from loguru import logger

def select_reviews(source_path,to_path,items,users):
    g = gzip.open(source_path, 'r')
    review_list = []
    i = 0
    for line in g:
        d = eval(line, {"true": True, "false": False, "null": None})
        if (d['asin'] in items) and (d['reviewerID'] in users):
            if i % 10000 == 0:
                logger.info(i)
            i+=1
            review_list.append([d['reviewerID'], d['asin'],d['reviewText']])
    df = pd.DataFrame(review_list, columns=['user_id', 'item_id','review_text'])  # 转换为dataframe
    df.to_csv(to_path, index=False)

def concat_reviews(source_path,to_path):
    df = pd.read_csv(source_path)
    df['review_text'].fillna(' ', inplace=True)
    df_lists = []
    # print('start group by')
    logger.info('start group by')
    for x in df.groupby(by='user_id'):
        each_df = pd.DataFrame({
            'user_id': [x[0]],
            'review_texts': [';'.join(x[1]['review_text'])]
        })
        df_lists.append(each_df)

    df = pd.concat(df_lists, axis=0)
    # print('start store')
    logger.info('start store')
    df.to_csv(to_path, index=False)

def generate_review_emb(source_path,source_path_text,to_path):
    df = pd.read_csv(source_path)
    df.sort_values(by=['user_id'], inplace=True)
    orig_user_ids  = df['user_id'].unique().tolist()
    df_text = pd.read_csv(source_path_text)
    # logger.info('process null value')
    logger.info('process null value')
    df_text.sort_values(by=['user_id'], inplace=True)
    # df_text['description'] = df_text['description'].fillna(" ")
    # df_text['title'] = df_text['title'].fillna(" ")
    # df_text['categories'] = df_text['categories'].fillna(" ")
    sentences = []
    lack_index = []
    for each_user in orig_user_ids:
        row = df_text[df_text['user_id']==each_user]
        if len(row) == 0:
            print(each_user)
            lack_index.append(orig_user_ids.index(each_user))
        else:
            sen = row['review_texts'].iloc[0]
            sen = sen.replace('\n', ' ')
            sentences.append(sen)

    # print('start transform')
    logger.info('start transform')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    # assert sentence_embeddings.shape[0] == df_text.shape[0]
    fill_list = [0] * 384
    for each_index in lack_index:
        sentence_embeddings = np.insert(sentence_embeddings,each_index,fill_list,axis=0)
    np.save(to_path, sentence_embeddings)
    logger.info('done')
    # print('done!')


if __name__ == '__main__':
    # first step
    df = pd.read_csv('/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_inter.csv')
    items = df['item_id'].unique().tolist()
    users = df['user_id'].unique().tolist()
    select_reviews(source_path='/home/lwang9/Data/amazon_data_process/datasets/reviews_Cell_Phones_and_Accessories.json.gz',
                   to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_reviews_orig.csv', items=items, users=users)
    # # second step
    concat_reviews(source_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_reviews_orig.csv',
                   to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_reviews.csv')
    generate_review_emb(source_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_inter.csv',
                        source_path_text='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_reviews.csv',
                        to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_review_feat.npy')







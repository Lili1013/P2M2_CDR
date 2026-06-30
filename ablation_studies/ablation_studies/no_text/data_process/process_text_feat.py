import pandas as pd
import gzip
from sentence_transformers	import SentenceTransformer
import numpy as np
import os
from loguru import logger

def select_data(df,meta_path,to_path):

    items = df['item_id'].unique().tolist()
    g = gzip.open(meta_path, 'rb')
    i = 0
    for line in g:
        d = eval(line)
        if d['asin'] in items:
            try:
                item_id = d['asin']
            except:
                item_id = ''
            try:
                im_url = d['imUrl']
            except:
                im_url = ''
            try:
                description = d['description']
            except:
                description = ''
            try:
                categories = ','.join(d['categories'][0])
            except:
                categories = ''
            try:
                title = d['title']
            except:
                title = ''
            each_line = {
                'item_id':[item_id],
                'im_url':[im_url],
                'description':[description],
                'categories':[categories],
                'title':[title]
            }
            each_line_df = pd.DataFrame(each_line)
            each_line_df.to_csv(to_path,mode='a',header=False,index=False)
            # print(i)
            i += 1
def process_text_data(source_path,source_path_text,to_path):
    df = pd.read_csv(source_path)
    df.sort_values(by=['item_id'], inplace=True)
    orig_item_ids  = df['item_id'].unique().tolist()
    df_text = pd.read_csv(source_path_text)
    df_text.columns = ['item_id', 'im_url', 'description', 'categories', 'title']
    logger.info('process null value')
    df_text.sort_values(by=['item_id'], inplace=True)
    df_text['description'] = df_text['description'].fillna(" ")
    df_text['title'] = df_text['title'].fillna(" ")
    df_text['categories'] = df_text['categories'].fillna(" ")
    sentences = []
    lack_index = []
    for each_item in orig_item_ids:
        row = df_text[df_text['item_id']==each_item]
        if len(row) == 0:
            print(each_item)
            lack_index.append(orig_item_ids.index(each_item))
        else:
            sen = row['title'].iloc[0] + ' ' + row['categories'].iloc[0] + ' ' + row['description'].iloc[0]
            sen = sen.replace('\n', ' ')
            sentences.append(sen)

    logger.info('start transform')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    # assert sentence_embeddings.shape[0] == df_text.shape[0]
    fill_list = [0] * 384
    for each_index in lack_index:
        sentence_embeddings = np.insert(sentence_embeddings,each_index,fill_list,axis=0)
    np.save(to_path, sentence_embeddings)
    logger.info('done!')

if __name__ == '__main__':
    df = pd.read_csv('/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_inter.csv')
    select_data(df,meta_path='/home/lwang9/Data/amazon_data_process/datasets/meta_Cell_Phones_and_Accessories.json.gz',
                to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/meta_phone_data.csv')
    process_text_data(source_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_inter.csv',
                      source_path_text='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/meta_phone_data.csv',
                      to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_text_feat.npy')


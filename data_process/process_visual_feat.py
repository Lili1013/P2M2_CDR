import array

import pandas as pd
import numpy as np
from loguru import logger


def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10).decode('UTF-8')
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()
def process_visual_feat(source_path,source_image_path,to_path):
  logger.info('read csv')
  df = pd.read_csv(source_path)
  df.sort_values(by=['item_id'], inplace=True)
  item2id = df['item_id'].unique().tolist()
  img_data = readImageFeatures(source_image_path)
  # item2id = dict(zip(df['asin'], df['itemID']))
  feats = {}
  avg = []
  logger.info('start image data')
  for d in img_data:
    if d[0] in item2id:
      feats[d[0]] = d[1]
      avg.append(d[1])
  avg = np.array(avg).mean(0).tolist()

  ret = []
  non_no = []
  logger.info('start filter')
  for i in item2id:
    if i in feats:
      ret.append(feats[i])
    else:
      non_no.append(i)
      ret.append(avg)

  logger.info('# of items not in processed image features:', len(non_no))
  assert len(ret) == len(item2id)
  np.save(to_path, np.array(ret))
  logger.info('complete')

if __name__ == '__main__':
  process_visual_feat(source_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_inter.csv',
                      source_image_path='/home/lwang9/Data/amazon_data_process/datasets/image_features_Cell_Phones_and_Accessories.b',
                      to_path='/home/lwang9/Data/amazon_data_process/datasets/phone_sport/phone/phone_visual_feat.npy')
  # process_visual_feat(source_path='../datasets/elec/elec_inter.csv',
  #                     source_image_path='../datasets/image_features_Electronics.b',
  #                     to_path='../datasets/elec/elec_visual_feat.npy')

  # process_visual_feat(source_path='../datasets/phone_sport/phone/phone_inter.csv',
  #                     source_image_path='../datasets/image_features_Cell_Phones_and_Accessories.b',
  #                     to_path='../datasets/phone_sport/phone/phone_visual_feat.npy')
  #
  # process_visual_feat(source_path='../datasets/phone_sport/sport/sport_inter.csv',
  #                     source_image_path='../datasets/image_features_Sports_and_Outdoors.b',
  #                     to_path='../datasets/phone_sport/sport/sport_visual_feat.npy')
  # process_visual_feat(source_path='../datasets/sport_cloth/cloth/cloth_inter.csv',
  #                     source_image_path='../datasets/image_features_Clothing_Shoes_and_Jewelry.b',
  #                     to_path='../datasets/sport_cloth/cloth/cloth_visual_feat.npy')

  # process_visual_feat(source_path='../datasets/sport_cloth/sport/sport_inter.csv',
  #                     source_image_path='../datasets/image_features_Sports_and_Outdoors.b',
  #                     to_path='../datasets/sport_cloth/sport/sport_visual_feat.npy')
  # process_visual_feat(source_path='../datasets/elec_cloth/cloth/cloth_inter.csv',
  #                     source_image_path='../datasets/image_features_Clothing_Shoes_and_Jewelry.b',
  #                     to_path='../datasets/elec_cloth/cloth/cloth_visual_feat.npy')
  #
  # process_visual_feat(source_path='../datasets/book_music/book/book_inter.csv',
  #                     source_image_path='../datasets/image_features_Books.b',
  #                     to_path='../datasets/book_music/book/book_visual_feat.npy')


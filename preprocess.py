import time
import os
import pickle
import pandas as pd
import numpy as np

data_name = 'Beibei'
# data_name = 'Taobao'

if os.path.isdir('preprocess/'+data_name+'/') == False:
    os.makedirs('preprocess/'+data_name+'/')

data_path = './preprocess/'+data_name+'/'

view = pd.read_csv(data_path+'view.pkl', sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})[['user_id', 'item_id']]
cart = pd.read_csv(data_path+'cart.pkl', sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})[['user_id', 'item_id']]
buy_train = pd.read_csv(data_path+'buy.train.pkl', sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})
buy_test = pd.read_csv(data_path+'buy.test.pkl', sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})
print(time.strftime('%Y-%m-%d %H:%M:%S'), data_name, '数据读取完毕')

cart = cart.append(buy_train)
view = view.append(cart)

buy = buy_train.append(buy_test)
whole_users = set(buy['user_id'].unique())
whole_items = set(buy['item_id'].unique())
num_whole_users = max(whole_users) + 1
num_whole_items = max(whole_items) + 1
print(time.strftime('%Y-%m-%d %H:%M:%S'), data_name, 'Number of Users: ', num_whole_users)
print(time.strftime('%Y-%m-%d %H:%M:%S'), data_name, 'Number of Items: ', num_whole_items)

view_positive_item = pd.merge(pd.DataFrame(np.array(range(num_whole_users)), columns=['user_id']), view.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns={'item_id':'positives'}), how='outer', on='user_id')
user_nan_view = list(view_positive_item[view_positive_item['positives'].isnull().T]['user_id'])
view_positive_item_list = []
for i in range(num_whole_users):
    if i in user_nan_view:
        view_positive_item_list.append([])
    else:
        view_positive_item_list.append(list(view_positive_item.iloc[i].tolist()[1]))
with open('preprocess/'+data_name+'/view.pkl','wb') as save1:
    pickle.dump(view_positive_item_list, save1, protocol=pickle.HIGHEST_PROTOCOL)
print(time.strftime('%Y-%m-%d %H:%M:%S'), data_name,  'view Train 统计完毕')

cart_positive_item = pd.merge(pd.DataFrame(np.array(range(num_whole_users)), columns=['user_id']), cart.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns={'item_id':'positives'}), how='outer', on='user_id')
user_nan_cart = list(cart_positive_item[cart_positive_item['positives'].isnull().T]['user_id'])
cart_positive_item_list = []
for i in range(num_whole_users):
    if i in user_nan_cart:
        cart_positive_item_list.append([])
    else:
        cart_positive_item_list.append(list(cart_positive_item.iloc[i].tolist()[1]))
with open('preprocess/'+data_name+'/cart.pkl','wb') as save2:
    pickle.dump(cart_positive_item_list, save2, protocol=pickle.HIGHEST_PROTOCOL)
print(time.strftime('%Y-%m-%d %H:%M:%S'), data_name,  'Cart Train 统计完毕')

buy_train_positive_item = buy_train.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns={'item_id':'positives'})
buy_train_positive_item_list = [list(buy_train_positive_item.iloc[i].tolist()[1]) for i in range(num_whole_users)]
with open('preprocess/'+data_name+'/buy_train.pkl','wb') as save3:
    pickle.dump(buy_train_positive_item_list, save3, protocol=pickle.HIGHEST_PROTOCOL)
print(time.strftime('%Y-%m-%d %H:%M:%S'), data_name,  'Buy Train 统计完毕')

buy_test_positive_item = [buy_test.iloc[i].tolist()[1] for i in range(num_whole_users)]
with open('preprocess/'+data_name+'/buy_test.pkl','wb') as save4:
    pickle.dump(buy_test_positive_item, save4, protocol=pickle.HIGHEST_PROTOCOL)
print(time.strftime('%Y-%m-%d %H:%M:%S'), data_name,  'Buy Test 统计完毕')
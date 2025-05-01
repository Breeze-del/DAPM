from loguru import logger
import argparse
import torch
import time
import numpy as np
import random
time = 0
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2025)
parser = argparse.ArgumentParser(description='REC')
parser.add_argument('--learning_rate', default='0.001', type=float,
                        help='Learning rate.')
parser.add_argument('--dropout_ration', default='0.2', type=float,
                        help='Drop out ration.')
parser.add_argument('--batch_size', default='1024', type=int,
                        help='batch size.')
parser.add_argument('--weight_negative', default='0.5', type=float,
                        help='weight for negative entry.')
parser.add_argument('--alpha', default='0.0', type=float,
                        help='alpha.')
parser.add_argument('--gpu', default='0', type=int,
                        help='Dataset name.')
parser.add_argument('--dataset', default='0', type=int,
                        help='0 for beibei and 1 for taobao，2 for tmall ')
parser.add_argument('--v', default='0', type=int,
                        help='show the training process')
parser.add_argument('--weight_1', default='0.166667', type=float,
                        help='Dataset name.')
parser.add_argument('--weight_2', default='0.666667', type=float,
                        help='Dataset name.')
parser.add_argument('--weight_3', default='0.166667', type=float,
                        help='Dataset name.')
parser.add_argument('--weight_4', default='0.0', type=float,
                        help='Dataset name.')
parser.add_argument('--n_layers', default='1', type=int,
                        help='num_layers for GCN.')
parser.add_argument('--temperature', default='0.7', type=float,
                        help='temperature for contrastive learning.')
parser.add_argument('--weight_ssl', default='1.0', type=float,
                        help='weight for contrastive learning.')
parser.add_argument('--weight_L2', default='1e-4', type=float,
                        help='weight for L2 regularization.')
parser.add_argument('--weight_boundary', default='0.01', type=float,
                        help='weight for boundary constraint.')

args = parser.parse_args()
if args.dataset == 0:
    data_name = 'Beibei'
elif args.dataset == 1:
    data_name = 'Taobao'
elif args.dataset == 2:
    data_name = 'Tmall'

batch_size = args.batch_size
L2_weight = args.weight_L2

logger.info(args)

Epochs = 500


learning_rate = args.learning_rate
dropout_ration = args.dropout_ration
dim_embedding = int(64)

lambdas = {'view':args.weight_1, 'cart':args.weight_2, 'buy':args.weight_3, 'collect':args.weight_4}

each_process_users = 64

torch.cuda.set_device(args.gpu)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

################################################################################################################################################################

import time
from itertools import chain
import pickle
# preprocess
with open('preprocess/'+data_name+'/view.pkl','rb') as load1:
    view = pickle.load(load1)
with open('preprocess/'+data_name+'/cart.pkl','rb') as load2:
    cart = pickle.load(load2)
with open('preprocess/'+data_name+'/buy_train.pkl','rb') as load3:
    buy_train = pickle.load(load3)
with open('preprocess/'+data_name+'/buy_test.pkl','rb') as load4:
    buy_test = pickle.load(load4)
# with open('preprocess/'+data_name+'/collect.pkl','rb') as load5:
#     collect = pickle.load(load5)

num_whole_users = len(buy_train)
num_whole_items = max(list(chain.from_iterable(buy_train))+buy_test) + 1
max_length_test = max([num_whole_items-len(buy_train[i]) for i in range(num_whole_users)])
################################################################################################################################################################

import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
class Read_Data(Dataset):
    def __init__(self, num_whole_users, num_whole_items, train_data_view=None, train_data_cart=None, train_data_buy=None, train_data_collect=None):
        self.num_whole_users = num_whole_users
        self.num_whole_items = num_whole_items
        self.train_data_view = train_data_view
        self.train_data_cart = train_data_cart
        self.train_data_buy = train_data_buy
        # self.train_data_collect = train_data_collect

        self.max_train_length_view = max([len(self.train_data_view[i]) for i in range(self.num_whole_users)])
        self.max_train_length_cart = max([len(self.train_data_cart[i]) for i in range(self.num_whole_users)])
        self.max_train_length_buy = max([len(self.train_data_buy[i]) for i in range(self.num_whole_users)])
        # self.max_train_length_collect = max([len(self.train_data_collect[i]) for i in range(self.num_whole_users)])
    def __getitem__(self, index):
        user_positive_items_view = self.train_data_view[index]
        user_positive_items_view.extend([self.num_whole_items]*(self.max_train_length_view-len(user_positive_items_view)))

        user_positive_items_cart = self.train_data_cart[index]
        user_positive_items_cart.extend([self.num_whole_items]*(self.max_train_length_cart-len(user_positive_items_cart)))

        user_positive_items_buy = self.train_data_buy[index]
        user_positive_items_buy.extend([self.num_whole_items]*(self.max_train_length_buy-len(user_positive_items_buy)))

        # user_id = self.train_data_view[index][0]
        # user_positive_items_view = self.train_data_view[index][1:]
        # user_positive_items_view.extend(
        #     [self.num_whole_items] * (self.max_train_length_view - len(user_positive_items_view)))
        #
        # user_positive_items_cart = self.train_data_cart[index][1:]
        # user_positive_items_cart.extend(
        #     [self.num_whole_items] * (self.max_train_length_cart - len(user_positive_items_cart)))
        #
        # user_positive_items_buy = self.train_data_buy[index][1:]
        # user_positive_items_buy.extend(
        #     [self.num_whole_items] * (self.max_train_length_buy - len(user_positive_items_buy)))

        # user_positive_items_collect = self.train_data_collect[index][1:]
        # user_positive_items_collect.extend(
        #     [self.num_whole_items] * (self.max_train_length_collect - len(user_positive_items_collect)))


        return index, torch.LongTensor(user_positive_items_view), torch.LongTensor(user_positive_items_cart), torch.LongTensor(user_positive_items_buy)
    def __len__(self):
        return self.num_whole_users

    def getAdjMat(self):
        # Generate behavior-specific graph
        R = sp.load_npz('preprocess/' + data_name + '/adj_buy.npz')
        R_pv = sp.load_npz('preprocess/' + data_name + '/adj_view.npz')
        R_cart = sp.load_npz('preprocess/' + data_name + '/adj_cart.npz')
        R_all = sp.load_npz('preprocess/' + data_name + '/adj_all.npz')
        # R_collect = sp.load_npz('preprocess/' + data_name + '/adj_collect.npz')
        return R, R_pv, R_cart, R_all
dataset = Read_Data(num_whole_users=num_whole_users, \
                    num_whole_items=num_whole_items, \
                    train_data_view=view, \
                    train_data_cart=cart, \
                    train_data_buy=buy_train)
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
################################################################################################################################################################
from model_full import Model
from utils import evaluation
time_list = []
import multiprocessing
if __name__ == '__main__':
    # Load adjacency matrix
    config = dict()
    pre_adj, pre_adj_pv, pre_adj_cart, pre_adj_all_behaviors = dataset.getAdjMat()
    config['buy_adj'] = pre_adj
    config['pv_adj'] = pre_adj_pv
    config['cart_adj'] = pre_adj_cart
    # config['collect_adj'] = pre_adj_collect
    config['all_behaviors'] = pre_adj_all_behaviors
    config['n_layers'] = args.n_layers
    config['temperature'] = args.temperature
    config['weight_SLL'] = args.weight_ssl
    config['weight_boundary'] = args.weight_boundary
    logger.info("data aleady")

    model = Model(num_users=num_whole_users, num_items=num_whole_items+1, dim_embedding=dim_embedding, dataConfig=config)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_weight)
    total_start = time.time()
    for epoch in range(Epochs):
        losses = 0 
        if time: 
            start = time.time() 
        for batch_data in dataloader:
            batch_users, batch_positive_items_view, batch_positive_items_cart, batch_positive_items_buy = batch_data
            model.forward(batch_users=batch_users.cuda(), whole_items=torch.LongTensor(range(num_whole_items+1)), dropout_ration=dropout_ration)
            model.compute_positive_loss(batch_positive_items_view=batch_positive_items_view.cuda(), \
                                        batch_positive_items_cart=batch_positive_items_cart.cuda(), \
                                        batch_positive_items_buy=batch_positive_items_buy.cuda())
            batch_loss = model.compute_all_loss(weight_negative=args.weight_negative, lambdas=lambdas, alpha = args.alpha)
            losses = losses + batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        if args.v == 1:
            logger.info(' view_cart_buy : Epoch [{}/{}]'.format(epoch+1, Epochs))

        # [120,130,140,150,160,170,180,190,200,210,220,230,240] taobao
        # [230,240,250,260,270,280,290,300,310,320,330,340,350] beibei
        if epoch+1 in [1,10,230,240,250,260,270,280,290,300,310,320,330,340,350]:
            logger.info(' view_cart_buy : Epoch [{}/{}]'.format(epoch + 1, Epochs))

            # scores = []
            results = []
            for step in range(0, int(num_whole_users / batch_size) + 1):
                start = step * batch_size
                end = (step + 1) * batch_size
                if end >= num_whole_users:
                    end = num_whole_users
                likelihood_buy = model.predict(batch_users=torch.LongTensor(range(start, end)).cuda(),
                                               whole_items=torch.LongTensor(range(num_whole_items + 1)).cuda())
                result = evaluation(200, num_whole_items, max_length_test, buy_train[start:end], likelihood_buy,
                                    buy_test[start:end])
                results.append(result)

            count_hr_10 = 0
            count_ndcg_10 = 0
            count_hr_20 = 0
            count_ndcg_20 = 0
            count_hr_100 = 0
            count_ndcg_100 = 0
            count_hr_200 = 0
            count_ndcg_200 = 0
            for result in results:
                x_10, y_10, x_20, y_20, x_100, y_100, x_200, y_200 = result[0], result[1], result[2], result[3], result[
                    4], result[5], result[6], result[7]
                count_hr_10 += x_10
                count_ndcg_10 += y_10
                count_hr_20 += x_20
                count_ndcg_20 += y_20
                count_hr_100 += x_100
                count_ndcg_100 += y_100
                count_hr_200 += x_200
                count_ndcg_200 += y_200

            HR_10 = count_hr_10 / num_whole_users
            NDCG_10 = count_ndcg_10 / num_whole_users
            HR_20 = count_hr_20 / num_whole_users
            NDCG_20 = count_ndcg_20 / num_whole_users
            HR_100 = count_hr_100 / num_whole_users
            NDCG_100 = count_ndcg_100 / num_whole_users
            HR_200 = count_hr_200 / num_whole_users
            NDCG_200 = count_ndcg_200 / num_whole_users

            # scores = []
            results = []

            logger.info(
                ' view_cart_buy | HR@10:{:.4f}, HR@20:{:.4f}, HR@100:{:.4f}, HR@200:{:.4f}'.format(HR_10, HR_20, HR_100,
                                                                                                   HR_200))
            logger.info(
                ' view_cart_buy | NDCG@10:{:.4f}, NDCG@20:{:.4f}, NDCG@100:{:.4f}, NDCG@200:{:.4f}'.format(NDCG_10,
                                                                                                           NDCG_20,
                                                                                                           NDCG_100,
                                                                                                           NDCG_200))

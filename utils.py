import math
import torch
import numpy as np
import scipy.sparse as sp
import pickle
def evaluation(top_N, num_whole_items, max_length_test, train_positive, scores, test_positive):
    test_padding = []
    for i in range(len(train_positive)):
        user_negative_items_test = list(set(range(num_whole_items+1))-set(train_positive[i]))
        user_negative_items_test += [num_whole_items]*(max_length_test - len(user_negative_items_test))
        test_padding.append(user_negative_items_test)
    negative_scores = torch.gather(input=torch.FloatTensor(scores.cpu()), dim=1, index=torch.LongTensor(test_padding))
    topk_indices = torch.gather(input=torch.LongTensor(test_padding), dim=1, index=torch.topk(input=negative_scores, k=top_N, dim=1, largest=True, sorted=True)[1]).tolist()
    count_hr_200 = 0
    count_ndcg_200 = 0

    count_hr_10 = 0
    count_ndcg_10 = 0

    count_hr_20 = 0
    count_ndcg_20 = 0

    count_hr_100 = 0
    count_ndcg_100 = 0

    for user_indices, user_positive_id in zip(topk_indices, test_positive):
        if user_positive_id in user_indices:
            count_hr_200 += 1
            idx_200 = user_indices.index(user_positive_id)+1
            count_ndcg_200 += math.log(2) / math.log(1 + idx_200)
            if idx_200 <= 10:
                count_hr_10 += 1
                idx_10 = idx_200
                count_ndcg_10 += math.log(2) / math.log(1 + idx_10)
            if idx_200 <= 20:
                count_hr_20 += 1
                idx_50 = idx_200
                count_ndcg_20 += math.log(2) / math.log(1 + idx_50)
            if idx_200 <= 100:
                count_hr_100 += 1
                idx_100 = idx_200
                count_ndcg_100 += math.log(2) / math.log(1 + idx_100)
    return count_hr_10, count_ndcg_10, count_hr_20, count_ndcg_20, count_hr_100, count_ndcg_100, count_hr_200, count_ndcg_200

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    rowsum[rowsum == 0] =1
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = torch.tensor([coo.row,coo.col])
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape)

# for 5-8 purchase record user
def get_sparsity(data_name, num):
    def get_sparsity_split():
        split_uids, split_state = [], []
        lines = open('/home/sdc_3_7T/MHFangGPU/lqfCode/DAPM/preprocess/' + data_name + '/sparsity/sparsity.split', 'r').readlines()

        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                split_state.append(line.strip())
            else:
                split_uids.append([int(uid) for uid in line.strip().split(' ')])
        return split_uids[0], split_state[0]

    with open('/home/sdc_3_7T/MHFangGPU/lqfCode/DAPM/preprocess/' + data_name + '/view.pkl', 'rb') as load1:
        view = pickle.load(load1)
    with open('/home/sdc_3_7T/MHFangGPU/lqfCode/DAPM/preprocess/' + data_name + '/cart.pkl', 'rb') as load1:
        cart = pickle.load(load1)
    with open('/home/sdc_3_7T/MHFangGPU/lqfCode/DAPM/preprocess/' + data_name + '/buy_train.pkl', 'rb') as load1:
        buy_train = pickle.load(load1)
    with open('/home/sdc_3_7T/MHFangGPU/lqfCode/DAPM/preprocess/' + data_name + '/buy_test.pkl', 'rb') as load1:
        buy_test = pickle.load(load1)
    tp = []
    uid, _ = get_sparsity_split()
    for i in range(int(num)):
        tp.append(view[uid[i]])
    view = tp
    tp = []
    for i in range(int(num)):
        tp.append(cart[uid[i]])
    cart = tp
    tp = []
    for i in range(int(num)):
        tp.append(buy_train[uid[i]])
    buy_train = tp
    tp = []
    for i in range(int(num)):
        tp.append(buy_test[uid[i]])
    buy_test = tp
    return view,cart,buy_train,buy_test

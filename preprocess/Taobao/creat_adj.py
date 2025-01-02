import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, vstack, hstack
num_whole_users = 48749
num_whole_items = 39493
weight_pv = 1
weight_cart = 4
weight_buy = 1


def normalize_adjacency_matrix(adj_matrix):
    # 计算每个节点的度，即行和
    row_sum = np.array(adj_matrix.sum(1)).flatten() + 1e-7

    # 对度的倒数开平方，避免除以0，使用np.power计算
    d_inv_sqrt = np.power(row_sum, -0.5).astype(float)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0  # 防止度为0时出现无穷大

    # 构造对角的度矩阵 D^(-1/2)
    d_inv_sqrt_matrix = diags(d_inv_sqrt)

    # 对称归一化 A_hat = D^(-1/2) * A * D^(-1/2)
    normalized_adj_matrix = d_inv_sqrt_matrix.dot(adj_matrix).dot(d_inv_sqrt_matrix)

    return normalized_adj_matrix

def create_combined_normalized_matrix(user_item_adj_matrix, num_users=21716, num_items=7977):
    # 创建项目-用户邻接矩阵的转置矩阵
    item_user_adj_matrix = user_item_adj_matrix.transpose()

    # 构建大矩阵，将邻接矩阵放在右上角，转置矩阵放在左下角
    zero_user_block = csr_matrix((num_users, num_users), dtype=np.float32)  # 左上角的零矩阵
    zero_item_block = csr_matrix((num_items, num_items), dtype=np.float32)  # 右下角的零矩阵

    # 上半部分 (用户-项目邻接矩阵)
    upper_block = hstack([zero_user_block, user_item_adj_matrix])

    # 下半部分 (项目-用户邻接矩阵的转置)
    lower_block = hstack([item_user_adj_matrix, zero_item_block])

    # 拼接为一个新的大矩阵
    combined_matrix = vstack([upper_block, lower_block])

    # 对新的大矩阵进行归一化
    normalized_combined_matrix = normalize_adjacency_matrix(combined_matrix)
    return normalized_combined_matrix

def create_adjacency_matrix(data, num_users=21716, num_items=7977):
    # 确保数据合法性，项目ID需要在合理范围内
    max_item_id = num_items - 1  # 项目ID从0开始，所以最大ID为 num_items - 1

    # 创建存储稀疏矩阵所需的行索引、列索引和数据
    row_indices = []
    col_indices = []
    values = []

    # 填充稀疏矩阵的行、列索引和数据
    for user_id, items in enumerate(data):
        for item in items:
            # 检查项目ID是否在有效范围内
            if item < 0 or item > max_item_id:
                raise ValueError(f"项目ID {item} 不在 [0, {max_item_id}] 范围内")

            row_indices.append(user_id)  # 用户ID即为行号
            col_indices.append(item)
            values.append(1.0)  # 将所有值设为 float 类型

    # 创建用户-项目邻接矩阵 (使用float类型)
    user_item_adj_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(num_users, num_items), dtype=float)
    return user_item_adj_matrix

with open('./buy_train.pkl','rb') as load3:
    buy_train = pickle.load(load3)
with open('./view.pkl','rb') as load1:
    view = pickle.load(load1)
with open('./cart.pkl','rb') as load2:
    cart = pickle.load(load2)

R_adj = create_adjacency_matrix(buy_train, num_whole_users, num_whole_items)
R_pv_adj = create_adjacency_matrix(view, num_whole_users, num_whole_items)
R_cart_adj = create_adjacency_matrix(cart, num_whole_users, num_whole_items)
R_all_adj = (R_adj * weight_buy) + (R_pv_adj *weight_pv) + (R_cart_adj *weight_cart)

R = create_combined_normalized_matrix(R_adj, num_whole_users, num_whole_items)
R_pv = create_combined_normalized_matrix(R_pv_adj, num_whole_users, num_whole_items)
R_cart = create_combined_normalized_matrix(R_cart_adj, num_whole_users, num_whole_items)
R_all = create_combined_normalized_matrix(R_all_adj, num_whole_users, num_whole_items)

sp.save_npz("./adj_buy.npz",R.tocoo())
sp.save_npz("./adj_view.npz",R_pv.tocoo())
sp.save_npz("./adj_cart.npz",R_cart.tocoo())
sp.save_npz("./adj_all.npz",R_all.tocoo())

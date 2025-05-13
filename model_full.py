import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, dim_embedding, dataConfig):
        super(Model,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim_embedding = dim_embedding

        self.buy_adj = dataConfig['buy_adj']
        self.pv_adj = dataConfig['pv_adj']
        self.cart_adj = dataConfig['cart_adj']
        self.all_behavior_adj = dataConfig['all_behaviors']
        # self.collect_adj = dataConfig['collect_adj']
        self.n_layers = dataConfig['n_layers']
        self.temperature = dataConfig['temperature']
        self.weight_SLL = dataConfig['weight_SLL']
        self.constraint_boundary = dataConfig['weight_boundary']
        self.embedding_behavior_pv = torch.nn.Parameter(torch.empty(self.dim_embedding))
        self.embedding_behavior_cart = torch.nn.Parameter(torch.empty(self.dim_embedding))
        self.embedding_behavior_buy = torch.nn.Parameter(torch.empty(self.dim_embedding))
        torch.nn.init.normal_(self.embedding_behavior_pv, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.embedding_behavior_cart, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.embedding_behavior_buy, mean=0.0, std=0.01)
        # self.view_mlp = torch.nn.Linear(self.dim_embedding, 1)
        # self.cart_mlp = torch.nn.Linear(self.dim_embedding, 1)
        # self.buy_mlp = torch.nn.Linear(self.dim_embedding, 1)

        self.gate_user_pv = torch.nn.Linear(self.dim_embedding , self.dim_embedding)
        self.gate_item_pv = torch.nn.Linear(self.dim_embedding , self.dim_embedding)
        self.gate_user_cart = torch.nn.Linear(self.dim_embedding , self.dim_embedding)
        self.gate_item_cart = torch.nn.Linear(self.dim_embedding , self.dim_embedding)
        self.gate_user_buy = torch.nn.Linear(self.dim_embedding , self.dim_embedding)
        self.gate_item_buy = torch.nn.Linear(self.dim_embedding , self.dim_embedding)
        # self.gate_user_collect = torch.nn.Linear(self.dim_embedding, self.dim_embedding)
        # self.gate_item_collect = torch.nn.Linear(self.dim_embedding, self.dim_embedding)


        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_embedding)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_embedding)
        self.scale_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=4)
        self.scale_item_1 = torch.nn.Parameter(torch.ones(1, self.num_items))
        self.scale_item_2 = torch.nn.Parameter(torch.ones(1, self.num_items) * 5)
        self.scale_item_3 = torch.nn.Parameter(torch.ones(1, self.num_items) * 5)
        self.scale_item_4 = torch.nn.Parameter(torch.ones(1, self.num_items) * 5)
        self.mask = torch.ones(self.scale_item_1.shape).cuda()
        self.mask[:,-1] =0
        
        torch.nn.init.normal_(self.embedding_user.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.embedding_item.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.scale_user.weight, mean=1, std=0)
        
     
        self.weight_pv = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight_pv, a=-math.sqrt(1.0/self.dim_embedding), b=math.sqrt(1.0/self.dim_embedding))
        self.weight_cart = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight_cart, a=-math.sqrt(1.0 / self.dim_embedding),
                               b=math.sqrt(1.0 / self.dim_embedding))
        self.weight_buy = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight_buy, a=-math.sqrt(1.0 / self.dim_embedding),
                               b=math.sqrt(1.0 / self.dim_embedding))
        # self.weight_collect = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        # torch.nn.init.uniform_(self.weight_collect, a=-math.sqrt(1.0 / self.dim_embedding),
        #                        b=math.sqrt(1.0 / self.dim_embedding))

        self.bn = torch.nn.BatchNorm1d(self.dim_embedding, affine=False)


    def forward(self, batch_users, whole_items, dropout_ration):
        GCN_embeddings = self.get_gcn_embed()

        # compute consistence and uniform loss
        self.ssl_loss = self.get_cons_uni_loss(GCN_embeddings, batch_users)
        # self.ssl_loss = self.contrastive_loss_user(GCN_embeddings, batch_users, self.temperature)

        self.perdict_embedding = [GCN_embeddings['users_all'], GCN_embeddings['users_buy'],
                                  GCN_embeddings['items_all'], GCN_embeddings['items_buy']]

        # self.generate_boundary(GCN_embeddings)
        self.batch_user_scale = self.scale_user(batch_users)

        self.likelihood_all = self.generate_likelihood(GCN_embeddings, batch_users, dropout_ration)

    
    def compute_positive_loss(self, batch_positive_items_view, batch_positive_items_cart, batch_positive_items_buy):

        self.likelihood_positive_view = torch.gather(input=self.likelihood_all[0], dim=1, index=batch_positive_items_view)
        self.likelihood_positive_cart = torch.gather(input=self.likelihood_all[1], dim=1, index=batch_positive_items_cart)
        self.likelihood_positive_buy = torch.gather(input=self.likelihood_all[2], dim=1, index=batch_positive_items_buy)
        # self.likelihood_positive_collect = torch.gather(input=self.likelihood_all[3], dim=1, index=batch_positive_items_collect)

        self.criterion_positive_view = torch.gather(input=self.batch_user_scale[:,0].unsqueeze(1)* self.scale_item_1*self.mask, dim = 1, index=batch_positive_items_view)
        self.criterion_positive_cart = torch.gather(input=self.batch_user_scale[:,1].unsqueeze(1)* self.scale_item_2*self.mask, dim = 1, index=batch_positive_items_cart)
        self.criterion_positive_buy = torch.gather(input=self.batch_user_scale[:,2].unsqueeze(1)* self.scale_item_3*self.mask, dim = 1, index=batch_positive_items_buy)
        # self.criterion_positive_collect = torch.gather(
        #     input=self.batch_user_scale[:, 3].unsqueeze(1) * self.scale_item_4 * self.mask, dim=1,
        #     index=batch_positive_items_collect)

        self.loss_positive_view = torch.sum(torch.pow(torch.nn.functional.relu(self.criterion_positive_view  - self.likelihood_positive_view), 2))
        self.loss_positive_cart = torch.sum(torch.pow(torch.nn.functional.relu(self.criterion_positive_cart - self.likelihood_positive_cart), 2))
        self.loss_positive_buy = torch.sum(torch.pow(torch.nn.functional.relu(self.criterion_positive_buy - self.likelihood_positive_buy), 2))
        # self.loss_positive_collect = torch.sum(
        #     torch.pow(torch.nn.functional.relu(self.criterion_positive_collect - self.likelihood_positive_collect), 2))

    def compute_all_loss(self, weight_negative, lambdas, alpha):
        
        loss_negative_view = torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_all[0] - alpha*self.batch_user_scale[:,0].unsqueeze(1)* self.scale_item_1*self.mask), 2)) - torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_positive_view - alpha* self.criterion_positive_view), 2))
        loss_negative_cart = torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_all[1] - alpha*self.batch_user_scale[:,1].unsqueeze(1)* self.scale_item_2*self.mask), 2)) - torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_positive_cart - alpha* self.criterion_positive_cart), 2))
        loss_negative_buy =  torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_all[2] - alpha*self.batch_user_scale[:,2].unsqueeze(1)* self.scale_item_3*self.mask), 2)) - torch.sum(torch.pow(torch.nn.functional.relu(self.likelihood_positive_buy - alpha* self.criterion_positive_buy), 2))
        # loss_negative_collect = torch.sum(torch.pow(torch.nn.functional.relu(
        #     self.likelihood_all[3] - alpha * self.batch_user_scale[:, 3].unsqueeze(1) * self.scale_item_4 * self.mask),
        #                                         2)) - torch.sum(
        #     torch.pow(torch.nn.functional.relu(self.likelihood_positive_collect - alpha * self.criterion_positive_collect), 2))

        # constraint for upper boundary
        view_c = torch.mul(self.batch_user_scale[:, 0].unsqueeze(1), self.scale_item_1)
        view_c = (-torch.log(view_c + 1e-6)).sum() / 1024
        cart_c = torch.mul(self.batch_user_scale[:, 1].unsqueeze(1), self.scale_item_2)
        cart_c = (-torch.log(cart_c + 1e-6)).sum() / 1024
        purchase_c = torch.mul(self.batch_user_scale[:, 2].unsqueeze(1), self.scale_item_3)
        purchase_c = (-torch.log(purchase_c + 1e-6)).sum() / 1024

        loss_view = self.loss_positive_view + weight_negative * loss_negative_view + view_c * self.constraint_boundary
        loss_cart = self.loss_positive_cart + weight_negative * loss_negative_cart + cart_c * self.constraint_boundary
        loss_buy = self.loss_positive_buy + weight_negative * loss_negative_buy + purchase_c * self.constraint_boundary
        # loss_collect = self.loss_positive_collect + weight_negative * loss_negative_collect

        self.loss = lambdas['view'] * loss_view + lambdas['cart'] * loss_cart + lambdas['buy'] * loss_buy
        return self.loss / 1024 + self.ssl_loss * self.weight_SLL


    def predict(self, batch_users, whole_items):
        self.batch_user_scale = self.scale_user(batch_users)
        user_buy = self.gate_fusion(self.perdict_embedding[0][batch_users],self.perdict_embedding[1][batch_users],
                                    self.gate_user_buy)
        item_buy = self.gate_fusion(self.perdict_embedding[2],self.perdict_embedding[3],
                                    self.gate_item_buy)

        user_buy = torch.nn.functional.normalize(user_buy, dim=-1)

        self.likelihood= F.relu(torch.mm(user_buy * self.weight_buy.T, item_buy.T))
        self.likelihood[:, -1] =0
        return self.likelihood/(F.relu(self.scale_item_3)+1e-4)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = torch.tensor([coo.row,coo.col])
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape)

    def get_gcn_embed(self):

        GCN_embeddings = {}

        users_pv, items_pv = self.lightgcn_propagate(self.embedding_user.weight + self.embedding_behavior_pv,
                                                     self.embedding_item.weight + self.embedding_behavior_pv, self.pv_adj,
                                                     self.n_layers)
        users_cart, items_cart = self.lightgcn_propagate(self.embedding_user.weight + self.embedding_behavior_cart,
                                                         self.embedding_item.weight + self.embedding_behavior_cart, self.cart_adj,
                                                     self.n_layers)
        users_buy, items_buy = self.lightgcn_propagate(self.embedding_user.weight + self.embedding_behavior_buy,
                                                       self.embedding_item.weight + self.embedding_behavior_buy, self.buy_adj,
                                                     self.n_layers)
        users_all, items_all = self.lightgcn_propagate(self.embedding_user.weight, self.embedding_item.weight, self.all_behavior_adj,
                                                     self.n_layers)
        # users_collect, items_collect = self.lightgcn_propagate(self.embedding_user.weight, self.embedding_item.weight,
        #                                                self.collect_adj,
        #                                                self.n_layers)

        GCN_embeddings['users_pv'] = users_pv
        GCN_embeddings['items_pv'] = items_pv
        GCN_embeddings['users_cart'] = users_cart
        GCN_embeddings['items_cart'] = items_cart
        GCN_embeddings['users_buy'] = users_buy
        GCN_embeddings['items_buy'] = items_buy
        GCN_embeddings['users_all'] = users_all
        GCN_embeddings['items_all'] = items_all
        # GCN_embeddings['users_collect'] = users_collect
        # GCN_embeddings['items_collect'] = items_collect
        return GCN_embeddings

    def concatenate_matrices(self, A, B):
        """
        Concatenates two matrices A (M, d) and B (N, d) to form a new tensor C (M, N, 2d) using PyTorch.

        Args:
            A (torch.Tensor): Matrix of shape (M, d)
            B (torch.Tensor): Matrix of shape (N, d)

        Returns:
            torch.Tensor: Tensor of shape (M, N, 2d) with A and B concatenated along the last dimension.
        """
        # Expand A and B to add an additional dimension for broadcasting
        A_expanded = A.unsqueeze(1)  # Shape (M, 1, d)
        B_expanded = B.unsqueeze(0)  # Shape (1, N, d)

        # Tile A_expanded along the second dimension and B_expanded along the first dimension
        A_tiled = A_expanded.expand(-1, B.shape[0], -1)  # Shape (M, N, d)
        B_tiled = B_expanded.expand(A.shape[0], -1, -1)  # Shape (M, N, d)

        # Concatenate along the last axis to form a tensor of shape (M, N, 2d)
        C = torch.cat((A_tiled, B_tiled), dim=-1)  # Shape (M, N, 2d)

        return C

    def gate_fusion(self, User_side_one, User_side_two ,gate_layer):
        # Compute the gating weights (between 0 and 1) for each element
        gate_weights = torch.sigmoid(gate_layer(User_side_one))  # Shape (M, N)
        #gate_weights = torch.sigmoid(gate_layer(torch.cat([User_side_one, User_side_two], dim=1)))  
        # Fuse A and B using the gating weights
        user_emb = gate_weights * User_side_one + (1 - gate_weights) * User_side_two

        return user_emb

    def generate_likelihood(self, GCN_embeddings, batch_users, dropout_ration):

        batch_user_pv = self.gate_fusion(GCN_embeddings['users_all'][batch_users], GCN_embeddings['users_pv'][batch_users],
                                   self.gate_user_pv)
        item_pv = self.gate_fusion(GCN_embeddings['items_all'], GCN_embeddings['items_pv'], self.gate_item_pv)
        batch_user_cart = self.gate_fusion(GCN_embeddings['users_all'][batch_users], GCN_embeddings['users_cart'][batch_users],
                                     self.gate_user_cart)
        item_cart = self.gate_fusion(GCN_embeddings['items_all'], GCN_embeddings['items_cart'], self.gate_item_cart)
        batch_user_buy = self.gate_fusion(GCN_embeddings['users_all'][batch_users], GCN_embeddings['users_buy'][batch_users],
                                    self.gate_user_buy)
        item_buy = self.gate_fusion(GCN_embeddings['items_all'], GCN_embeddings['items_buy'], self.gate_item_buy)

        # batch_user_collect = self.gate_fusion(GCN_embeddings['users_collect'][batch_users],
        #                                   GCN_embeddings['users_buy'][batch_users],
        #                                   self.gate_user_collect)
        # item_collect = self.gate_fusion(GCN_embeddings['items_collect'], GCN_embeddings['items_buy'], self.gate_item_collect)

        batch_user_pv = torch.nn.functional.normalize(torch.nn.functional.dropout(batch_user_pv, p=dropout_ration, training=True), dim=-1)
        batch_user_cart = torch.nn.functional.normalize(torch.nn.functional.dropout(batch_user_cart, p=dropout_ration, training=True), dim=-1)
        batch_user_buy = torch.nn.functional.normalize(torch.nn.functional.dropout(batch_user_buy, p=dropout_ration, training=True), dim=-1)

        # batch_user_collect = torch.nn.functional.normalize(torch.nn.functional.dropout(batch_user_collect, p=dropout_ration, training=True), dim=-1)

        likelihood_pv = torch.nn.functional.relu(torch.mm(batch_user_pv * self.weight_pv.T, item_pv.T)).clone()
        likelihood_pv[:, -1] = 0
        likelihood_cart = torch.nn.functional.relu(torch.mm(batch_user_cart * self.weight_cart.T, item_cart.T)).clone()
        likelihood_cart[:, -1] = 0
        likelihood_buy = torch.nn.functional.relu(torch.mm(batch_user_buy * self.weight_buy.T, item_buy.T)).clone()
        likelihood_buy[:, -1] = 0
        # likelihood_collect = torch.nn.functional.relu(torch.mm(batch_user_collect * self.weight_collect.T, item_collect.T)).clone()
        # likelihood_collect[:, -1] = 0
        # likelihood_all = [likelihood_pv, likelihood_cart, likelihood_buy, likelihood_collect]
        likelihood_all = [likelihood_pv, likelihood_cart, likelihood_buy]
        return likelihood_all

    def generate_boundary(self, GCN_embeddings):
        boundary_view_user = self.view_mlp(GCN_embeddings['users_pv'])
        boundary_cart_user = self.cart_mlp(GCN_embeddings['users_cart'])
        boundary_buy_user = self.buy_mlp(GCN_embeddings['users_buy'])
        boundary_view_item = self.view_mlp(GCN_embeddings['items_pv']).transpose(0, 1)
        boundary_cart_item = self.cart_mlp(GCN_embeddings['items_pv']).transpose(0, 1)
        boundary_buy_item = self.buy_mlp(GCN_embeddings['items_pv']).transpose(0, 1)
        self.scale_user = torch.cat([boundary_view_user, boundary_cart_user, boundary_buy_user], dim=1)
        self.scale_item_1 = boundary_view_item
        self.scale_item_2 = boundary_cart_item
        self.scale_item_3 = boundary_buy_item

    def contrastive_loss_v3(self, A, B, temperature=0.2):
        """
        A: Tensor of shape (M, D), where M is the number of users and D is the dimension of the embedding.
        B: Tensor of shape (M, D), where M is the number of users and D is the dimension of the embedding.
        temperature: Temperature scaling factor for the contrastive loss.

        Returns:
        loss: A scalar representing the contrastive loss.
        """
        # Normalize embeddings along the feature dimension
        A = F.normalize(A, dim=1)
        B = F.normalize(B, dim=1)

        batch_size = A.shape[0]

        # Compute similarity matrices (M, M)
        # A-B similarity
        similarity_A_B = torch.matmul(A, B.T) / temperature
        # # A-A similarity
        # similarity_A_A = torch.matmul(A, A.T) / temperature

        # Positive pairs: A_i and B_i
        pos_sim = torch.exp(torch.diag(similarity_A_B))

        # For negative pairs:
        # - A_i and B_j (j != i)
        # - A_i and A_j (j != i)

        # Mask to ignore self-similarities (diagonal elements)
        mask = torch.eye(batch_size, dtype=torch.bool).to(A.device)

        # Compute negative similarities for A_i with other B_j (j != i)
        neg_sim_A_B = torch.exp(similarity_A_B.masked_fill(mask, float('-inf')))
        neg_sim_A_B_sum = neg_sim_A_B.sum(dim=1)  # Sum over all other B_j

        # # Compute negative similarities for A_i with other A_j (j != i)
        # neg_sim_A_A = torch.exp(similarity_A_A.masked_fill(mask, float('-inf')))
        # neg_sim_A_A_sum = neg_sim_A_A.sum(dim=1)  # Sum over all other A_j

        # Combine the negative similarities
        # neg_sim_sum = neg_sim_A_B_sum + neg_sim_A_A_sum
        neg_sim_sum = neg_sim_A_B_sum

        # Contrastive loss: -log(positive / (negative_sum))
        loss = -torch.log(pos_sim / neg_sim_sum).mean()

        return loss

    def contrastive_loss_v2(self, A, temperature=0.2):
        """
        A: Tensor of shape (M, D), where M is the number of samples and D is the dimension of the embedding.
        temperature: Temperature scaling factor for the contrastive loss.

        Returns:
        loss: A scalar representing the contrastive loss.
        """
        # Normalize embeddings along the feature dimension
        A = F.normalize(A, dim=1)

        batch_size = A.shape[0]

        # Compute similarity matrix (M, M)
        # The similarity is computed as the dot product between each pair of vectors
        similarity_matrix = torch.matmul(A, A.T) / temperature

        # Create mask to ignore self-similarities (positive samples)
        mask = torch.eye(batch_size, dtype=torch.bool).to(A.device)

        # For positive pairs, similarity is 1
        pos_sim = torch.exp(torch.ones(batch_size).to(A.device) / temperature)  

        # For negative pairs, sum the similarities of all A_i with A_j (i != j)
        neg_sim = torch.exp(similarity_matrix.masked_fill(mask, float('-inf')))  
        neg_sim_sum = neg_sim.sum(dim=1)  

        # Contrastive loss is the negative log of the positive similarity over the negative similarities
        loss = -torch.log(pos_sim / neg_sim_sum).mean()

        return loss

    def contrastive_loss_user(self, embeddings, batch_users, temperature):

        contrastive_loss_b2v = self.contrastive_loss_v3(embeddings['users_buy'][batch_users], embeddings['users_pv'][batch_users], temperature)
        contrastive_loss_b2c = self.contrastive_loss_v3(embeddings['users_buy'][batch_users], embeddings['users_cart'][batch_users], temperature)
        contrastive_loss_b2all = self.contrastive_loss_v3(embeddings['users_buy'][batch_users], embeddings['users_all'][batch_users], temperature)
        contrastive_loss_b2b = self.contrastive_loss_v2(embeddings['users_buy'][batch_users], temperature)
        # contrastive_loss_b2collect = self.contrastive_loss_v3(embeddings['users_buy'][batch_users],
        #                                                 embeddings['users_collect'][batch_users], temperature)
        contrastive_loss = contrastive_loss_b2v + contrastive_loss_b2c  +contrastive_loss_b2b + contrastive_loss_b2all
        return contrastive_loss


    def lightgcn_propagate(self, user_emb, item_emb, adj_matrix, num_layers=3):
       
        adj_matrix = self.convert_to_torch_sparse(adj_matrix).cuda()
       
        all_emb = torch.cat([user_emb, item_emb[:-1,:]], dim=0)  

       
        all_layer_embs = [all_emb]

       
        for layer in range(num_layers):
           
            all_emb = torch.sparse.mm(adj_matrix, all_emb)  

            
            all_layer_embs.append(all_emb)

       
        final_emb = sum(all_layer_embs) / (num_layers + 1)

       
        updated_user_emb = final_emb[:user_emb.shape[0], :]  
        updated_item_emb = final_emb[user_emb.shape[0]:, :]  
        token_embedding = torch.zeros([1, self.dim_embedding]).cuda()
       
        updated_item_emb = torch.cat([updated_item_emb, token_embedding], 0)

        return updated_user_emb, updated_item_emb

    def convert_to_torch_sparse(self, sparse_matrix):
        
        if isinstance(sparse_matrix, csr_matrix):
            sparse_matrix = sparse_matrix.tocoo()  

       
        indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
        values = sparse_matrix.data
        size = sparse_matrix.shape

        
        torch_sparse_matrix = torch.sparse_coo_tensor(indices, values, size).to(torch.float32)

        return torch_sparse_matrix

    def get_cons_uni_loss(self, embeddings, batch_users):

        consistence_loss_b2v = self.bt(embeddings['users_buy'][batch_users],
                                                        embeddings['users_pv'][batch_users])
        consistence_loss_b2c = self.bt(embeddings['users_buy'][batch_users],
                                                        embeddings['users_cart'][batch_users])
        consistence_loss_b2all = self.bt(embeddings['users_buy'][batch_users],
                                                          embeddings['users_all'][batch_users])

        uniform_loss = (self.uniformity(embeddings['users_buy'][batch_users]) +
                        self.uniformity(embeddings['users_pv'][batch_users]) +
                        self.uniformity(embeddings['users_cart'][batch_users]) +
                        self.uniformity(embeddings['users_all'][batch_users]))

        ssl_loss = (consistence_loss_b2v + consistence_loss_b2c + consistence_loss_b2all) / 3 + uniform_loss / 4


        return ssl_loss

    # compute consistence loss
    def bt(self, user_e, item_e):
        # c = self.bn(user_e).T @ self.bn(item_e)
        # c = torch.matmul(self.bn(user_e), self.bn(item_e).T)
        c = torch.matmul(F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1).T)
        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div_(user_e.size()[0])
        #off_diag = self.off_diagonal(c).pow_(2).sum().div(self.dim_embedding)
        # bt = on_diag + 0.1 * off_diag
        bt = on_diag 
        return bt

    # compute uniform loss
    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
   








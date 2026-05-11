import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers
from layers import GCN, AvgReadout 
import tqdm
import numpy as np
import dgl
from utils import Calbound
from sklearn.decomposition import PCA
from layers import Attentivemod
from tools import *


class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, i, dropedge_rate, sparse, act):
        super(ATT_learner, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentivemod.Attentive(isize))

        self.non_linearity = 'relu'
        self.i = i
        self.sparse = sparse
        self.act = act
        self.dropedge_rate = dropedge_rate

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)

        return h

    def forward(self, features):
       
        embeddings = self.internal_forward(features)

        return embeddings

    def graph_process(self, k, embeddings):
        if self.sparse:
            rows, cols, values = knn_fast(embeddings, k, 100)      
            values[torch.isnan(values)] = 0  
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            values_ = F.dropout(values_, p=self.dropedge_rate, training=self.training)
            # learned_adj = dgl.graph((rows_, cols_), num_nodes = embeddings.shape[0],device='cuda')
            # learned_adj = learned_adj.to_dense()
            num_nodes = embeddings.shape[0]
            learned_adj = torch.zeros((num_nodes, num_nodes), device='cuda')  # 创建稠密矩阵
            learned_adj[rows_, cols_] = values_ 

            return learned_adj
        else:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, k + 1)
            similarities = symmetrize(similarities)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            learned_adj = normalize(similarities, 'sym')
            learned_adj = F.dropout(learned_adj, p=self.dropedge_rate, training=self.training)

            return learned_adj



class combineprompt(nn.Module):#对输入的两个图嵌入加权组合
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        
        # weight = F.softmax(self.weight, dim=1)
        # print("weight",weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


class matrixsquare(nn.Module):
    def __init__(self):
        super(matrixsquare,self).__init__()

    def forward(self, matrix):
        if matrix.is_sparse:
            square = torch.sparse.mm(matrix, matrix)
        else:
            square = torch.mm(matrix, matrix)
        return square

class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation,sample,num_layers_num,p,type):
        super(PrePrompt, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h,num_layers_num,p)
        self.read = AvgReadout()
        self.prompttype = type

        self.pretext1 = textprompt(n_in,type)
        self.pretext2 = textprompt(n_in,type)
        self.pretext3 = textprompt(n_in,type)
        self.pretext4 = textprompt(n_in,type)
        self.pretext5 = textprompt(n_in,type)
        self.pretext6 = textprompt(n_in,type)
        self.pretext7 = textprompt(n_in,type)
        self.pretext8 = textprompt(n_in,type)
        self.pretext9 = textprompt(n_in,type)

        self.sumtext = textprompt(n_in,type)
        #self.structureprompt = textprompt(n_h,type)
        
        
        self.texttoken1 = textprompt(n_h,type)
        self.texttoken2 = textprompt(n_h,type)
        self.texttoken3 = textprompt(n_h,type)
        self.texttoken4 = textprompt(n_h,type)
        self.texttoken5 = textprompt(n_h,type)
        self.texttoken6 = textprompt(n_h,type)
        self.texttoken7 = textprompt(n_h,type)
        self.texttoken8 = textprompt(n_h,type)
        self.texttoken9 = textprompt(n_h,type)

        # self.sample = torch.tensor(sample,dtype=int).cuda()
        # print("sample",self.sample)
        self.learner = ATT_learner(2, 50, 6, 0.5, sparse = True, act = 'relu') # where to fine-tune

        self.negative_sample = torch.tensor(sample,dtype=int).cuda()

        # self.comparelosses = Calbound.calc_lower_bound()
        self.matrixsquare = matrixsquare()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, seq1,seq2,seq3,seq4,seq5,seq6,seq7,adj1,adj2,adj3,adj4,adj5,adj6,adj7,
                sparse, msk, samp_bias1, samp_bias2):

        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        seq5 = torch.squeeze(seq5,0)
        seq6 = torch.squeeze(seq6,0)
        seq7 = torch.squeeze(seq7,0)

        preseq1 = self.pretext1(seq1)
        preseq2 = self.pretext2(seq2)
        preseq3 = self.pretext3(seq3)
        preseq4 = self.pretext4(seq4)
        preseq5 = self.pretext5(seq5)
        preseq6 = self.pretext6(seq6)
        preseq7 = self.pretext7(seq7)

        preseq1 = self.sumtext(preseq1)
        preseq2 = self.sumtext(preseq2)
        preseq3 = self.sumtext(preseq3)
        preseq4 = self.sumtext(preseq4)
        preseq5 = self.sumtext(preseq5)
        preseq6 = self.sumtext(preseq6)
        preseq7 = self.sumtext(preseq7)

        reseq1 = torch.sparse.mm(self.matrixsquare(adj1),preseq1)
        reseq1 = torch.cat((preseq1, reseq1), dim = 1)
        reseq2 = torch.sparse.mm(self.matrixsquare(adj2),preseq2)
        reseq2 = torch.cat((preseq2, reseq2), dim = 1)
        reseq3 = torch.sparse.mm(self.matrixsquare(adj3),preseq3)
        reseq3 = torch.cat((preseq3, reseq3), dim = 1)
        reseq4 = torch.sparse.mm(self.matrixsquare(adj4),preseq4)
        reseq4 = torch.cat((preseq4, reseq4), dim = 1)
        reseq5 = torch.sparse.mm(self.matrixsquare(adj5),preseq5)
        reseq5 = torch.cat((preseq5, reseq5), dim = 1)
        reseq6 = torch.sparse.mm(self.matrixsquare(adj6),preseq6)
        reseq6 = torch.cat((preseq6, reseq6), dim = 1)
        reseq7 = torch.sparse.mm(self.matrixsquare(adj7),preseq7)
        reseq7 = torch.cat((preseq7, reseq7), dim = 1)

        refinedadj1 = self.learner.graph_process(30, reseq1).to('cuda')
        refinedadj2 = self.learner.graph_process(35, reseq2).to('cuda')
        refinedadj3 = self.learner.graph_process(21, reseq3).to('cuda')
        refinedadj4 = self.learner.graph_process(13, reseq4).to('cuda')
        refinedadj5 = self.learner.graph_process(15, reseq5).to('cuda')
        refinedadj6 = self.learner.graph_process(66, reseq6).to('cuda')
        refinedadj7 = self.learner.graph_process(30, reseq7).to('cuda')
        # 30, 35, 21, 66, 15, 66, 30, 13

        num1, _ = refinedadj1.size()
        num2, _ = refinedadj2.size()
        num3, _ = refinedadj3.size()
        num4, _ = refinedadj4.size()
        num5, _ = refinedadj5.size()
        num6, _ = refinedadj6.size()
        num7, _ = refinedadj7.size()

        pos_eye1 = torch.eye(num1).to(refinedadj1.device)
        pos_eye2 = torch.eye(num2).to(refinedadj1.device)
        pos_eye3 = torch.eye(num3).to(refinedadj1.device)
        pos_eye4 = torch.eye(num4).to(refinedadj1.device)
        pos_eye5 = torch.eye(num5).to(refinedadj1.device)
        pos_eye6 = torch.eye(num6).to(refinedadj1.device)
        pos_eye7 = torch.eye(num7).to(refinedadj1.device)

        prelogits1 = self.lp(self.gcn,preseq1,refinedadj1,sparse)
        prelogits2 = self.lp(self.gcn,preseq2,refinedadj2,sparse)
        prelogits3 = self.lp(self.gcn,preseq3,refinedadj3,sparse)
        prelogits4 = self.lp(self.gcn,preseq4,refinedadj4,sparse)
        prelogits5 = self.lp(self.gcn,preseq5,refinedadj5,sparse)
        prelogits6 = self.lp(self.gcn,preseq6,refinedadj6,sparse)
        prelogits7 = self.lp(self.gcn,preseq7,refinedadj7,sparse)

        orilogits1 = self.texttoken1(prelogits1)
        orilogits2 = self.texttoken2(prelogits2)
        orilogits3 = self.texttoken3(prelogits3)
        orilogits4 = self.texttoken4(prelogits4)
        orilogits5 = self.texttoken5(prelogits5)
        orilogits6 = self.texttoken6(prelogits6)
        orilogits7 = self.texttoken7(prelogits7)   

        logits1 = self.lp(self.gcn,seq1,adj1,sparse)
        logits2 = self.lp(self.gcn,seq2,adj2,sparse)
        logits3 = self.lp(self.gcn,seq3,adj3,sparse)
        logits4 = self.lp(self.gcn,seq4,adj4,sparse)
        logits5 = self.lp(self.gcn,seq5,adj5,sparse)
        logits6 = self.lp(self.gcn,seq6,adj6,sparse)
        logits7 = self.lp(self.gcn,seq7,adj7,sparse)

        lploss = Calbound.calc_lower_bound(orilogits1, logits1, pos_eye1)+Calbound.calc_lower_bound(orilogits2, logits2, pos_eye2)+Calbound.calc_lower_bound(orilogits3, logits3, pos_eye3)+Calbound.calc_lower_bound(orilogits4, logits4, pos_eye4)+Calbound.calc_lower_bound(orilogits5, logits5, pos_eye5)+Calbound.calc_lower_bound(orilogits6, logits6, pos_eye6)+Calbound.calc_lower_bound(orilogits7, logits7, pos_eye7)
        lploss.requires_grad_(True)
        return lploss
  
    def embedding(self, seq1,seq2,seq3,seq4,seq5,seq6,seq7,
                adj1,adj2,adj3,adj4,adj5,adj6,adj7,
                sparse, msk, samp_bias1, samp_bias2):
        
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        seq5 = torch.squeeze(seq5,0)
        seq6 = torch.squeeze(seq6,0)
        seq7 = torch.squeeze(seq7,0)


        preseq1 = self.pretext1(seq1)
        preseq2 = self.pretext2(seq2)
        preseq3 = self.pretext3(seq3)
        preseq4 = self.pretext4(seq4)
        preseq5 = self.pretext5(seq5)
        preseq6 = self.pretext6(seq6)
        preseq7 = self.pretext7(seq7)
    

        prelogits1 = self.lp(self.gcn,preseq1,adj1,sparse)
        prelogits2 = self.lp(self.gcn,preseq2,adj2,sparse)
        prelogits3 = self.lp(self.gcn,preseq3,adj3,sparse)
        prelogits4 = self.lp(self.gcn,preseq4,adj4,sparse)
        prelogits5 = self.lp(self.gcn,preseq5,adj5,sparse)
        prelogits6 = self.lp(self.gcn,preseq6,adj6,sparse)
        prelogits7 = self.lp(self.gcn,preseq7,adj7,sparse)

        return prelogits1.detach(),prelogits2.detach(),prelogits3.detach(),prelogits4.detach(),prelogits5.detach(),prelogits6.detach(),prelogits7.detach()

    def embed(self, seq, adj, sparse, msk,LP):#接受一个序列和邻接矩阵，使用图卷积计算输出
        
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class textprompt(nn.Module):#对图嵌入进行处理
    def __init__(self,hid_units,type):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        #一个可学习的参数
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype =type
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)
    def forward(self, graph_embedding):
        # print("weight",self.weight)
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding=self.weight * graph_embedding

        return graph_embedding


def mygather(feature, index):#从给定的特征张量中根据指定的索引提取对应的行，并且对提取的结果进行适当的形状转换
    # print("index",index)
    # print("indexsize",index.shape)  
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)

    # print("feature",feature)
    # print("featuresize",feature.shape)
    # print("index",index)
    # print("indexsize",index.shape)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))



def compareloss(feature,tuples,temperature):#计算一个与特征（feature）相关的损失值(对比损失)
    # print("feature",feature)
    # print("tuple",tuples)
    # feature=feature.cpu()
    # tuples = tuples.cpu()
    h_tuples=mygather(feature,tuples)
    # print("tuples",h_tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    temp=temp.cuda()
    h_i = mygather(feature, temp)
    # print("h_i",h_i)
    # print("h_tuple",h_tuples)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj,n):#从给定的稀疏邻接矩阵中为每个节点采样一组连接节点和一组非连接节点，结果存储在一个二维数组中
    nodenum=adj.shape[0]
    #indptr提示的是非零数在稀疏矩阵中的位置信息。indices是具体的连接边的一个节点的编号
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    # print("#############")
    # print("start sampling disconnected tuples")
    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)


def pca_compression(seq,k):#使用 PCA 方法将输入的高维数据 seq 降维到指定维度 k
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):# 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
    res = np.zeros_like(seq)
    # 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
 
    return res
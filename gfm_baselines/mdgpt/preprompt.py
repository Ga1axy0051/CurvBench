import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp,GcnLayers
from layers import GCN, AvgReadout 
import tqdm
import numpy as np
from sklearn.decomposition import PCA

def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    #print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    # 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
 
    return res

class combineprompt(nn.Module):
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
        
        self.texttoken1 = textprompt(n_h,type)
        self.texttoken2 = textprompt(n_h,type)
        self.texttoken3 = textprompt(n_h,type)
        self.texttoken4 = textprompt(n_h,type)
        self.texttoken5 = textprompt(n_h,type)
        self.texttoken6 = textprompt(n_h,type)
        # self.sample = torch.tensor(sample,dtype=int).cuda()
        # print("sample",self.sample)

        self.negative_sample = torch.tensor(sample,dtype=int).cuda()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, seq1,seq2,seq3,seq4,seq5,seq6,adj1,adj2,adj3,adj4,adj5,adj6,
                sparse, msk, samp_bias1, samp_bias2):
        # seq1=seq1[:2485,]
        # seq3=seq3[:19717,]
        # seq2=seq2[:2110,]
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        seq5 = torch.squeeze(seq5,0)
        seq6 = torch.squeeze(seq6,0)



        preseq1 = self.pretext1(seq1)
        preseq2 = self.pretext2(seq2)
        preseq3 = self.pretext3(seq3)
        preseq4 = self.pretext4(seq4)
        preseq5 = self.pretext5(seq5)
        preseq6 = self.pretext6(seq6)
        # print(seq1.shape)
        # print(seq2.shape)
        # logits1 = self.lp(self.gcn,seq1,adj1,sparse)
        # logits2 = self.lp(self.gcn,seq2,adj2,sparse)
        # logits3 = self.lp(self.gcn,seq3,adj3,sparse)
        # logits4 = self.lp(self.gcn,seq4,adj4,sparse)
        

        prelogits1 = self.lp(self.gcn,preseq1,adj1,sparse)
        prelogits2 = self.lp(self.gcn,preseq2,adj2,sparse)
        prelogits3 = self.lp(self.gcn,preseq3,adj3,sparse)
        prelogits4 = self.lp(self.gcn,preseq4,adj4,sparse)
        prelogits5 = self.lp(self.gcn,preseq5,adj5,sparse)
        prelogits6 = self.lp(self.gcn,preseq6,adj6,sparse)

        # # print(seq1.shape)

        # logits111 = self.texttoken1(logits1)
        # logits222 = self.texttoken2(logits2)
        # logits333 = self.texttoken3(logits3)
        # logits444 = self.texttoken4(logits4)


        # logits11 = 0.1*prelogits1+logits111
        # logits22 = 0.1*prelogits2+logits222
        # logits33 = 0.1*prelogits3+logits333
        # logits44 = 0.1*prelogits4+logits444

        



        logits = torch.cat((prelogits1,prelogits2,prelogits3,prelogits4,prelogits5,prelogits6),dim=0)
        # print("logits1=",logits1)
        # print(logits.shape)
        # print(self.negative_sample.shape)
        lploss = compareloss(logits,self.negative_sample,temperature=1)
        lploss.requires_grad_(True)
        
        # print("promptdgi",self.dgi.prompt)
        # print("gcn",self.gcn.fc.weight)
        # print("promptLP",self.lp.prompt)

        # print("dgiloss",dgiloss)
        # print("graphcl",graphcledgeloss)
        # print("lploss",'{:.8f}'.format(lploss)) 
        # print("a1=", self.a1, "a2=", self.a2,"a3=",self.a3)
        return lploss
    

    def embedding(self, seq1,seq2,seq3,seq4,seq5,seq6,adj1,adj2,adj3,adj4,adj5,adj6,
                sparse, msk, samp_bias1, samp_bias2):
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        seq5 = torch.squeeze(seq5,0)
        seq6 = torch.squeeze(seq6,0)


        preseq1 = self.pretext1(seq1)
        preseq2 = self.pretext2(seq2)
        preseq3 = self.pretext3(seq3)
        preseq4 = self.pretext4(seq4)
        preseq5 = self.pretext5(seq5)
        preseq6 = self.pretext6(seq6)
        # print(seq1.shape)
        # print(seq2.shape)
        # logits1 = self.lp(self.gcn,seq1,adj1,sparse)
        # logits2 = self.lp(self.gcn,seq2,adj2,sparse)
        # logits3 = self.lp(self.gcn,seq3,adj3,sparse)
        # logits4 = self.lp(self.gcn,seq4,adj4,sparse)
        

        prelogits1 = self.lp(self.gcn,preseq1,adj1,sparse)
        prelogits2 = self.lp(self.gcn,preseq2,adj2,sparse)
        prelogits3 = self.lp(self.gcn,preseq3,adj3,sparse)
        prelogits4 = self.lp(self.gcn,preseq4,adj4,sparse)
        prelogits5 = self.lp(self.gcn,preseq5,adj5,sparse)
        prelogits6 = self.lp(self.gcn,preseq6,adj6,sparse)
        # # print(seq1.shape)

        # logits111 = self.texttoken1(logits1)
        # logits222 = self.texttoken2(logits2)
        # logits333 = self.texttoken3(logits3)
        # logits444 = self.texttoken4(logits4)


        # logits11 = 0.1*prelogits1+logits111
        # logits22 = 0.1*prelogits2+logits222
        # logits33 = 0.1*prelogits3+logits333
        # logits44 = 0.1*prelogits4+logits444

        


        return prelogits1.detach(),prelogits2.detach(),prelogits3.detach(),prelogits4.detach(),prelogits5.detach(),prelogits6.detach() 

    def embed(self, seq, adj, sparse, msk,LP):
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class textprompt(nn.Module):
    def __init__(self,hid_units,type):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
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


def mygather(feature, index):
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


def compareloss(feature,tuples,temperature):
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


def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
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



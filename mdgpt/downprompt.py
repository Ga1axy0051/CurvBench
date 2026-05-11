import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL
from layers import GCN, AvgReadout
import torch_scatter
class prefeatureprompt(nn.Module):
    def __init__(self,texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,texttoken6,dim,type:str,head_num=8):
        super(prefeatureprompt, self).__init__()
        self.precomposedfeature = composedtoken(texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,texttoken6,type)
        self.preopenfeature = downstreamprompt(dim)
        self.combineprompt = combineprompt()
    def forward(self,seq):
        seq1 = self.precomposedfeature(seq)
        seq2 = self.preopenfeature(seq)
        # print(seq1[0])
        # print(seq2[0])
        # print(self.preopenfeature.weight)
        # ret = seq2
        ret =   self.combineprompt(seq1 ,seq2) 
        return ret

        




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        # print(embed_dim)
        # print(num_heads)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # print(self.query.weight)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x


        x = torch.squeeze(x)

        x = torch.sum(x,dim=0)


    
        return x

class composedtoken(nn.Module):
    def __init__(self,texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,texttoken6,type:str,head_num=8):
        super(composedtoken, self).__init__()
        # print(texttoken1.shape)
        self.texttoken = torch.cat((texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,texttoken6),dim=0)
        # print(self.texttoken.shape)
        self.prompt = weighted_prompt(6).cuda()
        self.type = type

    def forward(self,seq):
        # print(seq.shape)
        
        texttoken = self.prompt(self.texttoken)

        
        # print(texttoken.shape)
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets





class downprompt(nn.Module):
    def __init__(self,token1,token2,token3,token4,token5,token6,pretoken1,pretoken2,pretoken3,pretoken4,pretoken5,pretoken6, ft_in, nb_classes, type,feature_dim):
        super(downprompt, self).__init__()
        # self.prompt1 = prompt1
        # self.prompt2 = prompt2
        # self.prompt3 = prompt3
        self.downprompt = downstreamprompt(ft_in)
        self.composedprompt = composedtoken(token1,token2,token3,token4,token5,token6,type=type)
        self.prefeature = prefeatureprompt(pretoken1,pretoken2,pretoken3,pretoken4,pretoken5,pretoken6,dim=feature_dim,type=type,head_num=4)
        self.combineprompt1 = combineprompt()
        self.combineprompt2 = combineprompt()

        self.nb_classes = nb_classes


        self.leakyrelu = nn.ELU()
        # self.prompt = prompt3
        # self.prompt = prompt3
        # self.a = nn.Parameter(torch.FloatTensor(1, 3), requires_grad=True).cuda()
        # self.reset_parameters()




        self.one = torch.ones(1,ft_in).cuda()

        # for x in range(0, nb_classes):
        #     if labels[x].item() == 0:
        #         self.aveemb0 = feature[index[x].item()]
        #     if labels[x].item() == 1:
        #         self.aveemb1 = feature[index[x].item()]
        #     if labels[x].item() == 2:
        #         self.aveemb2 = feature[index[x].item()]
        #     if labels[x].item() == 3:
        #         self.aveemb3 = feature[index[x].item()]
        #     if labels[x].item() == 4:
        #         self.aveemb4 = feature[index[x].item()]
        #     if labels[x].item() == 5:
        #         self.aveemb5 = feature[index[x].item()]
        #     if labels[x].item() == 6:
        #         self.aveemb6 = feature[index[x].item()]
        
        
        self.ave = torch.FloatTensor(nb_classes,ft_in).cuda()
        # print("avesize",self.ave.size(),"ave",self.ave)

        # emb0 = torch.zeros(1, 1, 512).cuda()
        # emb1 = torch.zeros(1, 1, 512).cuda()
        # emb2 = torch.zeros(1, 1, 512).cuda()
        # emb3 = torch.zeros(1, 1, 512).cuda()
        # emb4 = torch.zeros(1, 1, 512).cuda()
        # emb5 = torch.zeros(1, 1, 512).cuda()
        # emb6 = torch.zeros(1, 1, 512).cuda()

        # for x in range(0, embs.shape[0]):
        #     if lbls[x] == 0:
        #         emb0 =torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 1:
        #         emb1 = torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 2:
        #         emb2 = torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 3:
        #         print('emb3', emb3.squeeze(0).shape)
        #         print('ems[3]', embs[x].unsqueeze(0).shape)
        #         emb3 = torch.mean(torch.stack((emb0.squeeze(0), embs[x].unsqueeze(0)), dim=1))
        #     if lbls[x] == 4:
        #         print('ems[4]', embs[x].squeeze(0).unsqueeze(0).shape)
        #         emb4 = torch.stack((emb4.squeeze(0), embs[x].unsqueeze(0)), dim=1)
        #     if lbls[x] == 5:
        #         emb5 = torch.stack((emb5.squeeze(0), embs[x].unsqueeze(0)), dim=1)
        #     if lbls[x] == 6:
        #         emb6 = torch.stack((emb6.squeeze(0), embs[x].unsqueeze(0)), dim=1)

    def forward(self,features,adj,sparse,gcn,idx,seq,labels=None,train=0):
        # promptweight = torch.FloatTensor(1,3).cuda()
        # promptweight[0][0] = 0.3
        # promptweight[0][1] = 0.3
        # promptweight[0][2] = 0.3
        # print(self.a)

        features1 = self.prefeature(features)
        embeds1 = gcn(features1,adj, sparse, None).squeeze()
        # print(embeds1.shape)
        pretrain_embs1 = embeds1[idx]        
        # print(pretrain_embs1.shape)

        
        # rawret1 = self.composedprompt(seq)
        
        # rawret2 = self.downprompt(seq)
        # rawret =rawret2

        rawret = pretrain_embs1




        # rawret = seq
        rawret = rawret.cuda()
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret,nb_class=self.nb_classes)
        ret = torch.FloatTensor(seq.shape[0],self.nb_classes).cuda()
        rawret = torch.cat((rawret,self.ave),dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:seq.shape[0],seq.shape[0]:]
        ret = F.softmax(ret, dim=1)

        # ret = torch.argmax(ret, dim=1)
        # print('ret=', ret)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0)
        # self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        
        # weight = F.softmax(self.weight, dim=1)
        # print("weight",self.weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)




def averageemb(labels,rawret,nb_class):
    retlabel = torch_scatter.scatter(src=rawret,index=labels,dim=0,reduce='mean')
    return retlabel

# def center_embedding(input, index, label_num=7, debug=False):
#     result = torch.zeros(7, 512).cuda()
#     device = input.device
#     index = index.to(device)
#     mean = torch.ones(index.size(0)).to(device)
#     _mean = torch.zeros(label_num, device=device).scatter_add_(dim=0, index=index, src=mean).to(device)
#     index = index.reshape(-1, 1)
#     index = index.expand(input.size())
#     print("label_num", label_num)
#     print("inputsize", input.size(1))
#     result = result.scatter_add_(dim=0, index=index, src=input)
#     _mean = _mean.reshape(-1, 1)
#     result = result / _mean
#     return result
#
#
# def distance2center(f, center):
#     _f = torch.broadcast_to(f, (center.size(0), f.size(0), f.size(1)))
#     _center = torch.broadcast_to(center, (f.size(0), center.size(0), center.size(1)))
#     _f = _f.permute(1, 0, 2)
#     _center = _center.reshape(-1, _center.size(2))
#     _f = _f.reshape(-1, _f.size(2))
#     cos = torch.cosine_similarity(_f, _center, dim=1)
#     res = cos
#     res = res.reshape(f.size(0), center.size(0))
#     return res

class weighted_prompt(nn.Module):
    def __init__(self,weightednum):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding=torch.mm(self.weight,graph_embedding)
        return graph_embedding
    



class weighted_feature(nn.Module):
    def __init__(self,weightednum):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)
    def forward(self, graph_embedding1,graph_embedding2):
        # print("weight",self.weight)
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    

class downstreamprompt(nn.Module):
    def __init__(self,hid_units):
        super(downstreamprompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)
    def forward(self, graph_embedding):
        # print("weight",self.weight)
        # weight = self.weight.repeat(graph_embedding.shape[0],1)
        graph_embedding=self.weight * graph_embedding
        return graph_embedding
    





class featureprompt(nn.Module):
    def __init__(self,prompt1,prompt2,prompt3):
        super(featureprompt, self).__init__()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.weightprompt = weighted_prompt(3)
    def forward(self,feature):
        # print("prompt",self.weightprompt.weight)
        weight = self.weightprompt(self.prompt)
        feature = weight * feature
        return feature
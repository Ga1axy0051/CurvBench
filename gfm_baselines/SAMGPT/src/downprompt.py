import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter
from layers.prompt import *
class downstreamprompt(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, fea_pretext_weights, str_pretext_weights, 
                combines, type_ = 'mul', ablation = 'all'):
        super(downstreamprompt, self).__init__()
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList([
            composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
            for i in range(num_layers_num)
        ])
        
        self.open_prompt_fea = textprompt(feature_dim)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            new_layer = textprompt(in_features, type_)
            self.open_prompt_str.append(new_layer)
        #nn.ModuleList([textprompt(hidden_dim, type) for _ in range(num_layers_num)])

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

    def forward(self, seq, gcn, adj, sparse):
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
            
        composed_seq_fea = self.composedprompt_fea(seq)
        open_seq_fea = self.open_prompt_fea(seq)
        
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)
        if self.ablation_choice[-2:] == 'fo':
            seq_fea == open_seq_fea
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = composed_seq_fea
        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == 'ft':
            return embed_fea
        
        composed_embed_str = gcn(seq, adj, sparse, None, self.composedprompt_str)
        open_embed_str = gcn(seq, adj, sparse, None, self.open_prompt_str)
        if self.beta < 0:
            embed_str = self.weighted_prompt([composed_embed_str, open_embed_str])
        else:
            embed_str = composed_embed_str + self.beta * open_embed_str
        if self.ablation_choice[:2] == 'so':
            embed_str = open_embed_str
        elif self.ablation_choice[:2] == 'sc':
            embed_str = composed_embed_str
        if self.ablation_choice == 'st':
            return embed_str
        
        ret = embed_fea + self.alpha * embed_str
        return ret


class downprompt(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  fea_pretext_weights, str_pretext_weights,
                  combines, type_='mul', ablation = 'all'):
        super(downprompt, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreamprompt(feature_dim, ft_in, num_layers_num, 
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)


    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = embeds[idx]
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)
        ret = torch.FloatTensor(num,self.nb_classes)
        rawret = torch.cat((rawret,self.ave),dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downprompt_graph(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  fea_pretext_weights, str_pretext_weights,
                  combines, type_='mul', ablation = 'all'):
        super(downprompt_graph, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreamprompt(feature_dim, ft_in, num_layers_num, 
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)


    def forward(self,features,adj,sparse,gcn,idx,batch,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)
        ret = torch.FloatTensor(num,self.nb_classes)
        rawret = torch.cat((rawret,self.ave),dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

def averageemb(labels, rawret):
    retlabel = torch_scatter.scatter(src=rawret,index=labels,dim=0,reduce='mean')
    return retlabel
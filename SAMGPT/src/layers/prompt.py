import torch
import torch.nn as nn

class textprompt(nn.Module):
    def __init__(self, hid_units, type_='mul'):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype = type_
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding=self.weight * graph_embedding

        return graph_embedding
    


class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        # graph_embedding=torch.mm(self.weight, graph_embedding)
        assert len(graph_embedding) == self.weight.shape[1], 'length must equal'
        ans = torch.zeros_like(graph_embedding[0])
        for i in range(len(graph_embedding)):
            ans += self.weight[0][i] * graph_embedding[i]
        return ans

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding1, graph_embedding2):

        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    
class composedtoken(nn.Module):
    def __init__(self, texttokens, type_='mul'):
        super(composedtoken, self).__init__()
        # print(texttoken1.shape)
        self.texttoken = torch.cat(texttokens,dim=0)
        # print(self.texttoken.shape)
        self.prompt = weighted_prompt( len(texttokens) )
        self.type = type_

    def forward(self, seq):
        # print(seq.shape)
        
        texttoken = self.prompt(self.texttoken)
        
        # print(texttoken.shape)
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets
    
class composedNet(nn.Module):
    def __init__(self, length):
        super(composedNet, self).__init__()
        #self.texttoken = torch.cat(texttokens,dim=0)
        self.length = length
        self.prompt = weighted_prompt( length ).cuda()

    def forward(self, paras):
        # print(seq.shape)
        assert self.length == len(paras), 'number of paras must equal to self.length'
        target = {}
        for key, value in paras[0].items():
            target[key] = torch.zeros_like(value)
        for key in paras[0].keys():
            para_key = [para[key] for para in paras]
            target[key] = self.prompt(para_key)

        return target
 
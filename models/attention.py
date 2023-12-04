import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available else "cpu")
class Attention_mask(nn.Module):

    def __init__(self, hidden, att_dim, r, activation='tanh', return_attention=True,
                 attention_regularizer_weight=0.0, normalize=False, attmod='smooth', sharp_beta=1):
        super().__init__()
        self.W_initializer = nn.init.xavier_uniform_
        
        # W_initializer = nn.init.xavier_uniform_
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        self.hidden = hidden
        self.att_dim = att_dim
        self.r = r
        self.return_attention = return_attention
        self.attention_regularizer_weight = attention_regularizer_weight
        self.normalize = normalize
        self.attmod = attmod
        self.sharp_beta = sharp_beta
        # self.W1 = nn.Parameter(self.W_initializer(torch.Tensor(hidden, att_dim))).to(device)
        self.W1 = nn.Linear(hidden, att_dim, bias=False)
        self.W2 = nn.Linear(att_dim, r, bias=False)

    def forward(self, H, masks=None):
        H1 = H[:, :-1, :] #(b, hidden, length)
        attention_mask = H[:, -1, :] #[b, length]
        H1_t = H1.transpose(1, 2) #(b, length, hidden)
        H_t = self.activation(self.W1(H1_t)) # (b, length, att_dim)
        temp = self.W2(H_t) # [b, length, r] #same with unmasked K?
        adder = (1.0 - attention_mask) * -10000.0 #[b, length]
        temp += torch.repeat_interleave(adder.unsqueeze(-1), self.r, dim = 2) #[b, length, r]
        if 'softmax' in self.attmod:
            A = F.softmax(temp * self.sharp_beta, dim=2)  #  [b, length, r]
        elif 'smooth' in self.attmod:
            _at = torch.sigmoid(temp * self.sharp_beta)
            s = torch.sum(_at, dim=1, keepdim=True)
            A = _at / s

        if self.normalize:
            length = torch.sum(attention_mask.float(), dim=1, keepdim=True) / attention_mask.size(1)#mean
            lengthr = torch.repeat_interleave(length, self.r, dim=1)
            A = A * lengthr.unsqueeze(1)
        M = torch.bmm(H1, A) # [none, hidden, r]
        if self.return_attention:
            return [M, A]
        return M
  

    def _attention_regularizer(self, attention):
        batch_size = attention.shape[0]
        identity = torch.eye(self.r).to(attention.device)  # [r,r]
        temp = torch.bmm(attention, attention.transpose(1, 2)) - identity  # [none, r, r]
        penal = self.attention_regularizer_weight * torch.sum(temp**2) / batch_size
        return penal
    

import torch
import torch.nn as nn

class QKVAttention(nn.Module):
    def __init__(self, hidden, att_dim, headnum=5):
        super().__init__()
        self.hidden = hidden
        self.att_dim = att_dim
        self.headnum = headnum
        self.W_q = nn.Linear(hidden, att_dim * headnum)
        self.W_k = nn.Linear(hidden, att_dim * headnum)
        self.W_v = nn.Linear(hidden, hidden * headnum)
        self.W_o = nn.Linear(hidden * headnum, hidden)

    def forward(self, H, masks=None):
        H = H[:, :-1, :] #(none,32,1000)
        batch_size = H.shape[0]
        # print("H", H.shape)
        # print("weight", self.W_q)
        mask = H[:, -1, :] #[1,1000]
        H = torch.transpose(H, 1,2)
        # Compute queries, keys, and values
        q = self.W_q(H) # [batch_size, seq_len, att_dim * headnum]
        k = self.W_k(H) # [batch_size, seq_len, att_dim * headnum]
        v = self.W_v(H) # [batch_size, seq_len, hidden * headnum]

        # Split heads
        q = q.view(batch_size, -1, self.headnum, self.att_dim).transpose(1, 2) # [batch_size, headnum, seq_len, att_dim]
        k = k.view(batch_size, -1, self.headnum, self.att_dim).transpose(1, 2) # [batch_size, headnum, seq_len, att_dim]
        v = v.view(batch_size, -1, self.headnum, self.hidden).transpose(1, 2) # [batch_size, headnum, seq_len, hidden]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2,-1)) / (self.att_dim ** 0.5) # [batch_size, headnum, seq_len, seq_len]
        weights = torch.softmax(scores, dim=-1)
        A = weights.view(batch_size, self.headnum, -1)
        # print("weight", weights.shape)
        # print("scores", scores.shape)
        if masks is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            # print("mask", mask.shape)
            # print("scores", scores.shape)
            scores = scores.masked_fill(mask == 0 , float('-inf'))
        # Compute attended features
        M = torch.matmul(weights, v) # [batch_size,headnum , seq_len ,hidden]
        
        M=M.transpose(1 ,2).contiguous().view(batch_size,-1,self.hidden*self.headnum)
        
        M=self.W_o(M)#[batch_size, seq_len, hidden]
        
        
        return M.permute(0, 2, 1), A



if __name__ == "__main__":

    H = torch.rand(10, 33, 1000)

    attention = QKVAttention(hidden = 32, att_dim = 80, headnum = 5)
    print(attention.W_q.weight.shape)
    print(attention.W_o.weight.shape)
    output = attention(H, masks = True)
    # print(output)
    # print(output.shape)
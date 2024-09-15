import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_k, d_v):
        super(Attention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, query, key, value, mask=None):
        # key: [batch_size, seq_len_k, d_k]
        # query: [batch_size, num_heads, seq_len_q, d_k]

        # 扩展key和value的维度以匹配query的形状
        key = key.unsqueeze(1)  # key变为 [batch_size, 1, seq_len_k, d_k]
        value = value.unsqueeze(1)  # value变为 [batch_size, 1, seq_len_k, d_v]

        # 计算点积注意力得分 (Scaled Dot-Product Attention)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        # attention_scores的形状：[batch_size, num_heads, seq_len_q, seq_len_k]

        # 如果有mask，应用mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 计算softmax后的注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 根据注意力权重对value进行加权求和
        output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len_q, d_v]

        return output, attention_weights


# 测试Attention模块
batch_size = 2
num_heads = 3
seq_len_q = 4
seq_len_k = 5
d_k = 8  # 假设key的维度
d_v = 8  # 假设value的维度

# query的形状为 [batch_size, num_heads, seq_len_q, d_k]
query = torch.rand(batch_size, num_heads, seq_len_q, d_k)
# key的形状为 [batch_size, seq_len_k, d_k]，value的形状相同
key = torch.rand(batch_size, seq_len_k, d_k)
value = torch.rand(batch_size, seq_len_k, d_v)

# 创建一个Attention实例
attention_layer = Attention(d_k, d_v)

# 前向传播
output, attention_weights = attention_layer(query, key, value)

print("Output Shape:", output.shape)  # [batch_size, num_heads, seq_len_q, d_v]
print("Attention Weights Shape:", attention_weights.shape)  # [batch_size, num_heads, seq_len_q, seq_len_k]

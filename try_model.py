import torch
import torch.nn as nn
import torch.nn.functional as F

pic=torch.randn(4, 3, 224, 224)
x=torch.randn(1, 77,768)
class Attention2(nn.Module):
    def __init__(self, hidden_size=768):
        super(Attention2, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        scores = torch.matmul(Q, K.unsqueeze(1).transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32)
        )
        attention_weights = nn.functional.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V.unsqueeze(1))
        result = self.output_layer(weighted_values)

        return result

class Attention(nn.Module):
    def __init__(self, hidden_size=768):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32)
        )
        attention_weights = nn.functional.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V)
        result = self.output_layer(weighted_values)

        return result

class pic_transformer(nn.Module):
    def __init__(self, num_pictures=4, channels=3, height=224, width=224, patch_size=16):
        super(pic_transformer, self).__init__()
        self.num_pictures = num_pictures
        self.channels = channels
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)  # 计算总的patch数量
        self.patch_dim = channels * patch_size * patch_size  # 每个patch的维度
        self.linear_layer1 = nn.Linear(self.patch_dim, self.patch_dim)
        self.attention_layer1 = Attention()

    def forward(self, x):
        batch_size = x.size(0)  # 获取batch大小
        # x: [batch_size, num_pictures, channels, height, width]

        # 将图像划分为patches，并转换为形状 (batch_size, num_pictures, num_patches, patch_dim)
        x = x.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, self.num_pictures, self.channels, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, self.num_pictures, self.num_patches,
                                                          self.patch_dim)

        # 对patches应用linear layer
        x = self.linear_layer1(x)

        # 对每个图片中的patches进行attention操作
        s = []
        for i in range(self.num_pictures):
            attn_out = self.attention_layer1(x[:, 0, :, :], x[:, i, :, :], x[:, i, :, :])
            s.append(attn_out)

        # stack处理后的结果
        x = torch.stack(s, dim=1)  # 维度 [batch_size, num_pictures, num_patches, patch_dim]
        return x
class ranker(nn.Module):
    def __init__(self,input_size=784*768,depth=2,hidden_size=768):
        super(ranker, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x=x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Adapter(nn.Module):
    def __init__(self, depth=2, adapter_weight=0.01, sd_text_size=768):
        super(Adapter, self).__init__()

        self.adapter_weight = adapter_weight

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=sd_text_size, nhead=8, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=depth
        )

        # Attension
        self.attention1 = Attention(hidden_size=sd_text_size)
        self.pic_attention = pic_transformer()
        self.attention2 = Attention2(hidden_size=sd_text_size)
        # LLM layer
        self.fc = nn.Linear(sd_text_size, sd_text_size)
        nn.init.zeros_(self.fc.weight)
        self.final_layer = nn.Linear(in_features=784,out_features= 77)
        self.ranker=ranker()
    def forward(self, x,pic,adapter_weight=0.1):
        out_transformer_encoder = self.transformer_encoder(x)
        out_attention = self.attention1(query=out_transformer_encoder, key=x, value=x)
        out_llm = self.fc(out_attention)
        out_pic=self.pic_attention(pic)
        out_attention2=self.attention2(query=out_pic, key=out_llm,value=out_llm)
        out_adapter=self.final_layer(out_attention2.reshape(x.shape[0],784,768).transpose(-2, -1)).transpose(-2, -1)
        out = self.adapter_weight * out_adapter + (1 - adapter_weight) * x
        t=self.ranker(out_attention2)
        return out,t



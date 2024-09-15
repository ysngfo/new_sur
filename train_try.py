import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from try_model import Adapter
# 定义一个简单的神经网络模型

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer

class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dict, tokenizer, transform=None):
        """
        dataframe: 包含id, input_sentence, target_sentence的DataFrame
        image_dict: 包含图片路径的字典，格式为 {id: [img1_path, img2_path, img3_path, img4_path]}
        tokenizer: 用于对句子进行token化的tokenizer
        transform: 用于图像预处理的transform
        """
        self.dataframe = dataframe
        self.image_dict = image_dict
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 从DataFrame中获取样本id, 输入句子和目标句子
        row = self.dataframe.iloc[idx]
        sample_id = row['id']
        input_sentence = row['input_sentence']
        target_sentence = row['target_sentence']

        # 根据id获取图片路径列表
        image_paths = self.image_dict[sample_id]
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # 将4张图片拼接成一个Tensor
        images = torch.stack(images)

        # 处理输入句子
        input_encoding = self.tokenizer(input_sentence, return_tensors="pt", padding='max_length', max_length=64, truncation=True)

        # 处理目标句子
        target_encoding = self.tokenizer(target_sentence, return_tensors="pt", padding='max_length', max_length=64, truncation=True)

        return {
            'images': images,  # 返回4个视角图片
            'input_ids': input_encoding['input_ids'].squeeze(0),  # 输入句子的token
            'attention_mask': input_encoding['attention_mask'].squeeze(0),  # 输入句子的attention mask
            'target_ids': target_encoding['input_ids'].squeeze(0),  # 目标句子的token
            'target_attention_mask': target_encoding['attention_mask'].squeeze(0)  # 目标句子的attention mask
        }

# 图像预处理，如Resize和标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 使用BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




# 训练函数
def train(model, dataloader, tokenizer, clip_model, clip_processor, optimizer, device, num_epochs=5):
    model.train()
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # 获取图像特征和 token 化的输入/目标句子
            image_features = batch['image_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # 将模型和数据放到GPU上
            model = model.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(image_features, input_ids)  # (batch_size, vocab_size)
            
            # 计算目标句子的概率分布
            target_probs = torch.nn.functional.one_hot(target_ids, num_classes=vocab_size).float()
            
            # 使用KLDivLoss，需要对output做log_softmax
            output_log_probs = torch.nn.functional.log_softmax(output, dim=-1)
            
            # 计算 KL 散度损失
            loss = criterion(output_log_probs, target_probs)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )
# 主训练流程
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, ')


# 模型、优化器和损失函数的定义
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs=10

    model = Adapter().to(device)
    criterion = nn.CrossEntropyLoss()  # 分类任务
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()

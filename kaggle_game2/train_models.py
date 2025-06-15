from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn

# 设备配置
device = torch.device('cpu')


# 1. 数据准备 (假设数据格式：CSV包含'text'和'label'列)
# 示例数据格式：
# text,label
# "这部电影太精彩了",1
# "剧情糟糕透顶",0

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 2. 构建BERT情感分类模型
class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


# 3. 训练函数
def train_model(model, data_loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy


# 4. 评估函数
def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy


# 主函数
def main():
    # 参数配置
    BERT_MODEL_NAME = 'bert-base-chinese'
    MAX_LEN = 512
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2  # 二分类


    df = pd.read_csv('result/data.csv')
    df = df.fillna('')
    df = df[['Label', 'Content']]
    df = df.rename(columns={'Label': 'label', 'Content': 'text'})
    df = df.sample(frac=1).reset_index(drop=True)

    # 分割数据集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 创建数据加载器
    train_dataset = SentimentDataset(
        texts=train_df.text.to_numpy(),
        labels=train_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = SentimentDataset(
        texts=val_df.text.to_numpy(),
        labels=val_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    model = BERTSentimentClassifier(BERT_MODEL_NAME, NUM_CLASSES).to(device)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * EPOCHS

    # 训练循环
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        train_loss, train_acc = train_model(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = eval_model(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model.bin')
            best_accuracy = val_acc

    print(f"训练完成，最佳验证集准确率: {best_accuracy:.4f}")

    # 测试单个样本
    def predict_sentiment(text, model, tokenizer, max_len, device):
        model.load_state_dict(torch.load('best_model.bin'))
        model.eval()

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, prediction = torch.max(outputs, dim=1)

        return "正面" if prediction == 1 else "负面"

    # 示例预测
    test_text = "这部电影让我感动得流泪"
    print(f"文本: '{test_text}'")
    print(f"预测情感: {predict_sentiment(test_text, model, tokenizer, MAX_LEN, device)}")


if __name__ == "__main__":
    main()
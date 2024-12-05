import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wget
import tarfile
import matplotlib.pyplot as plt


# 1. 数据下载和加载
def download_pawsx():
    if not os.path.exists("data"):
        os.makedirs("data")

    url = "https://storage.googleapis.com/paws/pawsx/x-final.tar.gz"
    if not os.path.exists("data/pawsx.tar.gz"):
        filename = wget.download(url, "data/pawsx.tar.gz")

        with tarfile.open("data/pawsx.tar.gz", "r:gz") as tar:
            tar.extractall("data")
        print("\n数据集下载完成！")
    else:
        print("数据集已存在！")


def load_pawsx(language="en", split="train"):
    file_path = f"data/x-final/{language}/{split}.tsv"
    df = pd.read_csv(file_path, sep="\t", quoting=3)
    return df


# 2. 数据集类定义
class PAWSXDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text1 = str(row["sentence1"])
        text2 = str(row["sentence2"])
        label = int(row["label"])

        encoding = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# 3. 模型定义
class TextSimilarityModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        super(TextSimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# 4. 训练函数
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": loss.item()})

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = accuracy_score(actual_labels, predictions)

    return epoch_loss, epoch_accuracy


# 5. 评估函数
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(eval_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions, average="binary"
    )

    return avg_loss, accuracy, precision, recall, f1


# 6. 训练过程可视化
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()


# 7. 主函数
def main():
    # 设置参数
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下载并加载数据
    download_pawsx()
    train_df = load_pawsx("en", "train")
    dev_df = load_pawsx("en", "dev")
    test_df = load_pawsx("en", "test")

    # 初始化tokenizer和数据集
    tokenizer = BertTokenizer.from_pretrained("sentence-transformers/LaBSE")

    train_dataset = PAWSXDataset(train_df, tokenizer, MAX_LENGTH)
    dev_dataset = PAWSXDataset(dev_df, tokenizer, MAX_LENGTH)
    test_dataset = PAWSXDataset(test_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化模型和优化器
    model = TextSimilarityModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 训练循环
    best_val_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # 训练
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(
            model, dev_loader, criterion, DEVICE
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {
              train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(
            f"Val Precision: {val_precision:.4f}, Val Recall: {
                val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")

    # 绘制训练历史
    plot_training_history(train_losses, val_losses,
                          train_accuracies, val_accuracies)

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()

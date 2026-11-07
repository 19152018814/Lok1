#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移学习脚本
将trinder_1数据集训练的CNN模型迁移到glucose_1数据集

功能:
1. 加载预训练的CNN模型权重
2. 冻结卷积层，只训练全连接层
3. 支持不同的迁移学习策略
4. 保存迁移学习后的模型

使用方法:
    python transfer_learning_glucose.py

作者: AI Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import argparse
from datetime import datetime

torch.manual_seed(42)

class CustomImageDataset(Dataset):
    """自定义数据集类，从文件名中提取label"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = []
        self.labels = []
        self.transform = transform

        # 遍历文件夹，提取文件路径和对应的标签
        for label_folder in os.listdir(img_dir):
            label_path = os.path.join(img_dir, label_folder)
            if os.path.isdir(label_path):  # 只处理文件夹
                for img_file in os.listdir(label_path):
                    self.image_files.append(os.path.join(label_path, img_file))
                    self.labels.append(int(label_folder))  # 假设文件夹名称即为类别

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")  # 确保图像转换为RGB格式
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            image = image.reshape([3, 40, 40])
        return image, label

class CNNModel(nn.Module):
    """CNN模型定义，与原始模型保持一致"""
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransferLearningTrainer:
    """迁移学习训练器"""
    
    def __init__(self, source_model_path, target_task='glucose_1/', num_classes=8, 
                 transfer_strategy='fine_tune', learning_rate=1e-4):
        """
        初始化迁移学习训练器
        
        Args:
            source_model_path: 预训练模型路径
            target_task: 目标任务路径
            num_classes: 类别数量
            transfer_strategy: 迁移策略 ('freeze_conv', 'fine_tune', 'feature_extractor')
            learning_rate: 学习率
        """
        self.source_model_path = source_model_path
        self.target_task = target_task
        self.num_classes = num_classes
        self.transfer_strategy = transfer_strategy
        self.learning_rate = learning_rate
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        # 创建结果目录 - 根据策略动态命名
        self.results_dir = os.path.join(target_task, f'results_{transfer_strategy}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化模型
        self.model = self._load_pretrained_model()
        
        # 数据加载器
        self.train_loader, self.test_loader = self._prepare_data()
        
        # 优化器、损失函数和学习率调度器
        self.optimizer, self.criterion, self.scheduler = self._setup_training()
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
    def _load_pretrained_model(self):
        """加载预训练模型"""
        print(f"📥 加载预训练模型: {self.source_model_path}")
        
        # 创建模型实例
        model = CNNModel(self.num_classes).to(self.device)
        
        # 加载预训练权重
        if os.path.exists(self.source_model_path):
            checkpoint = torch.load(self.source_model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint)
            print("✅ 预训练模型加载成功")
        else:
            print("⚠️  预训练模型文件不存在，将使用随机初始化的权重")
            
        return model
    
    def _apply_transfer_strategy(self):
        """应用迁移学习策略"""
        print(f"🔄 应用迁移学习策略: {self.transfer_strategy}")
        
        if self.transfer_strategy == 'freeze_conv':
            # 冻结卷积层，只训练全连接层
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.conv2.parameters():
                param.requires_grad = False
            for param in self.model.conv3.parameters():
                param.requires_grad = False
            print("🔒 卷积层已冻结，只训练全连接层")
            
        elif self.transfer_strategy == 'fine_tune':
            # 微调所有层，但使用较小的学习率
            for param in self.model.parameters():
                param.requires_grad = True
            print("🔧 所有层都将进行微调")
            
        elif self.transfer_strategy == 'feature_extractor':
            # 完全冻结卷积层，只训练分类器
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc1.parameters():
                param.requires_grad = True
            for param in self.model.fc2.parameters():
                param.requires_grad = True
            print("🎯 卷积层作为特征提取器，只训练分类器")
            
        elif self.transfer_strategy == 'layer_wise_lr':
            # 分层学习率：所有层可训练，但使用不同学习率
            for param in self.model.parameters():
                param.requires_grad = True
            print("📊 分层学习率模式：所有层可训练，使用不同学习率")
            
        elif self.transfer_strategy == 'progressive_unfreeze':
            # 渐进式解冻：初始只训练全连接层
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.conv2.parameters():
                param.requires_grad = False
            for param in self.model.conv3.parameters():
                param.requires_grad = False
            print("🔄 渐进式解冻模式：初始冻结卷积层，后续逐步解冻")
    
    def _prepare_data(self):
        """准备数据加载器"""
        print("📊 准备数据加载器...")
        
        # 图像转换
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 数据路径
        train_dir = os.path.join(self.target_task, "train_dataset_cnn")
        test_dir = os.path.join(self.target_task, "test_dataset_cnn")
        
        # 创建数据集
        train_dataset = CustomImageDataset(img_dir=train_dir, transform=transform)
        test_dataset = CustomImageDataset(img_dir=test_dir, transform=transform)
        
        print(f"📈 训练集大小: {len(train_dataset)}")
        print(f"📉 测试集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        return train_loader, test_loader
    
    def _setup_training(self):
        """设置训练参数"""
        # 应用迁移学习策略
        self._apply_transfer_strategy()
        
        # 设置优化器
        if self.transfer_strategy == 'layer_wise_lr':
            # 分层学习率：不同层使用不同学习率
            conv_params = list(self.model.conv1.parameters()) + list(self.model.conv2.parameters()) + list(self.model.conv3.parameters())
            fc_params = list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())
            
            param_groups = [
                {'params': conv_params, 'lr': self.learning_rate * 0.1},  # 卷积层使用较小学习率
                {'params': fc_params, 'lr': self.learning_rate}  # 全连接层使用正常学习率
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=1e-5)
            print(f"📊 分层学习率优化器: 卷积层lr={self.learning_rate * 0.1}, 全连接层lr={self.learning_rate}")
        else:
            # 其他策略使用统一学习率
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=1e-5)
            print(f"🎯 可训练参数数量: {sum(p.numel() for p in trainable_params)}")
        
        # 学习率调度器 - 与conc_cls_cnn.py保持一致
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        print(f"📊 总参数数量: {sum(p.numel() for p in self.model.parameters())}")
        print(f"📈 学习率调度器: CosineAnnealingLR (T_max=1000, eta_min=1e-5)")
        
        return optimizer, criterion, scheduler
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(self.train_loader)
        train_acc = 100 * correct / total
        
        return avg_loss, train_acc
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        return test_acc
    
    def train(self, num_epochs=500, save_best=True):
        """训练模型"""
        print(f"🚀 开始迁移学习训练，共 {num_epochs} 个epoch")
        print(f"📋 迁移策略: {self.transfer_strategy}")
        print(f"📚 学习率: {self.learning_rate}")
        print("-" * 60)
        
        best_test_accuracy = 0.0
        best_model_path = os.path.join(self.results_dir, 'best_transfer_model.pth')
        
        for epoch in range(num_epochs):
            # 渐进式解冻逻辑
            if self.transfer_strategy == 'progressive_unfreeze':
                self._progressive_unfreeze(epoch, num_epochs)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 评估
            test_acc = self.evaluate()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:4d}/{num_epochs}], "
                      f"Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Test Acc: {test_acc:.2f}%")
            
            # 保存最佳模型
            if save_best and test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                torch.save(self.model.state_dict(), best_model_path)
                if (epoch + 1) % 5 == 0:
                    print(f"💾 新的最佳模型已保存，准确率: {best_test_accuracy:.2f}%")
            
            # 更新学习率调度器
            self.scheduler.step()
        
        print(f"\n✅ 训练完成！最佳测试准确率: {best_test_accuracy:.2f}%")
        return best_test_accuracy
    
    def _progressive_unfreeze(self, epoch, total_epochs):
        """渐进式解冻策略"""
        # 前1/3的epoch只训练全连接层
        if epoch < total_epochs // 3:
            # 保持卷积层冻结
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.conv2.parameters():
                param.requires_grad = False
            for param in self.model.conv3.parameters():
                param.requires_grad = False
        # 中间1/3的epoch解冻conv3
        elif epoch < 2 * total_epochs // 3:
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.conv2.parameters():
                param.requires_grad = False
            for param in self.model.conv3.parameters():
                param.requires_grad = True
            if epoch == total_epochs // 3:
                print("🔄 解冻conv3层")
        # 最后1/3的epoch解冻所有层
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            if epoch == 2 * total_epochs // 3:
                print("🔄 解冻所有卷积层")
        
        # 重新创建优化器以包含新解冻的参数
        if epoch in [total_epochs // 3, 2 * total_epochs // 3]:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
            print("🔄 重新创建优化器以包含新解冻的参数")
    
    def save_results(self):
        """保存训练结果"""
        print("💾 保存训练结果...")
        
        # 保存训练历史
        history_df = pd.DataFrame({
            'Epoch': range(1, len(self.train_losses) + 1),
            'Train_Loss': self.train_losses,
            'Train_Accuracy': self.train_accuracies,
            'Test_Accuracy': self.test_accuracies
        })
        history_df.to_csv(os.path.join(self.results_dir, 'training_history.csv'), index=False)
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        # 生成混淆矩阵
        self._generate_confusion_matrix()
        
        # 保存最终模型
        final_model_path = os.path.join(self.results_dir, 'final_transfer_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        print(f"📁 结果已保存到: {self.results_dir}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue', alpha=0.7)
        ax2.plot(self.test_accuracies, label='Test Accuracy', color='red', alpha=0.7)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_confusion_matrix(self):
        """生成混淆矩阵"""
        # 加载最佳模型进行评估
        best_model_path = os.path.join(self.results_dir, 'best_transfer_model.pth')
        if os.path.exists(best_model_path):
            print("📥 加载最佳模型进行混淆矩阵评估...")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
        else:
            print("⚠️  最佳模型不存在，使用当前模型进行评估")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算混淆矩阵和准确率
        conf_mat = confusion_matrix(all_labels, all_preds)
        accuracy = sum(pred == label for pred, label in zip(all_preds, all_labels)) / len(all_labels) * 100
        
        # 可视化
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Transfer Learning\nAccuracy: {accuracy:.2f}%')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存CSV
        pd.DataFrame(conf_mat).to_csv(os.path.join(self.results_dir, 'confusion_matrix.csv'), index=False)
        
        print(f"📊 混淆矩阵已生成，准确率: {accuracy:.2f}%")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='迁移学习训练脚本')
    parser.add_argument('--source-model', type=str, default='trinder_1/results_cnn/best_model.pth',
                       help='预训练模型路径')
    parser.add_argument('--target-task', type=str, default='glucose_1/',
                       help='目标任务路径')
    parser.add_argument('--strategy', type=str, default='progressive_unfreeze',
                       choices=['freeze_conv', 'fine_tune', 'layer_wise_lr', 'progressive_unfreeze'],
                       help='迁移学习策略')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--num-classes', type=int, default=8,
                       help='类别数量')
    
    args = parser.parse_args()
    
    print("🔄 迁移学习训练开始")
    print(f"📥 源模型: {args.source_model}")
    print(f"🎯 目标任务: {args.target_task}")
    print(f"📋 策略: {args.strategy}")
    print(f"📚 学习率: {args.lr}")
    print(f"🔄 训练轮数: {args.epochs}")
    print("=" * 60)
    
    # 创建训练器
    trainer = TransferLearningTrainer(
        source_model_path=args.source_model,
        target_task=args.target_task,
        num_classes=args.num_classes,
        transfer_strategy=args.strategy,
        learning_rate=args.lr
    )
    
    # 训练模型
    best_accuracy = trainer.train(num_epochs=args.epochs)
    
    # 保存结果
    trainer.save_results()
    
    print(f"\n🎉 迁移学习完成！")
    print(f"🏆 最佳准确率: {best_accuracy:.2f}%")
    print(f"📁 结果保存在: {trainer.results_dir}")

if __name__ == "__main__":
    main()

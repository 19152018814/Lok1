#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回归模型迁移学习脚本
用于将trinder_1/results_reg_cnn/best_model.pth迁移到glucose_1数据集

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
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import argparse
import logging
from pathlib import Path
import sys

# 设置随机种子
torch.manual_seed(42)

class CustomImageDataset(Dataset):
    """自定义图像数据集类"""
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
                    self.labels.append(float(label_folder))  # 假设文件夹名称即为标签（数值）

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

class CNNRegModel(nn.Module):
    """CNN回归模型"""
    def __init__(self):
        super(CNNRegModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.3)  # Dropout层
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 1)  # 输出为单一回归值

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # 输出连续数值
        return x

class TransferLearningRegTrainer:
    """回归迁移学习训练器"""
    
    def __init__(self, 
                 source_model_path='trinder_1/results_reg_cnn/best_model.pth',
                 target_dir='glucose_1',
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 epochs=500,
                 batch_size=32,
                 strategy='fine_tune'):
        """
        初始化迁移学习训练器
        
        Args:
            source_model_path: 源模型路径
            target_dir: 目标数据集目录
            learning_rate: 学习率
            weight_decay: 权重衰减
            epochs: 训练轮数
            batch_size: 批次大小
            strategy: 迁移学习策略 ('freeze_conv', 'fine_tune', 'feature_extractor')
        """
        self.source_model_path = source_model_path
        self.target_dir = target_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.strategy = strategy
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.setup_logging()
        
        # 创建结果目录 - 根据策略动态命名
        self.results_dir = Path(target_dir) / f'results_reg_{strategy}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 图像转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 初始化模型和训练组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('transfer_learning_reg.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _prepare_data(self):
        """准备数据"""
        train_dir = os.path.join(self.target_dir, "train_dataset_cnn")
        test_dir = os.path.join(self.target_dir, "test_dataset_cnn")
        
        # 创建数据集
        train_dataset = CustomImageDataset(img_dir=train_dir, transform=self.transform)
        test_dataset = CustomImageDataset(img_dir=test_dir, transform=self.transform)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        self.logger.info(f"训练集大小: {len(train_dataset)}")
        self.logger.info(f"测试集大小: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def _setup_model(self):
        """设置模型"""
        # 创建模型
        self.model = CNNRegModel().to(self.device)
        
        # 加载预训练权重
        if os.path.exists(self.source_model_path):
            self.logger.info(f"加载预训练模型: {self.source_model_path}")
            state_dict = torch.load(self.source_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.logger.warning(f"预训练模型不存在: {self.source_model_path}")
        
        # 根据策略设置可训练参数
        if self.strategy == 'freeze_conv':
            # 冻结卷积层
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.conv2.parameters():
                param.requires_grad = False
            for param in self.model.conv3.parameters():
                param.requires_grad = False
            self.logger.info("冻结卷积层，只训练全连接层")
        elif self.strategy == 'feature_extractor':
            # 完全冻结卷积层，只训练回归头
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc1.parameters():
                param.requires_grad = True
            for param in self.model.fc2.parameters():
                param.requires_grad = True
            self.logger.info("特征提取器模式：冻结所有卷积层，只训练全连接层")
        elif self.strategy == 'layer_wise_lr':
            # 分层学习率：所有层可训练，但使用不同学习率
            for param in self.model.parameters():
                param.requires_grad = True
            self.logger.info("分层学习率模式：所有层可训练，使用不同学习率")
        elif self.strategy == 'progressive_unfreeze':
            # 渐进式解冻：初始只训练全连接层
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.conv2.parameters():
                param.requires_grad = False
            for param in self.model.conv3.parameters():
                param.requires_grad = False
            self.logger.info("渐进式解冻模式：初始冻结卷积层，后续逐步解冻")
        else:  # fine_tune
            self.logger.info("微调所有层")
        
        # 设置优化器
        if self.strategy == 'layer_wise_lr':
            # 分层学习率：不同层使用不同学习率
            conv_params = list(self.model.conv1.parameters()) + list(self.model.conv2.parameters()) + list(self.model.conv3.parameters())
            fc_params = list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())
            
            param_groups = [
                {'params': conv_params, 'lr': self.learning_rate * 0.1},  # 卷积层使用较小学习率
                {'params': fc_params, 'lr': self.learning_rate}  # 全连接层使用正常学习率
            ]
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
            self.logger.info(f"分层学习率优化器: 卷积层lr={self.learning_rate * 0.1}, 全连接层lr={self.learning_rate}")
        else:
            # 其他策略使用统一学习率
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.logger.info(f"优化器: AdamW (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        
        # 设置损失函数
        self.criterion = nn.MSELoss()
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500, eta_min=5e-5)
        self.logger.info(f"学习率调度器: CosineAnnealingLR (T_max=500, eta_min=5e-5)")
    
    def train(self):
        """训练模型"""
        self.logger.info(f"开始训练，策略: {self.strategy}")
        
        # 准备数据
        train_loader, test_loader = self._prepare_data()
        
        # 设置模型
        self._setup_model()
        
        # 存储训练过程中的损失
        train_losses = []
        test_losses = []
        
        best_test_loss = float('inf')
        best_model_path = self.results_dir / 'best_model.pth'
        
        for epoch in range(self.epochs):
            # 渐进式解冻逻辑
            if self.strategy == 'progressive_unfreeze':
                self._progressive_unfreeze(epoch)
            
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 测试阶段
            self.model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), labels.float())
                    total_loss += loss.item()
            
            avg_test_loss = total_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
            
            # 保存最佳模型
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(self.model.state_dict(), best_model_path)
                if (epoch + 1) % 5 == 0:
                    self.logger.info(f"保存最佳模型，测试损失: {best_test_loss:.4f}")
            
            # 更新学习率
            self.scheduler.step()
        
        # 保存训练历史
        self._save_training_history(train_losses, test_losses)
        
        # 最终评估
        self._evaluate_model(test_loader)
        
        self.logger.info("训练完成!")
    
    def _progressive_unfreeze(self, epoch):
        """渐进式解冻策略"""
        total_epochs = self.epochs
        
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
                self.logger.info("解冻conv3层")
        # 最后1/3的epoch解冻所有层
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            if epoch == 2 * total_epochs // 3:
                self.logger.info("解冻所有卷积层")
        
        # 重新创建优化器以包含新解冻的参数
        if epoch in [total_epochs // 3, 2 * total_epochs // 3]:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500, eta_min=5e-5)
            self.logger.info("重新创建优化器以包含新解冻的参数")
    
    def _save_training_history(self, train_losses, test_losses):
        """保存训练历史"""
        # 保存损失数据
        pd.DataFrame({
            'Epoch': range(1, self.epochs + 1), 
            'Train Loss': train_losses
        }).to_csv(self.results_dir / 'train_loss.csv', index=False)
        
        pd.DataFrame({
            'Epoch': range(1, self.epochs + 1), 
            'Test Loss': test_losses
        }).to_csv(self.results_dir / 'test_loss.csv', index=False)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), train_losses, label='Train Loss', color='blue')
        plt.plot(range(1, self.epochs + 1), test_losses, label='Test Loss', color='green')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Loss over Epochs', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(self.results_dir / 'loss_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("训练历史已保存")
    
    def _evaluate_model(self, test_loader):
        """评估模型"""
        self.logger.info("开始模型评估...")
        
        # 加载最佳模型
        best_model_path = self.results_dir / 'best_model.pth'
        if best_model_path.exists():
            self.logger.info("📥 加载最佳模型进行评估...")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
        else:
            self.logger.warning("⚠️  最佳模型不存在，使用当前模型进行评估")
        
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算回归指标
        mse = mean_squared_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        
        self.logger.info(f'测试集均方误差 (MSE): {mse:.4f}')
        self.logger.info(f'测试集R²分数: {r2:.4f}')
        
        # 绘制拟合曲线
        plt.figure(figsize=(8, 6))
        plt.scatter(all_labels, all_preds, color='blue', alpha=0.5, label='Predicted vs Actual')
        plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--', label='Perfect Fit')
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title(f'Fitted Curve (R² = {r2:.4f}, MSE = {mse:.4f})', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(self.results_dir / 'fitted_curve_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'Actual_Values': all_labels,
            'Predicted_Values': all_preds
        })
        results_df.to_csv(self.results_dir / 'real_vs_predicted.csv', index=False)
        
        self.logger.info("模型评估完成，结果已保存")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回归模型迁移学习脚本')

    parser.add_argument('--source-model',
                       default='exp_cu_7/results_reg_10_cnn/best_model.pth',
                       help='源模型路径 (默认: trinder_1/results_reg_cnn/best_model.pth)')

    parser.add_argument('--target-dir',
                       default='transfer_from_different_models',
                       help='目标数据集目录 (默认: glucose_1)')

    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')

    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='权重衰减 (默认: 1e-5)')

    parser.add_argument('--epochs', type=int, default=500,
                       help='训练轮数 (默认: 500)')

    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小 (默认: 32)')

    parser.add_argument('--strategy', choices=['freeze_conv', 'fine_tune', 'layer_wise_lr', 'progressive_unfreeze'],
                       default= 'layer_wise_lr',
                       help='迁移学习策略 (默认: feature_extractor)')

    args = parser.parse_args()

    try:
        # 验证参数
        if not os.path.exists(args.source_model):
            print(f"警告: 源模型不存在: {args.source_model}")

        if not os.path.exists(args.target_dir):
            print(f"错误: 目标目录不存在: {args.target_dir}")
            return 1


        print(f"回归迁移学习参数:")
        print(f"  源模型: {args.source_model}")
        print(f"  目标目录: {args.target_dir}")
        print(f"  学习率: {args.learning_rate}")
        print(f"  权重衰减: {args.weight_decay}")
        print(f"  训练轮数: {args.epochs}")
        print(f"  批次大小: {args.batch_size}")
        print(f"  迁移策略: {args.strategy}")
        print()

        # 创建训练器并开始训练
        trainer = TransferLearningRegTrainer(
            source_model_path=args.source_model,
            target_dir=args.target_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            strategy=args.strategy
        )

        trainer.train()

        print("✅ 回归迁移学习完成!")
        return 0

    except KeyboardInterrupt:
        print("\n用户中断操作")
        return 1
    except Exception as e:
        print(f"程序执行出错: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

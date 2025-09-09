"""
CNN 기반 사이드스캔 소나 기물 탐지기

ResNet 백본과 어텐션 메커니즘을 활용한 고성능 탐지 모델입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """CNN 모델 설정"""
    # 아키텍처
    backbone: str = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50', 'custom'
    input_channels: int = 1     # 그레이스케일
    num_classes: int = 2        # 기물/배경
    
    # 특징 차원
    feature_dim: int = 512
    hidden_dim: int = 256
    
    # 드롭아웃
    dropout_rate: float = 0.3
    
    # 어텐션
    use_attention: bool = True
    attention_heads: int = 8
    
    # 입력 크기
    input_size: Tuple[int, int] = (224, 224)
    
    # 정규화
    use_batch_norm: bool = True
    use_layer_norm: bool = False


class SpatialAttention(nn.Module):
    """공간 어텐션 모듈"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, H, W]
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class ChannelAttention(nn.Module):
    """채널 어텐션 모듈"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 평균 풀링 경로
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        
        # 최대 풀링 경로
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # 결합 및 시그모이드
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResNetBackbone(nn.Module):
    """ResNet 백본 네트워크"""
    
    def __init__(self, backbone_name: str, input_channels: int = 1, pretrained: bool = True):
        super().__init__()
        
        # 백본 모델 로드
        if backbone_name == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet34':
            self.backbone = resnet34(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone_name}")
        
        # 입력 채널 수정 (그레이스케일용)
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # 분류 레이어 제거
        self.feature_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # avgpool, fc 제거
        
    def forward(self, x):
        features = self.backbone(x)
        return features


class MultiHeadClassifier(nn.Module):
    """다중 헤드 분류기"""
    
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()
        
        # 공통 특징 추출
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 분류 헤드
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 신뢰도 헤드 (옵션)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 기하학적 특징 헤드
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim, 4),  # x, y, width, height
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # 분류 출력
        classification = self.classifier(features)
        
        # 신뢰도 출력
        confidence = self.confidence_head(features)
        
        # 기하학적 특징 출력
        geometry = self.geometry_head(features)
        
        return {
            'classification': classification,
            'confidence': confidence.squeeze(-1),
            'geometry': geometry
        }


class SidescanTargetDetector(nn.Module):
    """
    사이드스캔 소나 기물 탐지기
    
    ResNet 백본과 어텐션 메커니즘을 결합한 고성능 모델입니다.
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        
        # 백본 네트워크
        self.backbone = ResNetBackbone(
            self.config.backbone, 
            self.config.input_channels,
            pretrained=True
        )
        
        # 어텐션 모듈
        if self.config.use_attention:
            self.attention = CBAM(self.backbone.feature_dim)
        else:
            self.attention = nn.Identity()
        
        # 분류기
        self.classifier = MultiHeadClassifier(
            self.backbone.feature_dim,
            self.config.num_classes,
            self.config.hidden_dim,
            self.config.dropout_rate
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
        logger.info(f"SidescanTargetDetector 초기화 완료 - 백본: {self.config.backbone}")
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 백본을 통한 특징 추출
        features = self.backbone(x)
        
        # 어텐션 적용
        attended_features = self.attention(features)
        
        # 분류
        outputs = self.classifier(attended_features)
        
        return outputs
    
    def extract_features(self, x):
        """특징 벡터만 추출"""
        features = self.backbone(x)
        attended_features = self.attention(features)
        
        # 글로벌 평균 풀링
        pooled_features = F.adaptive_avg_pool2d(attended_features, 1)
        feature_vector = pooled_features.flatten(1)
        
        return feature_vector


class EnsembleDetector(nn.Module):
    """앙상블 탐지기"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights)
        
        logger.info(f"앙상블 탐지기 초기화 - {len(models)}개 모델")
    
    def forward(self, x):
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # 가중 평균
        ensemble_output = {}
        
        for key in outputs[0].keys():
            weighted_sum = torch.zeros_like(outputs[0][key])
            
            for i, output in enumerate(outputs):
                weighted_sum += self.weights[i] * output[key]
            
            ensemble_output[key] = weighted_sum
        
        return ensemble_output


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SidescanDataset(Dataset):
    """사이드스캔 소나 데이터셋"""
    
    def __init__(self, images: List[np.ndarray], labels: List[int], 
                 masks: Optional[List[np.ndarray]] = None,
                 transform: Optional[transforms.Compose] = None):
        self.images = images
        self.labels = labels
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 이미지를 PIL 형태로 변환하거나 텐서로 변환
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # [H, W] -> [1, H, W]
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image.transpose(2, 0, 1)  # [H, W, 1] -> [1, H, W]
        
        # 정규화
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        image = torch.from_numpy(image)
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # 마스크 추가 (있는 경우)
        if self.masks is not None and self.masks[idx] is not None:
            mask = self.masks[idx]
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0
            else:
                mask = mask.astype(np.float32)
            
            sample['mask'] = torch.from_numpy(mask)
        
        return sample


class ModelTrainer:
    """모델 훈련기"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 손실 함수
        self.classification_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.confidence_loss = nn.BCELoss()
        self.geometry_loss = nn.SmoothL1Loss()
        
        # 최적화기 (나중에 설정)
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"모델 훈련기 초기화 - 디바이스: {device}")
    
    def setup_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """최적화기 설정"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 학습률 스케줄러
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """손실 계산"""
        total_loss = 0.0
        loss_components = {}
        
        # 분류 손실
        if 'classification' in outputs and 'label' in targets:
            cls_loss = self.classification_loss(outputs['classification'], targets['label'])
            total_loss += cls_loss
            loss_components['classification'] = cls_loss.item()
        
        # 신뢰도 손실
        if 'confidence' in outputs and 'label' in targets:
            # 정답 여부를 신뢰도 타겟으로 사용
            confidence_targets = (targets['label'] == 1).float()
            conf_loss = self.confidence_loss(outputs['confidence'], confidence_targets)
            total_loss += 0.5 * conf_loss  # 가중치 적용
            loss_components['confidence'] = conf_loss.item()
        
        # 기하학적 손실 (마스크가 있는 경우)
        if 'geometry' in outputs and 'mask' in targets:
            # 마스크로부터 바운딩 박스 추출하여 타겟 생성
            geom_targets = self._extract_geometry_from_mask(targets['mask'])
            geom_loss = self.geometry_loss(outputs['geometry'], geom_targets)
            total_loss += 0.3 * geom_loss  # 가중치 적용
            loss_components['geometry'] = geom_loss.item()
        
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components
    
    def _extract_geometry_from_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """마스크로부터 기하학적 정보 추출"""
        batch_size = masks.shape[0]
        geometry = torch.zeros(batch_size, 4, device=self.device)
        
        for i, mask in enumerate(masks):
            if mask.sum() > 0:  # 마스크에 객체가 있는 경우
                coords = torch.nonzero(mask, as_tuple=False).float()
                if len(coords) > 0:
                    y_coords, x_coords = coords[:, 0], coords[:, 1]
                    
                    # 바운딩 박스 계산 (정규화)
                    h, w = mask.shape
                    geometry[i, 0] = x_coords.min() / w  # x
                    geometry[i, 1] = y_coords.min() / h  # y
                    geometry[i, 2] = (x_coords.max() - x_coords.min()) / w  # width
                    geometry[i, 3] = (y_coords.max() - y_coords.min()) / h  # height
        
        return geometry
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에폭 훈련"""
        self.model.train()
        
        epoch_losses = {'total': 0.0, 'classification': 0.0, 'confidence': 0.0, 'geometry': 0.0}
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # 데이터를 디바이스로 이동
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            targets = {'label': labels}
            if 'mask' in batch:
                targets['mask'] = batch['mask'].to(self.device)
            
            # 순전파
            outputs = self.model(images)
            
            # 손실 계산
            loss, loss_components = self.compute_loss(outputs, targets)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 가중치 업데이트
            self.optimizer.step()
            
            # 손실 누적
            for key, value in loss_components.items():
                epoch_losses[key] += value
            
            # 진행률 로그
            if batch_idx % 10 == 0:
                logger.debug(f"배치 {batch_idx}/{num_batches}, 손실: {loss.item():.4f}")
        
        # 평균 손실 계산
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        
        val_losses = {'total': 0.0, 'classification': 0.0, 'confidence': 0.0, 'geometry': 0.0}
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                targets = {'label': labels}
                if 'mask' in batch:
                    targets['mask'] = batch['mask'].to(self.device)
                
                # 순전파
                outputs = self.model(images)
                
                # 손실 계산
                loss, loss_components = self.compute_loss(outputs, targets)
                
                for key, value in loss_components.items():
                    val_losses[key] += value
                
                # 정확도 계산
                predictions = torch.argmax(outputs['classification'], dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        # 평균 계산
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        val_losses['accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return val_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int = 100, save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """전체 훈련 프로세스"""
        if self.optimizer is None:
            self.setup_optimizer()
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        logger.info(f"훈련 시작 - {num_epochs} 에폭")
        
        for epoch in range(num_epochs):
            # 훈련
            train_losses = self.train_epoch(train_loader)
            
            # 검증
            val_losses = self.validate(val_loader)
            
            # 히스토리 업데이트
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            history['val_accuracy'].append(val_losses['accuracy'])
            
            # 학습률 스케줄링
            self.scheduler.step(val_losses['total'])
            
            # 조기 종료 확인
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                # 모델 저장
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_losses['total'],
                        'val_accuracy': val_losses['accuracy']
                    }, save_path)
            else:
                patience_counter += 1
            
            # 로그 출력
            logger.info(f"에폭 {epoch+1}/{num_epochs} - "
                       f"훈련 손실: {train_losses['total']:.4f}, "
                       f"검증 손실: {val_losses['total']:.4f}, "
                       f"검증 정확도: {val_losses['accuracy']:.4f}")
            
            # 조기 종료
            if patience_counter >= max_patience:
                logger.info(f"조기 종료 - {max_patience} 에폭 동안 개선 없음")
                break
        
        logger.info("훈련 완료")
        return history
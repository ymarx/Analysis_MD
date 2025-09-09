#!/usr/bin/env python3
"""
해양환경 시나리오별 모의데이터 생성기

다양한 해양 환경에 따른 시나리오별 사이드스캔 소나 데이터를 생성합니다.
- 시나리오 A: 깊은 바다 (심해, 단조로운 해저)
- 시나리오 B: 얕은 연안 (복잡한 지형, 높은 노이즈)
- 시나리오 C: 중간 깊이 (균형잡힌 환경)
- 시나리오 D: 해류가 강한 지역 (동적 잡음)
- 시나리오 E: 모래/암초 지형 (복잡한 텍스처)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import time
from scipy import ndimage
from scipy.signal import convolve2d

logger = logging.getLogger(__name__)


class MarineEnvironment(Enum):
    """해양환경 타입"""
    DEEP_OCEAN = "deep_ocean"      # 깊은 바다
    SHALLOW_COASTAL = "shallow_coastal"  # 얕은 연안
    MEDIUM_DEPTH = "medium_depth"        # 중간 깊이
    HIGH_CURRENT = "high_current"        # 강한 해류
    SANDY_ROCKY = "sandy_rocky"          # 모래/암초


@dataclass
class ScenarioConfig:
    """시나리오별 설정 파라미터"""
    environment: MarineEnvironment
    depth_range: Tuple[float, float]  # 수심 범위 (m)
    noise_level: float                # 노이즈 강도 (0-1)
    texture_complexity: float         # 텍스처 복잡도 (0-1)
    current_strength: float           # 해류 강도 (0-1)
    sediment_type: str               # 퇴적물 타입
    acoustic_properties: Dict[str, float]  # 음향 특성
    target_visibility: float         # 기뢰 가시성 (0-1)
    shadow_strength: float           # 음향 그림자 강도 (0-1)
    
    def __post_init__(self):
        """파라미터 유효성 검사"""
        self.noise_level = max(0.0, min(1.0, self.noise_level))
        self.texture_complexity = max(0.0, min(1.0, self.texture_complexity))
        self.current_strength = max(0.0, min(1.0, self.current_strength))
        self.target_visibility = max(0.0, min(1.0, self.target_visibility))
        self.shadow_strength = max(0.0, min(1.0, self.shadow_strength))


class ScenarioDataGenerator:
    """시나리오별 데이터 생성기"""
    
    def __init__(self):
        """초기화"""
        self.scenarios = self._create_predefined_scenarios()
        logger.info(f"시나리오 데이터 생성기 초기화 - {len(self.scenarios)}개 시나리오")
    
    def _create_predefined_scenarios(self) -> Dict[str, ScenarioConfig]:
        """사전 정의된 시나리오 생성"""
        scenarios = {}
        
        # 시나리오 A: 깊은 바다 (심해, 단조로운 해저)
        scenarios['A_deep_ocean'] = ScenarioConfig(
            environment=MarineEnvironment.DEEP_OCEAN,
            depth_range=(200.0, 1000.0),
            noise_level=0.1,
            texture_complexity=0.2,
            current_strength=0.1,
            sediment_type='fine_mud',
            acoustic_properties={
                'backscatter_strength': -25.0,  # dB
                'absorption_coefficient': 0.02,
                'bottom_roughness': 0.1,
                'penetration_depth': 0.5
            },
            target_visibility=0.9,
            shadow_strength=0.8
        )
        
        # 시나리오 B: 얕은 연안 (복잡한 지형, 높은 노이즈)
        scenarios['B_shallow_coastal'] = ScenarioConfig(
            environment=MarineEnvironment.SHALLOW_COASTAL,
            depth_range=(5.0, 50.0),
            noise_level=0.6,
            texture_complexity=0.8,
            current_strength=0.5,
            sediment_type='coarse_sand',
            acoustic_properties={
                'backscatter_strength': -15.0,
                'absorption_coefficient': 0.05,
                'bottom_roughness': 0.7,
                'penetration_depth': 0.2
            },
            target_visibility=0.4,
            shadow_strength=0.3
        )
        
        # 시나리오 C: 중간 깊이 (균형잡힌 환경)
        scenarios['C_medium_depth'] = ScenarioConfig(
            environment=MarineEnvironment.MEDIUM_DEPTH,
            depth_range=(50.0, 200.0),
            noise_level=0.3,
            texture_complexity=0.5,
            current_strength=0.3,
            sediment_type='mixed_sand_mud',
            acoustic_properties={
                'backscatter_strength': -20.0,
                'absorption_coefficient': 0.03,
                'bottom_roughness': 0.4,
                'penetration_depth': 0.3
            },
            target_visibility=0.7,
            shadow_strength=0.6
        )
        
        # 시나리오 D: 해류가 강한 지역 (동적 잡음)
        scenarios['D_high_current'] = ScenarioConfig(
            environment=MarineEnvironment.HIGH_CURRENT,
            depth_range=(20.0, 100.0),
            noise_level=0.7,
            texture_complexity=0.4,
            current_strength=0.9,
            sediment_type='shifting_sand',
            acoustic_properties={
                'backscatter_strength': -18.0,
                'absorption_coefficient': 0.04,
                'bottom_roughness': 0.5,
                'penetration_depth': 0.1
            },
            target_visibility=0.3,
            shadow_strength=0.2
        )
        
        # 시나리오 E: 모래/암초 지형 (복잡한 텍스처)
        scenarios['E_sandy_rocky'] = ScenarioConfig(
            environment=MarineEnvironment.SANDY_ROCKY,
            depth_range=(10.0, 80.0),
            noise_level=0.4,
            texture_complexity=0.9,
            current_strength=0.2,
            sediment_type='sand_rock_mix',
            acoustic_properties={
                'backscatter_strength': -12.0,
                'absorption_coefficient': 0.06,
                'bottom_roughness': 0.8,
                'penetration_depth': 0.05
            },
            target_visibility=0.5,
            shadow_strength=0.7
        )
        
        return scenarios
    
    def generate_background_texture(self, scenario: ScenarioConfig, 
                                   size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        시나리오별 배경 텍스처 생성
        
        Args:
            scenario: 시나리오 설정
            size: 이미지 크기
            
        Returns:
            np.ndarray: 배경 텍스처 이미지
        """
        h, w = size
        
        # 기본 랜덤 텍스처
        np.random.seed(hash(str(scenario.environment)) % 2**32)
        base_texture = np.random.normal(0.5, 0.1, (h, w))
        
        # 환경별 특성 적용
        if scenario.environment == MarineEnvironment.DEEP_OCEAN:
            # 깊은 바다: 매우 균일하고 부드러운 텍스처
            base_texture = ndimage.gaussian_filter(base_texture, sigma=3.0)
            base_texture = 0.3 + 0.4 * base_texture  # 어두운 배경
            
        elif scenario.environment == MarineEnvironment.SHALLOW_COASTAL:
            # 얕은 연안: 복잡하고 거친 텍스처
            # 다중 스케일 노이즈 추가
            for scale in [1, 2, 4, 8]:
                noise = np.random.normal(0, 0.05, (h//scale, w//scale))
                noise_resized = np.kron(noise, np.ones((scale, scale)))[:h, :w]
                base_texture += noise_resized
            
            base_texture = 0.4 + 0.6 * base_texture  # 밝은 배경
            
        elif scenario.environment == MarineEnvironment.MEDIUM_DEPTH:
            # 중간 깊이: 적당한 텍스처
            base_texture = ndimage.gaussian_filter(base_texture, sigma=1.5)
            base_texture = 0.35 + 0.5 * base_texture
            
        elif scenario.environment == MarineEnvironment.HIGH_CURRENT:
            # 강한 해류: 동적이고 불규칙한 패턴
            # 유동 패턴 시뮬레이션
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            flow_x = 0.1 * np.sin(2 * np.pi * x / 20) * scenario.current_strength
            flow_y = 0.05 * np.cos(2 * np.pi * y / 15) * scenario.current_strength
            
            # 유동에 따른 텍스처 왜곡
            for i in range(h):
                for j in range(w):
                    src_i = int(max(0, min(h-1, i + flow_y[i, j])))
                    src_j = int(max(0, min(w-1, j + flow_x[i, j])))
                    base_texture[i, j] = base_texture[src_i, src_j]
            
            base_texture = 0.4 + 0.5 * base_texture
            
        elif scenario.environment == MarineEnvironment.SANDY_ROCKY:
            # 모래/암초: 고대비 패치워크 텍스처
            # 암반 영역 생성
            rock_mask = np.random.random((h//4, w//4)) > 0.7
            rock_mask = np.kron(rock_mask, np.ones((4, 4)))[:h, :w]
            
            # 모래 영역과 암반 영역 구분
            sand_texture = ndimage.gaussian_filter(base_texture, sigma=1.0)
            rock_texture = 0.8 + 0.2 * np.random.normal(0, 0.1, (h, w))
            
            base_texture = np.where(rock_mask, rock_texture, sand_texture)
            base_texture = 0.2 + 0.8 * base_texture  # 높은 대비
        
        # 노이즈 레벨 적용
        noise = np.random.normal(0, scenario.noise_level * 0.1, (h, w))
        base_texture += noise
        
        # 클리핑 및 정규화
        base_texture = np.clip(base_texture, 0, 1)
        
        return base_texture.astype(np.float32)
    
    def generate_mine_target(self, scenario: ScenarioConfig, 
                           size: Tuple[int, int] = (24, 16),
                           mine_type: str = 'spherical') -> np.ndarray:
        """
        시나리오별 기뢰 객체 생성
        
        Args:
            scenario: 시나리오 설정
            size: 기뢰 크기 (width, height)
            mine_type: 기뢰 타입 ('spherical', 'cylindrical', 'complex')
            
        Returns:
            np.ndarray: 기뢰 마스크
        """
        w, h = size
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        if mine_type == 'spherical':
            # 구형 기뢰
            mask = ((x - center_x) / (w/2))**2 + ((y - center_y) / (h/2))**2 <= 1
            
        elif mine_type == 'cylindrical':
            # 원통형 기뢰
            mask = ((x - center_x) / (w/2))**2 <= 1
            
        elif mine_type == 'complex':
            # 복합형 기뢰 (구형 + 돌출부)
            main_body = ((x - center_x) / (w/2))**2 + ((y - center_y) / (h/2))**2 <= 1
            protrusion = ((x - center_x + w/4) / (w/8))**2 + ((y - center_y) / (h/4))**2 <= 1
            mask = main_body | protrusion
        
        # 가시성에 따른 강도 조정
        intensity = scenario.target_visibility
        
        return mask.astype(np.float32) * intensity
    
    def generate_acoustic_shadow(self, mine_mask: np.ndarray, 
                               scenario: ScenarioConfig,
                               shadow_direction: float = 45.0) -> np.ndarray:
        """
        음향 그림자 생성
        
        Args:
            mine_mask: 기뢰 마스크
            scenario: 시나리오 설정
            shadow_direction: 그림자 방향 (도)
            
        Returns:
            np.ndarray: 음향 그림자 마스크
        """
        h, w = mine_mask.shape
        shadow_mask = np.zeros_like(mine_mask)
        
        # 기뢰 위치 찾기
        mine_positions = np.where(mine_mask > 0)
        if len(mine_positions[0]) == 0:
            return shadow_mask
        
        # 기뢰 중심점
        center_y = int(np.mean(mine_positions[0]))
        center_x = int(np.mean(mine_positions[1]))
        
        # 그림자 방향 벡터
        angle_rad = np.radians(shadow_direction)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # 그림자 길이 (환경에 따라 조정)
        shadow_length = int(20 * scenario.shadow_strength)
        
        # 그림자 영역 생성
        for i in range(1, shadow_length):
            shadow_y = int(center_y + i * dy)
            shadow_x = int(center_x + i * dx)
            
            if 0 <= shadow_y < h and 0 <= shadow_x < w:
                # 거리에 따른 그림자 강도 감소
                intensity = scenario.shadow_strength * (1 - i / shadow_length)
                shadow_mask[shadow_y, shadow_x] = intensity
                
                # 그림자 확산
                for offset_y in [-1, 0, 1]:
                    for offset_x in [-1, 0, 1]:
                        sy, sx = shadow_y + offset_y, shadow_x + offset_x
                        if 0 <= sy < h and 0 <= sx < w:
                            shadow_mask[sy, sx] = max(shadow_mask[sy, sx], 
                                                    intensity * 0.5)
        
        return shadow_mask
    
    def add_environmental_effects(self, image: np.ndarray, 
                                scenario: ScenarioConfig) -> np.ndarray:
        """
        환경적 효과 추가
        
        Args:
            image: 기본 이미지
            scenario: 시나리오 설정
            
        Returns:
            np.ndarray: 환경 효과가 적용된 이미지
        """
        result = image.copy()
        h, w = result.shape
        
        # 1. 해류에 의한 왜곡
        if scenario.current_strength > 0.3:
            # 동적 왜곡 패턴
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            distortion_x = scenario.current_strength * 2 * np.sin(2 * np.pi * y / 30)
            distortion_y = scenario.current_strength * 1 * np.cos(2 * np.pi * x / 40)
            
            for i in range(h):
                for j in range(w):
                    src_i = int(max(0, min(h-1, i + distortion_y[i, j])))
                    src_j = int(max(0, min(w-1, j + distortion_x[i, j])))
                    result[i, j] = result[src_i, src_j]
        
        # 2. 흡수 및 산란 효과
        absorption = scenario.acoustic_properties['absorption_coefficient']
        if absorption > 0.03:
            # 거리에 따른 신호 감쇠
            distance_mask = np.sqrt((np.arange(w) - w/2)**2)
            attenuation = np.exp(-absorption * distance_mask / 10)
            result *= attenuation
        
        # 3. 다중 경로 효과 (얕은 물에서)
        if scenario.environment == MarineEnvironment.SHALLOW_COASTAL:
            # 다중 반사로 인한 고스트 이미지
            ghost = np.roll(result, 5, axis=1) * 0.2
            result += ghost
        
        # 4. 볼륨 산란 효과
        if scenario.noise_level > 0.4:
            # 수중 입자에 의한 볼륨 산란
            volume_noise = np.random.exponential(0.1, (h, w)) * scenario.noise_level
            result += volume_noise * 0.1
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def generate_scenario_sample(self, scenario_name: str, 
                               target_present: bool = True,
                               image_size: Tuple[int, int] = (128, 128),
                               mine_type: str = 'spherical') -> Dict[str, Union[np.ndarray, Dict]]:
        """
        시나리오별 샘플 생성
        
        Args:
            scenario_name: 시나리오 이름
            target_present: 기뢰 존재 여부
            image_size: 이미지 크기
            mine_type: 기뢰 타입
            
        Returns:
            Dict: 생성된 샘플 정보
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"알 수 없는 시나리오: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        h, w = image_size
        
        # 1. 배경 텍스처 생성
        background = self.generate_background_texture(scenario, image_size)
        
        # 2. 기뢰 및 그림자 추가
        if target_present:
            # 기뢰 위치 랜덤 선정
            mine_w, mine_h = 24, 16
            min_x, max_x = mine_w//2, w - mine_w//2
            min_y, max_y = mine_h//2, h - mine_h//2
            
            center_x = np.random.randint(min_x, max_x)
            center_y = np.random.randint(min_y, max_y)
            
            # 기뢰 생성
            mine_mask = self.generate_mine_target(scenario, (mine_w, mine_h), mine_type)
            
            # 기뢰를 배경에 합성
            start_y, end_y = center_y - mine_h//2, center_y + mine_h//2
            start_x, end_x = center_x - mine_w//2, center_x + mine_w//2
            
            background[start_y:end_y, start_x:end_x] += mine_mask * 0.5
            
            # 음향 그림자 생성 및 적용
            shadow_mask = np.zeros_like(background)
            shadow_patch = self.generate_acoustic_shadow(mine_mask, scenario)
            shadow_mask[start_y:end_y, start_x:end_x] = shadow_patch
            
            # 그림자 확장
            shadow_direction = np.random.uniform(30, 60)  # 그림자 방향
            extended_shadow = self.generate_acoustic_shadow(
                shadow_mask[start_y:end_y, start_x:end_x], 
                scenario, shadow_direction
            )
            
            # 그림자를 전체 이미지에 적용
            background = background - shadow_mask * 0.3
        
        # 3. 환경적 효과 적용
        final_image = self.add_environmental_effects(background, scenario)
        
        # 4. 메타데이터 생성
        metadata = {
            'scenario_name': scenario_name,
            'environment': scenario.environment.value,
            'target_present': target_present,
            'mine_type': mine_type if target_present else None,
            'image_size': image_size,
            'scenario_config': {
                'depth_range': scenario.depth_range,
                'noise_level': scenario.noise_level,
                'texture_complexity': scenario.texture_complexity,
                'current_strength': scenario.current_strength,
                'sediment_type': scenario.sediment_type,
                'target_visibility': scenario.target_visibility,
                'shadow_strength': scenario.shadow_strength
            },
            'acoustic_properties': scenario.acoustic_properties
        }
        
        return {
            'image': final_image,
            'metadata': metadata
        }
    
    def generate_scenario_dataset(self, scenario_name: str,
                                n_positive: int = 50,
                                n_negative: int = 50,
                                image_size: Tuple[int, int] = (128, 128)) -> Dict[str, List]:
        """
        시나리오별 데이터셋 생성
        
        Args:
            scenario_name: 시나리오 이름
            n_positive: 양성 샘플 수
            n_negative: 음성 샘플 수
            image_size: 이미지 크기
            
        Returns:
            Dict: 생성된 데이터셋
        """
        logger.info(f"시나리오 {scenario_name} 데이터셋 생성 시작: "
                   f"양성 {n_positive}개, 음성 {n_negative}개")
        
        dataset = {
            'images': [],
            'labels': [],
            'metadata': []
        }
        
        mine_types = ['spherical', 'cylindrical', 'complex']
        
        # 양성 샘플 생성
        for i in range(n_positive):
            mine_type = np.random.choice(mine_types)
            sample = self.generate_scenario_sample(
                scenario_name, target_present=True, 
                image_size=image_size, mine_type=mine_type
            )
            
            dataset['images'].append(sample['image'])
            dataset['labels'].append(1)
            dataset['metadata'].append(sample['metadata'])
            
            if (i + 1) % 10 == 0:
                logger.info(f"양성 샘플 {i+1}/{n_positive} 완료")
        
        # 음성 샘플 생성
        for i in range(n_negative):
            sample = self.generate_scenario_sample(
                scenario_name, target_present=False, 
                image_size=image_size
            )
            
            dataset['images'].append(sample['image'])
            dataset['labels'].append(0)
            dataset['metadata'].append(sample['metadata'])
            
            if (i + 1) % 10 == 0:
                logger.info(f"음성 샘플 {i+1}/{n_negative} 완료")
        
        logger.info(f"시나리오 {scenario_name} 데이터셋 생성 완료: "
                   f"총 {len(dataset['images'])}개 샘플")
        
        return dataset


def main():
    """메인 테스트 함수"""
    import matplotlib.pyplot as plt
    
    logging.basicConfig(level=logging.INFO)
    
    # 생성기 초기화
    generator = ScenarioDataGenerator()
    
    # 각 시나리오별 샘플 생성 및 저장
    output_dir = Path("data/scenario_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scenario_name in generator.scenarios.keys():
        logger.info(f"\n=== 시나리오 {scenario_name} 테스트 ===")
        
        # 양성 및 음성 샘플 각각 1개씩 생성
        positive_sample = generator.generate_scenario_sample(
            scenario_name, target_present=True
        )
        negative_sample = generator.generate_scenario_sample(
            scenario_name, target_present=False
        )
        
        # 시각화 및 저장
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(positive_sample['image'], cmap='gray')
        axes[0].set_title(f'{scenario_name} - 기뢰 있음')
        axes[0].axis('off')
        
        axes[1].imshow(negative_sample['image'], cmap='gray')
        axes[1].set_title(f'{scenario_name} - 기뢰 없음')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{scenario_name}_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 메타데이터 저장
        with open(output_dir / f'{scenario_name}_metadata.json', 'w') as f:
            json.dump({
                'positive': positive_sample['metadata'],
                'negative': negative_sample['metadata']
            }, f, indent=2)
        
        logger.info(f"시나리오 {scenario_name} 샘플 저장 완료")
    
    logger.info(f"\n모든 시나리오 샘플이 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
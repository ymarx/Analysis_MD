#!/usr/bin/env python3
"""
다중 환경 성능 벤치마킹 도구

로컬 CPU, GPU, Runpod 환경에서의 성능을 비교 측정합니다.
"""

import time
import torch
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.device_manager import DeviceManager
from src.models.cnn_detector import SidescanTargetDetector
from config.device_configs import get_optimal_config

class PerformanceBenchmark:
    """성능 벤치마킹 클래스"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.results = {}
        
    def benchmark_device_operations(self, iterations: int = 100) -> Dict[str, float]:
        """기본 GPU 연산 성능 측정"""
        device = self.device_manager.device
        
        # 다양한 크기의 텐서로 테스트
        test_sizes = [512, 1024, 2048]
        results = {}
        
        for size in test_sizes:
            # 메모리 할당 테스트
            start_time = time.time()
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)
            allocation_time = time.time() - start_time
            
            # 행렬 곱셈 테스트
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(iterations):
                z = torch.mm(x, y)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            compute_time = (time.time() - start_time) / iterations
            
            # 컨볼루션 테스트
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
            input_tensor = torch.randn(1, 3, size, size, device=device)
            
            start_time = time.time()
            for _ in range(iterations//10):  # 더 무거운 연산이므로 적게
                output = conv(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            conv_time = (time.time() - start_time) / (iterations//10)
            
            results[f'size_{size}'] = {
                'allocation_time_ms': allocation_time * 1000,
                'matmul_time_ms': compute_time * 1000,
                'conv_time_ms': conv_time * 1000,
                'memory_usage_mb': torch.cuda.memory_allocated() // (1024*1024) if device.type == 'cuda' else 0
            }
            
            # 메모리 정리
            del x, y, z, conv, input_tensor, output
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def benchmark_model_training(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """모델 훈련 성능 측정"""
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 64]
        
        device = self.device_manager.device
        results = {}
        
        # 간단한 CNN 모델 생성
        model = SidescanTargetDetector().to(device)
        model = self.device_manager.optimize_model(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch_size in batch_sizes:
            try:
                # 메모리 확인
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 가짜 데이터 생성
                inputs = torch.randn(batch_size, 3, 224, 224, device=device)
                targets = torch.randint(0, 2, (batch_size,), device=device)
                
                # 포워드/백워드 시간 측정
                iterations = 10
                total_time = 0
                
                for _ in range(iterations):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    total_time += time.time() - start_time
                
                avg_time = total_time / iterations
                throughput = batch_size / avg_time  # samples per second
                
                memory_used = torch.cuda.memory_allocated() // (1024*1024) if device.type == 'cuda' else 0
                
                results[f'batch_{batch_size}'] = {
                    'avg_time_ms': avg_time * 1000,
                    'throughput_samples_per_sec': throughput,
                    'memory_usage_mb': memory_used
                }
                
                print(f"배치 크기 {batch_size}: {avg_time*1000:.2f}ms, {throughput:.2f} samples/sec")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[f'batch_{batch_size}'] = {'error': 'OOM'}
                    print(f"배치 크기 {batch_size}: 메모리 부족")
                    break
                else:
                    raise e
            
            # 메모리 정리
            del inputs, targets, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        del model, optimizer, criterion
        return results
    
    def benchmark_data_loading(self) -> Dict[str, float]:
        """데이터 로딩 성능 측정"""
        from torch.utils.data import DataLoader, TensorDataset
        
        # 가짜 데이터셋 생성
        data = torch.randn(1000, 3, 224, 224)
        targets = torch.randint(0, 2, (1000,))
        dataset = TensorDataset(data, targets)
        
        config = self.device_manager.create_dataloader_config()
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            shuffle=True
        )
        
        # 로딩 시간 측정
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device_manager.device)
            targets = targets.to(self.device_manager.device)
            
            if batch_idx >= 50:  # 50개 배치만 측정
                break
        
        total_time = time.time() - start_time
        avg_batch_time = total_time / 50
        
        return {
            'total_time_sec': total_time,
            'avg_batch_time_ms': avg_batch_time * 1000,
            'samples_per_second': (50 * config['batch_size']) / total_time
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """전체 벤치마크 실행"""
        print("=== 성능 벤치마킹 시작 ===")
        
        # 시스템 정보
        device_info = self.device_manager.get_device_summary()
        
        print(f"디바이스: {device_info['device_name']}")
        print(f"메모리: {device_info['memory_info']['available_mb']}MB")
        print(f"배치 크기: {device_info['performance_config']['batch_size']}")
        
        results = {
            'device_info': device_info,
            'timestamp': time.time(),
            'benchmarks': {}
        }
        
        # 1. 기본 GPU 연산 벤치마크
        print("\n1. GPU 연산 성능 측정...")
        results['benchmarks']['device_operations'] = self.benchmark_device_operations()
        
        # 2. 모델 훈련 벤치마크
        print("\n2. 모델 훈련 성능 측정...")
        results['benchmarks']['model_training'] = self.benchmark_model_training()
        
        # 3. 데이터 로딩 벤치마크
        print("\n3. 데이터 로딩 성능 측정...")
        results['benchmarks']['data_loading'] = self.benchmark_data_loading()
        
        # 4. 메모리 정보
        results['final_memory_info'] = self.device_manager.get_memory_info()
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """결과를 파일로 저장"""
        if filename is None:
            device_name = results['device_info']['device_type']
            timestamp = int(results['timestamp'])
            filename = f"benchmark_{device_name}_{timestamp}.json"
        
        output_dir = Path("benchmarks")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"결과 저장: {output_dir / filename}")
        
    def create_performance_report(self, results: Dict[str, Any]):
        """성능 리포트 생성"""
        device_info = results['device_info']
        benchmarks = results['benchmarks']
        
        report = []
        report.append("=== 성능 벤치마크 리포트 ===")
        report.append(f"디바이스: {device_info['device_name']}")
        report.append(f"타입: {device_info['device_type']}")
        report.append(f"메모리: {device_info['memory_info']['total_mb']}MB")
        report.append("")
        
        # 기본 연산 성능
        report.append("== GPU 연산 성능 ==")
        for size, metrics in benchmarks['device_operations'].items():
            report.append(f"{size}:")
            report.append(f"  - 행렬곱: {metrics['matmul_time_ms']:.2f}ms")
            report.append(f"  - 컨볼루션: {metrics['conv_time_ms']:.2f}ms")
            report.append(f"  - 메모리: {metrics['memory_usage_mb']}MB")
        report.append("")
        
        # 모델 훈련 성능
        report.append("== 모델 훈련 성능 ==")
        for batch, metrics in benchmarks['model_training'].items():
            if 'error' not in metrics:
                report.append(f"{batch}: {metrics['avg_time_ms']:.2f}ms, {metrics['throughput_samples_per_sec']:.2f} samples/sec")
            else:
                report.append(f"{batch}: {metrics['error']}")
        report.append("")
        
        # 데이터 로딩 성능
        report.append("== 데이터 로딩 성능 ==")
        data_loading = benchmarks['data_loading']
        report.append(f"배치 로딩 시간: {data_loading['avg_batch_time_ms']:.2f}ms")
        report.append(f"처리량: {data_loading['samples_per_second']:.2f} samples/sec")
        
        report_text = '\n'.join(report)
        print(report_text)
        
        # 파일로 저장
        output_dir = Path("benchmarks")
        output_dir.mkdir(exist_ok=True)
        device_name = device_info['device_type']
        timestamp = int(results['timestamp'])
        
        with open(output_dir / f"report_{device_name}_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="성능 벤치마킹 도구")
    parser.add_argument('--save', action='store_true', help='결과를 파일로 저장')
    parser.add_argument('--device', default='auto', help='사용할 디바이스 (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    try:
        # 디바이스 관리자 초기화
        if args.device != 'auto':
            device_manager = DeviceManager(device=args.device)
        else:
            device_manager = DeviceManager()
        
        benchmark = PerformanceBenchmark()
        results = benchmark.run_full_benchmark()
        
        # 리포트 생성
        benchmark.create_performance_report(results)
        
        # 결과 저장
        if args.save:
            benchmark.save_results(results)
        
    except Exception as e:
        print(f"벤치마크 실행 중 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
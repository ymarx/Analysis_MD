#!/usr/bin/env python3
"""
다중 환경 호환성 테스트 스크립트

로컬 CPU, GPU, 클라우드 환경에서의 시스템 동작을 검증합니다.
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.utils.device_manager import DeviceManager
    from config.device_configs import get_optimal_config, DEVICE_CONFIGS
    import torch
    import numpy as np
except ImportError as e:
    print(f"Import 오류: {e}")
    print("필요한 패키지를 설치해주세요: pip install -r requirements_core.txt")
    sys.exit(1)

class MultiEnvironmentTester:
    """다중 환경 호환성 테스트 클래스"""
    
    def __init__(self):
        self.results = {}
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self):
        """테스트용 데이터 생성"""
        return {
            'small_tensor': torch.randn(10, 10),
            'medium_tensor': torch.randn(100, 100),
            'image_tensor': torch.randn(1, 3, 224, 224),
            'batch_tensor': torch.randn(8, 3, 224, 224)
        }
    
    def test_device_detection(self) -> Dict[str, Any]:
        """디바이스 감지 테스트"""
        test_name = "device_detection"
        print(f"\n=== {test_name} 테스트 ===")
        
        try:
            # 자동 감지
            dm_auto = DeviceManager(device='auto')
            auto_device = dm_auto.device
            
            # CPU 강제
            dm_cpu = DeviceManager(device='cpu')
            cpu_device = dm_cpu.device
            
            # CUDA 테스트 (가능한 경우)
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                dm_cuda = DeviceManager(device='cuda')
                cuda_device = dm_cuda.device
            else:
                cuda_device = None
            
            # MPS 테스트 (가능한 경우)
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if mps_available:
                dm_mps = DeviceManager(device='mps')
                mps_device = dm_mps.device
            else:
                mps_device = None
                
            result = {
                'status': 'success',
                'auto_device': str(auto_device),
                'cpu_device': str(cpu_device),
                'cuda_available': cuda_available,
                'cuda_device': str(cuda_device) if cuda_device else None,
                'mps_available': mps_available,
                'mps_device': str(mps_device) if mps_device else None,
                'summary': dm_auto.get_device_summary()
            }
            
            print(f"✅ 자동 감지 디바이스: {auto_device}")
            print(f"✅ CUDA 사용 가능: {cuda_available}")
            print(f"✅ MPS 사용 가능: {mps_available}")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"❌ 디바이스 감지 실패: {e}")
        
        self.results[test_name] = result
        return result
    
    def test_tensor_operations(self, device_manager: DeviceManager) -> Dict[str, Any]:
        """텐서 연산 테스트"""
        device_type = device_manager.device.type
        test_name = f"tensor_operations_{device_type}"
        print(f"\n=== {test_name} 테스트 ===")
        
        try:
            device = device_manager.device
            operations_results = {}
            
            # 기본 텐서 연산
            for name, tensor in self.test_data.items():
                start_time = time.time()
                
                # 디바이스로 이동
                tensor_device = tensor.to(device)
                
                # 간단한 연산 수행 (텐서 차원에 따라 다른 연산)
                if tensor.dim() == 2:
                    # 2D 텐서: 행렬 곱셈
                    result = torch.matmul(tensor_device, tensor_device.T)
                else:
                    # 4D 텐서 (이미지): element-wise 연산
                    result = tensor_device * tensor_device + 1.0
                    result = torch.mean(result, dim=[2, 3] if tensor.dim() == 4 else None, keepdim=True)
                
                # 결과를 CPU로 이동 (검증용)
                result_cpu = result.cpu()
                
                # GPU 동기화 (필요한 경우)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                elapsed_time = time.time() - start_time
                
                operations_results[name] = {
                    'success': True,
                    'time_ms': elapsed_time * 1000,
                    'input_shape': list(tensor.shape),
                    'output_shape': list(result.shape),
                    'memory_usage': self._get_memory_usage(device_manager)
                }
                
                print(f"✅ {name}: {elapsed_time*1000:.2f}ms")
            
            result = {
                'status': 'success',
                'device': str(device),
                'operations': operations_results
            }
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'device': str(device_manager.device)
            }
            print(f"❌ 텐서 연산 실패 ({device_type}): {e}")
        
        self.results[test_name] = result
        return result
    
    def test_model_loading(self, device_manager: DeviceManager) -> Dict[str, Any]:
        """모델 로딩 및 추론 테스트"""
        device_type = device_manager.device.type
        test_name = f"model_loading_{device_type}"
        print(f"\n=== {test_name} 테스트 ===")
        
        try:
            device = device_manager.device
            
            # 간단한 CNN 모델 생성
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 2)
            )
            
            # 모델을 디바이스로 이동
            start_time = time.time()
            model = model.to(device)
            model_load_time = time.time() - start_time
            
            # 추론 테스트
            test_input = self.test_data['image_tensor'].to(device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
                
            inference_time = time.time() - start_time
            
            # 결과 검증
            output_cpu = output.cpu()
            
            result = {
                'status': 'success',
                'device': str(device),
                'model_load_time_ms': model_load_time * 1000,
                'inference_time_ms': inference_time * 1000,
                'output_shape': list(output.shape),
                'memory_usage': self._get_memory_usage(device_manager)
            }
            
            print(f"✅ 모델 로딩: {model_load_time*1000:.2f}ms")
            print(f"✅ 추론: {inference_time*1000:.2f}ms")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'device': str(device_manager.device)
            }
            print(f"❌ 모델 로딩/추론 실패 ({device_type}): {e}")
        
        self.results[test_name] = result
        return result
    
    def test_configuration_compatibility(self) -> Dict[str, Any]:
        """설정 호환성 테스트"""
        test_name = "configuration_compatibility"
        print(f"\n=== {test_name} 테스트 ===")
        
        try:
            config_results = {}
            
            # 각 디바이스 타입별 설정 테스트
            for device_type, config in DEVICE_CONFIGS.items():
                try:
                    # DeviceManager 생성 시도
                    if device_type == 'cpu':
                        dm = DeviceManager(device='cpu')
                    elif device_type == 'cuda' and torch.cuda.is_available():
                        dm = DeviceManager(device='cuda')
                    elif device_type == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dm = DeviceManager(device='mps')
                    else:
                        config_results[device_type] = {'status': 'skipped', 'reason': 'device_not_available'}
                        continue
                    
                    # 설정 검증
                    dataloader_config = dm.create_dataloader_config()
                    memory_info = dm.get_memory_info()
                    
                    config_results[device_type] = {
                        'status': 'success',
                        'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
                        'dataloader_config': dataloader_config,
                        'memory_info': memory_info
                    }
                    
                    print(f"✅ {device_type} 설정 호환")
                    
                except Exception as e:
                    config_results[device_type] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"❌ {device_type} 설정 오류: {e}")
            
            result = {
                'status': 'success',
                'device_configs': config_results
            }
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"❌ 설정 호환성 테스트 실패: {e}")
        
        self.results[test_name] = result
        return result
    
    def test_fallback_mechanism(self) -> Dict[str, Any]:
        """폴백 메커니즘 테스트"""
        test_name = "fallback_mechanism"
        print(f"\n=== {test_name} 테스트 ===")
        
        try:
            fallback_results = {}
            
            # 존재하지 않는 디바이스 요청 → CPU 폴백
            try:
                dm_invalid = DeviceManager(device='invalid_device')
                fallback_results['invalid_device'] = {
                    'status': 'success',
                    'fallback_device': str(dm_invalid.device),
                    'expected': 'cpu'
                }
                print(f"✅ 잘못된 디바이스 → {dm_invalid.device} 폴백")
            except Exception as e:
                fallback_results['invalid_device'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # CUDA 없을 때 → CPU 폴백
            if not torch.cuda.is_available():
                try:
                    dm_cuda = DeviceManager(device='cuda')
                    fallback_results['cuda_unavailable'] = {
                        'status': 'success',
                        'fallback_device': str(dm_cuda.device),
                        'expected': 'cpu'
                    }
                    print(f"✅ CUDA 없음 → {dm_cuda.device} 폴백")
                except Exception as e:
                    fallback_results['cuda_unavailable'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            result = {
                'status': 'success',
                'fallback_tests': fallback_results
            }
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"❌ 폴백 메커니즘 테스트 실패: {e}")
        
        self.results[test_name] = result
        return result
    
    def _get_memory_usage(self, device_manager: DeviceManager) -> Dict[str, int]:
        """메모리 사용량 조회"""
        try:
            return device_manager.get_memory_info()
        except:
            return {'error': 'memory_info_unavailable'}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        print("🧪 다중 환경 호환성 테스트 시작")
        print("=" * 50)
        
        # 1. 디바이스 감지 테스트
        self.test_device_detection()
        
        # 2. 설정 호환성 테스트
        self.test_configuration_compatibility()
        
        # 3. 폴백 메커니즘 테스트
        self.test_fallback_mechanism()
        
        # 4. 사용 가능한 각 디바이스에서 연산 테스트
        device_managers = []
        
        # CPU는 항상 테스트
        device_managers.append(DeviceManager(device='cpu'))
        
        # CUDA 테스트
        if torch.cuda.is_available():
            device_managers.append(DeviceManager(device='cuda'))
        
        # MPS 테스트
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_managers.append(DeviceManager(device='mps'))
        
        # 각 디바이스에서 테스트 실행
        for dm in device_managers:
            self.test_tensor_operations(dm)
            self.test_model_loading(dm)
        
        return self.results
    
    def generate_report(self) -> str:
        """테스트 결과 리포트 생성"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("🧪 다중 환경 호환성 테스트 결과")
        report_lines.append("=" * 60)
        
        # 전체 통계
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r.get('status') == 'success')
        
        report_lines.append(f"\n📊 전체 통계:")
        report_lines.append(f"  - 총 테스트: {total_tests}")
        report_lines.append(f"  - 성공: {successful_tests}")
        report_lines.append(f"  - 실패: {total_tests - successful_tests}")
        report_lines.append(f"  - 성공률: {successful_tests/total_tests*100:.1f}%")
        
        # 디바이스별 상세 결과
        report_lines.append(f"\n🖥️  디바이스 정보:")
        if 'device_detection' in self.results:
            device_info = self.results['device_detection']
            if device_info.get('status') == 'success':
                report_lines.append(f"  - 자동 선택: {device_info.get('auto_device', 'N/A')}")
                report_lines.append(f"  - CUDA 지원: {device_info.get('cuda_available', False)}")
                report_lines.append(f"  - MPS 지원: {device_info.get('mps_available', False)}")
        
        # 각 테스트 결과
        report_lines.append(f"\n📋 테스트 상세:")
        for test_name, result in self.results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                report_lines.append(f"  ✅ {test_name}")
            else:
                report_lines.append(f"  ❌ {test_name}: {result.get('error', 'Unknown error')}")
        
        # 성능 정보 (있는 경우)
        performance_tests = [k for k in self.results.keys() if 'tensor_operations' in k or 'model_loading' in k]
        if performance_tests:
            report_lines.append(f"\n⚡ 성능 정보:")
            for test_name in performance_tests:
                result = self.results[test_name]
                if result.get('status') == 'success':
                    device = result.get('device', 'unknown')
                    report_lines.append(f"  - {device}:")
                    
                    if 'operations' in result:
                        for op_name, op_result in result['operations'].items():
                            time_ms = op_result.get('time_ms', 0)
                            report_lines.append(f"    • {op_name}: {time_ms:.2f}ms")
                    
                    if 'inference_time_ms' in result:
                        inf_time = result['inference_time_ms']
                        report_lines.append(f"    • 모델 추론: {inf_time:.2f}ms")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
    
    def save_results(self, filename: str = None):
        """결과를 파일로 저장"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"test_results_{timestamp}.json"
        
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 결과 저장: {filepath}")
        
        # 리포트도 저장
        report_filename = filepath.with_suffix('.txt')
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        
        print(f"💾 리포트 저장: {report_filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="다중 환경 호환성 테스트")
    parser.add_argument('--save', action='store_true', help='결과를 파일로 저장')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    
    args = parser.parse_args()
    
    try:
        tester = MultiEnvironmentTester()
        results = tester.run_all_tests()
        
        # 리포트 출력
        report = tester.generate_report()
        print(report)
        
        # 상세 출력
        if args.verbose:
            print("\n🔍 상세 결과:")
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        
        # 결과 저장
        if args.save:
            tester.save_results()
        
        # 전체 성공 여부 확인
        success_count = sum(1 for r in results.values() if r.get('status') == 'success')
        total_count = len(results)
        
        if success_count == total_count:
            print("\n🎉 모든 테스트 통과!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {total_count - success_count}개 테스트 실패")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Runpod 배포 및 관리 스크립트

기능:
- Runpod 인스턴스 생성/관리
- 프로젝트 자동 업로드
- 의존성 설치 자동화
- 작업 실행 및 결과 다운로드
"""

import os
import sys
import json
import time
import requests
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

class RunpodManager:
    """Runpod 클라우드 GPU 관리 클래스"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        if not self.api_key:
            raise ValueError("Runpod API 키가 필요합니다. RUNPOD_API_KEY 환경변수를 설정하세요.")
        
        self.api_url = "https://api.runpod.io/graphql"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.project_root = Path(__file__).parent.parent
        
    def query_graphql(self, query: str, variables: Dict = None) -> Dict[str, Any]:
        """GraphQL 쿼리 실행"""
        payload = {
            'query': query,
            'variables': variables or {}
        }
        
        response = requests.post(self.api_url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        if 'errors' in result:
            raise Exception(f"GraphQL 오류: {result['errors']}")
        
        return result['data']
    
    def get_gpu_types(self) -> List[Dict[str, Any]]:
        """사용 가능한 GPU 타입 조회"""
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                securePrice
                communityPrice
                lowestPrice(input: {gpuCount: 1}) {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        result = self.query_graphql(query)
        return result['gpuTypes']
    
    def create_pod(self, 
                   gpu_type_id: str,
                   image_name: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                   container_disk_gb: int = 50,
                   volume_gb: int = 100,
                   bid_per_gpu: float = None,
                   name: str = "mine-detection-pod") -> str:
        """새로운 Pod 생성"""
        
        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                imageName
                env
                machineId
                machine {
                    podHostId
                }
            }
        }
        """
        
        variables = {
            'input': {
                'cloudType': 'SECURE' if bid_per_gpu is None else 'COMMUNITY',
                'gpuTypeId': gpu_type_id,
                'name': name,
                'imageName': image_name,
                'containerDiskInGb': container_disk_gb,
                'volumeInGb': volume_gb,
                'volumeMountPath': '/workspace',
                'ports': '8888/http,6006/http,8080/http',
                'env': [
                    {'key': 'JUPYTER_PASSWORD', 'value': ''},
                    {'key': 'CUDA_VISIBLE_DEVICES', 'value': '0'},
                    {'key': 'PYTHONUNBUFFERED', 'value': '1'}
                ]
            }
        }
        
        if bid_per_gpu:
            variables['input']['bidPerGpu'] = bid_per_gpu
        
        result = self.query_graphql(mutation, variables)
        pod_info = result['podFindAndDeployOnDemand']
        
        print(f"Pod 생성 완료: {pod_info['id']}")
        return pod_info['id']
    
    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """Pod 상태 확인"""
        query = """
        query getPod($input: PodFilter!) {
            pod(input: $input) {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
                machine {
                    podHostId
                }
            }
        }
        """
        
        variables = {'input': {'podId': pod_id}}
        result = self.query_graphql(query, variables)
        return result['pod']
    
    def wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> bool:
        """Pod가 준비될 때까지 대기"""
        print(f"Pod {pod_id} 준비 대기 중...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                pod = self.get_pod_status(pod_id)
                if pod and pod.get('runtime'):
                    print("Pod 준비 완료!")
                    return True
            except Exception as e:
                print(f"상태 확인 중 오류: {e}")
            
            time.sleep(10)
            print(".", end="", flush=True)
        
        print(f"\nTimeout: Pod가 {timeout}초 내에 준비되지 않았습니다.")
        return False
    
    def upload_project(self, pod_id: str) -> bool:
        """프로젝트 파일을 Pod에 업로드"""
        try:
            pod = self.get_pod_status(pod_id)
            if not pod or not pod.get('runtime'):
                print("Pod가 실행 중이 아닙니다.")
                return False
            
            # SSH 포트 찾기
            ssh_port = None
            ssh_ip = None
            for port in pod['runtime']['ports']:
                if port['privatePort'] == 22:
                    ssh_port = port['publicPort']
                    ssh_ip = port['ip']
                    break
            
            if not ssh_port:
                print("SSH 포트를 찾을 수 없습니다.")
                return False
            
            print(f"프로젝트 업로드 중... ({ssh_ip}:{ssh_port})")
            
            # 프로젝트 압축
            archive_path = "/tmp/mine_detection_project.tar.gz"
            cmd = [
                "tar", "-czf", archive_path,
                "--exclude=.git",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--exclude=data/*.xtf",
                "--exclude=output",
                "--exclude=checkpoints",
                "-C", str(self.project_root.parent),
                str(self.project_root.name)
            ]
            subprocess.run(cmd, check=True)
            
            # SCP로 업로드
            scp_cmd = [
                "scp", "-P", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                archive_path,
                f"root@{ssh_ip}:/workspace/"
            ]
            subprocess.run(scp_cmd, check=True)
            
            # 압축 해제
            ssh_cmd = [
                "ssh", "-p", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"root@{ssh_ip}",
                f"cd /workspace && tar -xzf mine_detection_project.tar.gz && rm mine_detection_project.tar.gz"
            ]
            subprocess.run(ssh_cmd, check=True)
            
            print("프로젝트 업로드 완료!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"업로드 실패: {e}")
            return False
        except Exception as e:
            print(f"업로드 중 오류: {e}")
            return False
    
    def setup_environment(self, pod_id: str) -> bool:
        """Pod에서 환경 설정"""
        try:
            pod = self.get_pod_status(pod_id)
            ssh_port = None
            ssh_ip = None
            
            for port in pod['runtime']['ports']:
                if port['privatePort'] == 22:
                    ssh_port = port['publicPort']
                    ssh_ip = port['ip']
                    break
            
            if not ssh_port:
                return False
            
            print("환경 설정 중...")
            
            # 설치 스크립트 실행
            setup_commands = [
                "cd /workspace/Analysis_MD",
                "pip install -r requirements_core.txt",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "mkdir -p data output checkpoints logs",
                "python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")'",
                "echo 'Environment setup complete!'"
            ]
            
            ssh_cmd = [
                "ssh", "-p", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"root@{ssh_ip}",
                " && ".join(setup_commands)
            ]
            
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("환경 설정 완료!")
                print(result.stdout)
                return True
            else:
                print(f"환경 설정 실패: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"환경 설정 중 오류: {e}")
            return False
    
    def terminate_pod(self, pod_id: str) -> bool:
        """Pod 종료"""
        mutation = """
        mutation terminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input) {
                id
            }
        }
        """
        
        variables = {'input': {'podId': pod_id}}
        
        try:
            self.query_graphql(mutation, variables)
            print(f"Pod {pod_id} 종료 요청 완료")
            return True
        except Exception as e:
            print(f"Pod 종료 실패: {e}")
            return False
    
    def list_pods(self) -> List[Dict[str, Any]]:
        """실행 중인 Pod 목록"""
        query = """
        query {
            myself {
                pods {
                    id
                    name
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            publicPort
                            privatePort
                            type
                        }
                    }
                    gpuCount
                    costPerHr
                    machineId
                }
            }
        }
        """
        
        result = self.query_graphql(query)
        return result['myself']['pods']


def main():
    parser = argparse.ArgumentParser(description="Runpod 배포 및 관리")
    parser.add_argument('--action', choices=['create', 'list', 'terminate', 'setup', 'deploy'], 
                       default='deploy', help='실행할 작업')
    parser.add_argument('--gpu-type', default='NVIDIA RTX 4090', help='GPU 타입')
    parser.add_argument('--pod-id', help='Pod ID (terminate/setup 시 필요)')
    parser.add_argument('--bid-price', type=float, help='Spot 인스턴스 입찰가')
    parser.add_argument('--name', default='mine-detection', help='Pod 이름')
    
    args = parser.parse_args()
    
    try:
        manager = RunpodManager()
        
        if args.action == 'list':
            pods = manager.list_pods()
            print(f"실행 중인 Pod 수: {len(pods)}")
            for pod in pods:
                print(f"- {pod['name']} ({pod['id']}): {pod['costPerHr']}/hr")
                
        elif args.action == 'create':
            # GPU 타입 찾기
            gpu_types = manager.get_gpu_types()
            gpu_type_id = None
            
            for gpu in gpu_types:
                if args.gpu_type.lower() in gpu['displayName'].lower():
                    gpu_type_id = gpu['id']
                    print(f"GPU 타입 선택: {gpu['displayName']} ({gpu['memoryInGb']}GB)")
                    break
            
            if not gpu_type_id:
                print(f"GPU 타입 '{args.gpu_type}'을 찾을 수 없습니다.")
                sys.exit(1)
            
            pod_id = manager.create_pod(gpu_type_id, bid_per_gpu=args.bid_price, name=args.name)
            print(f"생성된 Pod ID: {pod_id}")
            
        elif args.action == 'terminate':
            if not args.pod_id:
                print("--pod-id가 필요합니다.")
                sys.exit(1)
            manager.terminate_pod(args.pod_id)
            
        elif args.action == 'setup':
            if not args.pod_id:
                print("--pod-id가 필요합니다.")
                sys.exit(1)
            
            if manager.wait_for_pod_ready(args.pod_id):
                manager.setup_environment(args.pod_id)
            
        elif args.action == 'deploy':
            # 전체 배포 프로세스
            print("=== Runpod 자동 배포 시작 ===")
            
            # 1. GPU 타입 선택
            gpu_types = manager.get_gpu_types()
            gpu_type_id = None
            
            for gpu in gpu_types:
                if args.gpu_type.lower() in gpu['displayName'].lower():
                    gpu_type_id = gpu['id']
                    print(f"GPU 선택: {gpu['displayName']} (${gpu['lowestPrice']['uninterruptablePrice']}/hr)")
                    break
            
            if not gpu_type_id:
                print(f"GPU 타입 '{args.gpu_type}'을 찾을 수 없습니다.")
                sys.exit(1)
            
            # 2. Pod 생성
            pod_id = manager.create_pod(gpu_type_id, bid_per_gpu=args.bid_price, name=args.name)
            
            # 3. Pod 준비 대기
            if not manager.wait_for_pod_ready(pod_id):
                print("Pod 생성 실패")
                sys.exit(1)
            
            # 4. 프로젝트 업로드
            if not manager.upload_project(pod_id):
                print("프로젝트 업로드 실패")
                sys.exit(1)
            
            # 5. 환경 설정
            if not manager.setup_environment(pod_id):
                print("환경 설정 실패")
                sys.exit(1)
            
            # 6. 접속 정보 출력
            pod = manager.get_pod_status(pod_id)
            for port in pod['runtime']['ports']:
                if port['privatePort'] == 8888:
                    print(f"\n=== 배포 완료 ===")
                    print(f"Jupyter Lab: http://{port['ip']}:{port['publicPort']}")
                    print(f"SSH: ssh -p {port['publicPort']} root@{port['ip']}")
                    print(f"Pod ID: {pod_id}")
                    break
    
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
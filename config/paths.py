"""
파일 경로 관리 유틸리티
"""
import os
from pathlib import Path
from typing import Dict, Optional

class PathManager:
    """프로젝트 파일 경로를 관리하는 클래스"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            'data/processed',
            'data/augmented', 
            'data/annotations',
            'logs',
            'outputs/models',
            'outputs/figures',
            'outputs/results'
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def sample_data(self) -> Path:
        """샘플 데이터 디렉토리"""
        return self.project_root / '[샘플]데이터'
    
    @property
    def datasets(self) -> Path:
        """데이터셋 디렉토리"""
        return self.project_root / 'datasets'
    
    @property
    def processed_data(self) -> Path:
        """전처리된 데이터 디렉토리"""
        return self.project_root / 'data' / 'processed'
    
    @property
    def augmented_data(self) -> Path:
        """증강된 데이터 디렉토리"""
        return self.project_root / 'data' / 'augmented'
    
    @property
    def annotations(self) -> Path:
        """어노테이션 디렉토리"""
        return self.project_root / 'data' / 'annotations'
    
    @property
    def models(self) -> Path:
        """모델 저장 디렉토리"""
        return self.project_root / 'outputs' / 'models'
    
    @property
    def figures(self) -> Path:
        """그림 저장 디렉토리"""
        return self.project_root / 'outputs' / 'figures'
    
    @property
    def results(self) -> Path:
        """결과 저장 디렉토리"""
        return self.project_root / 'outputs' / 'results'
    
    @property
    def logs(self) -> Path:
        """로그 디렉토리"""
        return self.project_root / 'logs'
    
    def get_sample_file(self, filename: str) -> Path:
        """샘플 데이터 파일 경로 반환"""
        return self.sample_data / filename
    
    def get_dataset_path(self, dataset_name: str, data_type: str = 'original') -> Path:
        """데이터셋 경로 반환
        
        Args:
            dataset_name: 데이터셋 이름
            data_type: 'original' 또는 'simulation'
        """
        return self.datasets / dataset_name / data_type
    
    def list_datasets(self) -> list:
        """사용 가능한 데이터셋 목록 반환"""
        if not self.datasets.exists():
            return []
        
        return [d.name for d in self.datasets.iterdir() if d.is_dir()]
    
    def list_sample_files(self) -> Dict[str, Path]:
        """샘플 파일 목록 반환"""
        if not self.sample_data.exists():
            return {}
        
        files = {}
        for file_path in self.sample_data.iterdir():
            if file_path.is_file():
                files[file_path.name] = file_path
        
        return files

# 전역 경로 관리자 인스턴스
path_manager = PathManager()

def ensure_directory(path):
    """디렉토리가 존재하지 않으면 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
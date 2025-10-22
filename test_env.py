#!/usr/bin/env python3
"""
环境测试脚本 - 验证 laps conda 环境是否满足所有项目的运行需求

测试覆盖:
1. action_classification - 动作分类和聚类
2. amplify_motion_tokenizer - 运动tokenizer训练和推理
3. amplify - 主amplify框架
4. video_action_segmenter - 视频动作分割

Usage:
    conda activate laps
    python test_env.py
"""

import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any


class TestResult:
    """测试结果记录"""
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.warnings: List[str] = []
    
    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"✓ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"✗ {test_name}")
        print(f"  错误: {error}")
    
    def add_warning(self, message: str):
        self.warnings.append(message)
        print(f"⚠ {message}")
    
    def summary(self):
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)
        print(f"通过: {len(self.passed)}/{len(self.passed) + len(self.failed)}")
        print(f"失败: {len(self.failed)}")
        print(f"警告: {len(self.warnings)}")
        
        if self.failed:
            print("\n失败的测试:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        
        if self.warnings:
            print("\n警告信息:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print("="*70)
        return len(self.failed) == 0


def test_basic_imports(result: TestResult):
    """测试基础Python库导入"""
    print("\n[1] 测试基础库导入")
    print("-" * 70)
    
    basic_libs = [
        ("numpy", "numpy"),
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("cv2", "opencv-python"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
    ]
    
    for module_name, display_name in basic_libs:
        try:
            __import__(module_name)
            result.add_pass(f"导入 {display_name}")
        except ImportError as e:
            result.add_fail(f"导入 {display_name}", str(e))


def test_deep_learning_libs(result: TestResult):
    """测试深度学习相关库"""
    print("\n[2] 测试深度学习库")
    print("-" * 70)
    
    # PyTorch版本和CUDA
    try:
        import torch
        version = torch.__version__
        result.add_pass(f"PyTorch 版本: {version}")
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            result.add_pass(f"CUDA 可用: {cuda_version}, {device_count} GPU(s)")
        else:
            result.add_warning("CUDA 不可用 - 将使用CPU模式")
    except Exception as e:
        result.add_fail("PyTorch CUDA检查", str(e))
    
    # einops
    try:
        import einops
        result.add_pass(f"einops 版本: {einops.__version__}")
    except ImportError as e:
        result.add_fail("导入 einops", str(e))
    
    # accelerate
    try:
        import accelerate
        result.add_pass(f"accelerate 版本: {accelerate.__version__}")
    except ImportError as e:
        result.add_fail("导入 accelerate", str(e))
    
    # tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        result.add_pass("tensorboard (SummaryWriter)")
    except ImportError as e:
        result.add_fail("导入 tensorboard", str(e))


def test_vector_quantization(result: TestResult):
    """测试vector quantization库 (motion tokenizer核心)"""
    print("\n[3] 测试 Vector Quantization")
    print("-" * 70)
    
    try:
        from vector_quantize_pytorch import FSQ
        result.add_pass("导入 vector_quantize_pytorch.FSQ")
        
        # 测试FSQ实例化
        import torch
        fsq = FSQ(levels=[8, 8], dim=256)
        test_input = torch.randn(2, 10, 256)
        quantized, indices = fsq(test_input)
        result.add_pass("FSQ 功能测试 (实例化和前向传播)")
    except ImportError as e:
        result.add_fail("导入 vector_quantize_pytorch", str(e))
    except Exception as e:
        result.add_fail("FSQ 功能测试", str(e))


def test_visualization_libs(result: TestResult):
    """测试可视化库"""
    print("\n[4] 测试可视化库")
    print("-" * 70)
    
    viz_libs = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    for module_name, display_name in viz_libs:
        try:
            __import__(module_name)
            result.add_pass(f"导入 {display_name}")
        except ImportError as e:
            result.add_fail(f"导入 {display_name}", str(e))
    
    # 测试matplotlib后端
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        result.add_pass("matplotlib Agg后端")
    except Exception as e:
        result.add_fail("matplotlib Agg后端", str(e))


def test_clustering_libs(result: TestResult):
    """测试聚类相关库 (action_classification核心)"""
    print("\n[5] 测试聚类库")
    print("-" * 70)
    
    # HDBSCAN
    try:
        import hdbscan
        # hdbscan可能没有__version__属性，尝试获取
        version = getattr(hdbscan, '__version__', 'unknown')
        result.add_pass(f"导入 hdbscan 版本: {version}")
        
        # 简单功能测试
        import numpy as np
        X = np.random.randn(100, 10)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        labels = clusterer.fit_predict(X)
        result.add_pass("HDBSCAN 功能测试")
    except ImportError as e:
        result.add_fail("导入 hdbscan", str(e))
    except Exception as e:
        result.add_fail("HDBSCAN 功能测试", str(e))
    
    # UMAP
    try:
        import umap
        result.add_pass("导入 umap-learn")
        
        # 简单功能测试
        import numpy as np
        X = np.random.randn(50, 20)
        reducer = umap.UMAP(n_components=2, n_neighbors=5)
        embedding = reducer.fit_transform(X)
        result.add_pass("UMAP 功能测试")
    except ImportError as e:
        result.add_fail("导入 umap-learn", str(e))
    except Exception as e:
        result.add_fail("UMAP 功能测试", str(e))


def test_project_imports(result: TestResult):
    """测试项目模块导入"""
    print("\n[6] 测试项目模块导入")
    print("-" * 70)
    
    # 添加项目路径
    workspace = Path(__file__).parent
    sys.path.insert(0, str(workspace))
    
    # amplify_motion_tokenizer
    try:
        from amplify_motion_tokenizer.models.motion_tokenizer import MotionTokenizer
        result.add_pass("导入 amplify_motion_tokenizer.models.motion_tokenizer")
    except ImportError as e:
        result.add_fail("导入 amplify_motion_tokenizer", str(e))
    
    try:
        from amplify_motion_tokenizer.dataset.velocity_dataset import get_dataloaders
        result.add_pass("导入 amplify_motion_tokenizer.dataset")
    except ImportError as e:
        result.add_fail("导入 amplify_motion_tokenizer.dataset", str(e))
    
    # action_classification
    try:
        from action_classification.data.features import read_json_sample
        result.add_pass("导入 action_classification.data.features")
    except ImportError as e:
        result.add_fail("导入 action_classification.data", str(e))
    
    try:
        from action_classification.embedding.common import SeqEncoder, SeqDataset
        result.add_pass("导入 action_classification.embedding")
    except ImportError as e:
        result.add_fail("导入 action_classification.embedding", str(e))
    
    # video_action_segmenter
    try:
        from video_action_segmenter.stream_utils.video import TimeResampler
        result.add_pass("导入 video_action_segmenter.stream_utils")
    except ImportError as e:
        result.add_fail("导入 video_action_segmenter", str(e))
    
    # amplify (可能需要额外依赖)
    try:
        sys.path.insert(0, str(workspace / "amplify"))
        from amplify.models.motion_tokenizer import get_fsq_level
        result.add_pass("导入 amplify.models.motion_tokenizer")
    except ImportError as e:
        result.add_warning(f"amplify.models 导入失败 (可能需要额外依赖): {e}")


def test_model_instantiation(result: TestResult):
    """测试模型实例化"""
    print("\n[7] 测试模型实例化")
    print("-" * 70)
    
    try:
        import torch
        import torch.nn as nn
        from vector_quantize_pytorch import FSQ
        
        # 测试简单的motion tokenizer组件
        class SimpleEncoder(nn.Module):
            def __init__(self, d_model=256):
                super().__init__()
                self.proj = nn.Linear(2, d_model)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
                    num_layers=2
                )
            
            def forward(self, x):
                x = self.proj(x)
                return self.transformer(x)
        
        encoder = SimpleEncoder()
        fsq = FSQ(levels=[8, 8], dim=256)
        
        # 测试前向传播
        x = torch.randn(2, 10, 2)
        encoded = encoder(x)
        quantized, indices = fsq(encoded)
        
        result.add_pass("模型实例化和前向传播测试")
    except Exception as e:
        result.add_fail("模型实例化测试", str(e))


def test_data_processing(result: TestResult):
    """测试数据处理能力"""
    print("\n[8] 测试数据处理")
    print("-" * 70)
    
    try:
        import numpy as np
        import cv2
        
        # 测试图像处理
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        resized = cv2.resize(img, (112, 112))
        result.add_pass("OpenCV 图像处理")
        
        # 测试numpy操作
        arr = np.random.randn(100, 50)
        normalized = (arr - arr.mean()) / (arr.std() + 1e-8)
        result.add_pass("NumPy 数据处理")
        
        # 测试torch tensor操作
        import torch
        tensor = torch.from_numpy(arr).float()
        tensor_gpu = tensor.cuda() if torch.cuda.is_available() else tensor
        result.add_pass("PyTorch Tensor 操作")
        
    except Exception as e:
        result.add_fail("数据处理测试", str(e))


def test_yaml_config(result: TestResult):
    """测试YAML配置文件读取"""
    print("\n[9] 测试配置文件处理")
    print("-" * 70)
    
    try:
        import yaml
        
        # 测试YAML读写
        test_config = {
            'model': {
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.1
            },
            'data': {
                'batch_size': 32,
                'sequence_length': 16
            }
        }
        
        yaml_str = yaml.dump(test_config)
        loaded = yaml.safe_load(yaml_str)
        
        assert loaded['model']['hidden_dim'] == 256
        result.add_pass("YAML 配置文件处理")
        
    except Exception as e:
        result.add_fail("YAML 配置测试", str(e))


def test_optional_dependencies(result: TestResult):
    """测试可选依赖"""
    print("\n[10] 测试可选依赖")
    print("-" * 70)
    
    # CoTracker (可选)
    try:
        import torch
        # 不实际加载模型，只测试torch.hub是否可用
        hub_dir = torch.hub.get_dir()
        result.add_pass("torch.hub 可用 (CoTracker支持)")
    except Exception as e:
        result.add_warning(f"torch.hub 不可用 - CoTracker可能无法使用: {e}")
    
    # timm (用于vision encoders)
    try:
        import timm
        result.add_pass(f"timm 可用 (vision encoders)")
    except ImportError:
        result.add_warning("timm 未安装 - vision encoders可能受限")
    
    # omegaconf (用于配置管理)
    try:
        from omegaconf import OmegaConf
        result.add_pass("omegaconf 可用")
    except ImportError:
        result.add_warning("omegaconf 未安装 - 某些amplify功能可能受限")


def main():
    """主测试函数"""
    print("="*70)
    print("环境测试 - laps conda 环境")
    print("="*70)
    print(f"Python 版本: {sys.version}")
    print(f"工作目录: {Path.cwd()}")
    print("="*70)
    
    result = TestResult()
    
    try:
        test_basic_imports(result)
        test_deep_learning_libs(result)
        test_vector_quantization(result)
        test_visualization_libs(result)
        test_clustering_libs(result)
        test_project_imports(result)
        test_model_instantiation(result)
        test_data_processing(result)
        test_yaml_config(result)
        test_optional_dependencies(result)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n\n未预期的错误: {e}")
        traceback.print_exc()
        return 1
    
    # 输出总结
    success = result.summary()
    
    if success:
        print("\n✓ 所有必需测试通过! 环境配置正确。")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

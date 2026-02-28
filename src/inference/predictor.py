"""
BERT推理预测模块
支持MRPC和SST-2任务的推理
"""

import os
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification


class BertPredictor:
    """BERT推理预测器"""
    
    TASK_CONFIG = {
        'mrpc': {
            'max_length': 100,
            'label_names': ['不等价 (0)', '等价 (1)'],
            'task_type': 'pair'
        },
        'sst2': {
            'max_length': 64,
            'label_names': ['负面 (0)', '正面 (1)'],
            'task_type': 'single'
        }
    }
    
    def __init__(self, task_name: str, model_path: str = None, experiment_id: str = None):
        """
        初始化预测器
        
        Args:
            task_name: 任务名称 ('mrpc' 或 'sst2')
            model_path: 模型路径，如果为None则自动查找最新的实验
            experiment_id: 指定实验ID，如果为None则使用最新的
        """
        if task_name not in self.TASK_CONFIG:
            raise ValueError(f"不支持的任务: {task_name}，请使用 'mrpc' 或 'sst2'")
        
        self.task_name = task_name
        self.task_config = self.TASK_CONFIG[task_name]
        
        if model_path is None:
            # 自动查找最新的实验模型
            model_path = self._find_latest_model(task_name, experiment_id)
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _find_latest_model(self, task_name, experiment_id=None):
        """查找最新的模型"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(project_root, 'results')
        
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"结果目录不存在: {results_dir}")
        
        # 查找符合条件的实验
        experiments = [d for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d)) and task_name in d]
        
        if not experiments:
            raise FileNotFoundError(f"没有找到 {task_name} 任务的训练结果")
        
        if experiment_id:
            if experiment_id not in experiments:
                raise FileNotFoundError(f"未找到实验: {experiment_id}")
            model_path = os.path.join(results_dir, experiment_id, 'model')
        else:
            # 使用最新的实验
            experiments.sort(reverse=True)
            model_path = os.path.join(results_dir, experiments[0], 'model')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        return model_path
        
    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型不存在: {self.model_path}，请先训练模型")
        
        print(f"正在加载模型: {self.model_path}")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path,
            return_dict=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载完成，使用设备: {self.device}")
        
    def predict(self, text1: str, text2: str = None):
        """
        单条预测
        
        Args:
            text1: 第一个句子（SST-2）或句子1（MRPC）
            text2: 第二个句子（仅MRPC需要）
            
        Returns:
            dict: 包含预测结果和概率
        """
        if self.model is None:
            self.load_model()
        
        # 根据任务类型准备输入
        if self.task_config['task_type'] == 'pair':
            if text2 is None:
                raise ValueError("MRPC任务需要提供两个句子")
            inputs = self.tokenizer(
                text1, text2,
                truncation=True,
                padding="max_length",
                max_length=self.task_config['max_length'],
                return_tensors='pt'
            )
        else:
            inputs = self.tokenizer(
                text1,
                truncation=True,
                padding="max_length",
                max_length=self.task_config['max_length'],
                return_tensors='pt'
            )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(-1).item()
        
        return {
            'prediction': pred,
            'label': self.task_config['label_names'][pred],
            'probabilities': probs.cpu().numpy()[0].tolist(),
            'confidence': probs.cpu().numpy()[0][pred]
        }
    
    def predict_batch(self, texts):
        """
        批量预测
        
        Args:
            texts: 文本列表，每个元素是 (text1,) 或 (text1, text2)
            
        Returns:
            list: 预测结果列表
        """
        if self.model is None:
            self.load_model()
        
        results = []
        for item in texts:
            if isinstance(item, str):
                result = self.predict(item)
            elif len(item) == 1:
                result = self.predict(item[0])
            else:
                result = self.predict(item[0], item[1])
            results.append(result)
        
        return results
    
    def interactive_predict(self):
        """交互式预测"""
        if self.model is None:
            self.load_model()
        
        print(f"\n{'='*60}")
        print(f"{self.task_name.upper()} 交互式预测")
        print(f"输入 'quit' 或 'q' 退出")
        print(f"{'='*60}\n")
        
        while True:
            if self.task_config['task_type'] == 'pair':
                print("请输入第一个句子:")
                text1 = input("> ").strip()
                if text1.lower() in ['quit', 'q']:
                    break
                
                print("请输入第二个句子:")
                text2 = input("> ").strip()
                if text2.lower() in ['quit', 'q']:
                    break
                
                result = self.predict(text1, text2)
            else:
                print("请输入句子:")
                text1 = input("> ").strip()
                if text1.lower() in ['quit', 'q']:
                    break
                
                result = self.predict(text1)
            
            print(f"\n预测结果: {result['label']}")
            print(f"置信度: {result['confidence']*100:.2f}%")
            print(f"各类别概率: {[f'{p*100:.2f}%' for p in result['probabilities']]}")
            print()
        
        print("退出预测")


if __name__ == '__main__':
    # 测试
    predictor = BertPredictor('sst2')
    predictor.interactive_predict()

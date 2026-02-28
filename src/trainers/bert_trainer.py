"""
BERT微调训练模块（改进版）
所有实验结果（模型、日志、图表、评估结果）统一保存在实验ID文件夹下
"""

import os
import json
import random
import datetime
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import matplotlib.pyplot as plt
from matplotlib import font_manager


class MetricsLogger:
    """训练指标记录器"""
    
    def __init__(self):
        self.train_losses = []
        self.learning_rates = []
        self.eval_metrics = []
        self.epoch_times = []
        self.start_time = None
        
    def log_train_loss(self, step, loss, lr=None):
        self.train_losses.append({
            'step': step,
            'loss': float(loss),
            'learning_rate': float(lr) if lr else None,
            'timestamp': time.time()
        })
        
    def log_learning_rate(self, step, lr):
        self.learning_rates.append({
            'step': step,
            'learning_rate': float(lr)
        })
        
    def log_eval_metrics(self, epoch, metrics):
        self.eval_metrics.append({
            'epoch': epoch,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in metrics.items()}
        })
    
    def log_epoch_time(self, epoch, duration):
        self.epoch_times.append({
            'epoch': epoch,
            'duration_seconds': duration
        })
    
    def to_dict(self):
        return {
            'train_losses': self.train_losses,
            'learning_rates': self.learning_rates,
            'eval_metrics': self.eval_metrics,
            'epoch_times': self.epoch_times
        }


class TrainingMonitorCallback(TrainerCallback):
    """训练监控回调"""
    
    def __init__(self, logger: MetricsLogger):
        self.logger = logger
        self.epoch_start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            self.logger.log_epoch_time(state.epoch, duration)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                lr = logs.get('learning_rate', None)
                self.logger.log_train_loss(state.global_step, logs['loss'], lr)
            if 'learning_rate' in logs:
                self.logger.log_learning_rate(state.global_step, logs['learning_rate'])
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.log_eval_metrics(state.epoch, metrics)


class BertTrainer:
    """BERT微调训练器（改进版）"""
    
    TASK_CONFIG = {
        'mrpc': {
            'dataset_name': 'glue',
            'dataset_config': 'mrpc',
            'text_columns': ['sentence1', 'sentence2'],
            'max_length': 100,
            'metrics': ['accuracy', 'f1'],
            'label_names': ['不等价', '等价']
        },
        'sst2': {
            'dataset_name': 'glue',
            'dataset_config': 'sst2',
            'text_columns': ['sentence'],
            'max_length': 64,
            'metrics': ['accuracy'],
            'label_names': ['负面', '正面']
        }
    }
    
    def __init__(
        self,
        task_name: str,
        model_name: str = 'prajjwal1/bert-mini',
        seed: int = 42,
        learning_rate: float = 3e-5,
        batch_size: int = 32,
        num_epochs: int = 2,
        output_dir: str = None
    ):
        """初始化训练器"""
        if task_name not in self.TASK_CONFIG:
            raise ValueError(f"不支持的任务: {task_name}，请使用 'mrpc' 或 'sst2'")
        
        self.task_name = task_name
        self.task_config = self.TASK_CONFIG[task_name]
        self.model_name = model_name
        self.seed = seed
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # 设置随机种子
        self._set_seed(seed)
        
        # 创建实验ID和输出目录
        self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{task_name}"
        
        if output_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.output_dir = os.path.join(project_root, 'results', self.experiment_id)
        else:
            self.output_dir = os.path.join(output_dir, self.experiment_id)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 模型保存在实验目录下
        self.model_save_path = os.path.join(self.output_dir, 'model')
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.encoded_dataset = None
        self.trainer = None
        self.metrics_logger = MetricsLogger()
        self.train_start_time = None
        self.train_end_time = None
        
    def _set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def load_data(self):
        """加载数据集"""
        print(f"正在加载 {self.task_name.upper()} 数据集...")
        self.dataset = load_dataset(
            self.task_config['dataset_name'],
            self.task_config['dataset_config']
        )
        print(f"训练集大小: {len(self.dataset['train'])}")
        print(f"验证集大小: {len(self.dataset['validation'])}")
        
        # 保存数据集统计信息
        dataset_info = {
            'train_size': len(self.dataset['train']),
            'validation_size': len(self.dataset['validation']),
            'features': str(self.dataset['train'].features),
            'sample': {k: str(v) for k, v in self.dataset['train'][0].items()}
        }
        
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        return self.dataset
    
    def load_model(self):
        """加载预训练模型和分词器"""
        print(f"正在加载预训练模型: {self.model_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            return_dict=True,
            use_safetensors=True  # 明确使用safetensors格式，避免torch.load的版本要求
        )
        
        # 保存模型配置
        model_config = {
            'pretrained_model': self.model_name,
            'num_labels': self.model.config.num_labels,
            'hidden_size': self.model.config.hidden_size,
            'num_hidden_layers': self.model.config.num_hidden_layers,
            'num_attention_heads': self.model.config.num_attention_heads,
            'vocab_size': self.model.config.vocab_size
        }
        
        with open(os.path.join(self.output_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2)
        
        return self.model
    
    def preprocess_data(self):
        """预处理数据"""
        print("正在预处理数据...")
        
        text_columns = self.task_config['text_columns']
        max_length = self.task_config['max_length']
        
        if len(text_columns) == 2:
            def tokenize(examples):
                return self.tokenizer(
                    examples[text_columns[0]],
                    examples[text_columns[1]],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length
                )
        else:
            def tokenize(examples):
                return self.tokenizer(
                    examples[text_columns[0]],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length
                )
        
        tokenized_dataset = self.dataset.map(tokenize, batched=True)
        self.encoded_dataset = tokenized_dataset.map(
            lambda examples: {'labels': examples['label']},
            batched=True
        )
        
        columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        self.encoded_dataset.set_format(type='torch', columns=columns)
        
        return self.encoded_dataset
    
    def _compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        
        results = {}
        for metric_name in self.task_config['metrics']:
            metric = load_metric(metric_name)
            result = metric.compute(predictions=preds, references=labels)
            results.update(result)
        
        return results
    
    def train(self):
        """执行训练"""
        print(f"\n{'='*60}")
        print(f"开始训练 {self.task_name.upper()} 任务")
        print(f"实验ID: {self.experiment_id}")
        print(f"随机种子: {self.seed}")
        print(f"学习率: {self.learning_rate}")
        print(f"批次大小: {self.batch_size}")
        print(f"训练轮数: {self.num_epochs}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
        
        self.train_start_time = time.time()
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, 'checkpoints'),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            warmup_ratio=0.1 if self.task_name == 'sst2' else 0,
            logging_steps=10,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            save_total_limit=2,
            seed=self.seed
        )
        
        # 创建监控回调
        monitor_callback = TrainingMonitorCallback(self.metrics_logger)
        
        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.encoded_dataset["train"],
            eval_dataset=self.encoded_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[monitor_callback]
        )
        
        # 开始训练
        train_result = self.trainer.train()
        
        self.train_end_time = time.time()
        
        # 最终评估
        eval_result = self.trainer.evaluate()
        
        print(f"\n{'='*60}")
        print("训练完成！最终评估结果:")
        for key, value in eval_result.items():
            if not key.startswith('eval_runtime'):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print(f"总训练时间: {self.train_end_time - self.train_start_time:.2f}秒")
        print(f"{'='*60}\n")
        
        return train_result, eval_result
    
    def save_model(self):
        """保存模型"""
        os.makedirs(self.model_save_path, exist_ok=True)
        print(f"正在保存模型到: {self.model_save_path}")
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
        print("模型保存完成！")
    
    def predict_examples(self, num_examples=5):
        """随机选取样本进行预测展示"""
        print(f"\n{'='*60}")
        print(f"随机预测示例 (随机种子: {self.seed})")
        print(f"{'='*60}\n")
        
        random.seed(self.seed)
        dataset_size = len(self.dataset['train'])
        random_indices = random.sample(range(dataset_size), num_examples)
        
        print(f"随机选取的样本索引: {random_indices}\n")
        
        # 加载模型进行预测
        model = BertForSequenceClassification.from_pretrained(
            self.model_save_path, 
            return_dict=True
        )
        model.eval()
        
        results = []
        text_columns = self.task_config['text_columns']
        label_names = self.task_config['label_names']
        
        for i, idx in enumerate(random_indices):
            example = self.dataset['train'][idx]
            
            # 准备输入
            if len(text_columns) == 2:
                inputs = self.tokenizer(
                    example[text_columns[0]],
                    example[text_columns[1]],
                    truncation=True,
                    padding="max_length",
                    max_length=self.task_config['max_length'],
                    return_tensors='pt'
                )
            else:
                inputs = self.tokenizer(
                    example[text_columns[0]],
                    truncation=True,
                    padding="max_length",
                    max_length=self.task_config['max_length'],
                    return_tensors='pt'
                )
            
            # 预测
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = logits.argmax(-1).item()
            
            true_label = example['label']
            is_correct = pred == true_label
            
            # 打印结果
            print(f"样本 {i+1} (索引: {idx}):")
            if len(text_columns) == 2:
                print(f"  句子1: {example[text_columns[0]]}")
                print(f"  句子2: {example[text_columns[1]]}")
            else:
                print(f"  句子: {example[text_columns[0]]}")
            print(f"  真实标签: {true_label} ({label_names[true_label]})")
            print(f"  预测标签: {pred} ({label_names[pred]})")
            print(f"  置信度: {probs[0][pred].item()*100:.2f}%")
            print(f"  结果: {'✓ 正确' if is_correct else '✗ 错误'}")
            print()
            
            results.append({
                'index': idx,
                'texts': {col: example[col] for col in text_columns},
                'true_label': int(true_label),
                'predicted_label': int(pred),
                'probabilities': probs[0].tolist(),
                'confidence': float(probs[0][pred].item()),
                'is_correct': bool(is_correct)
            })
        
        accuracy = sum(r['is_correct'] for r in results) / len(results)
        print(f"预测准确率: {accuracy*100:.1f}% ({sum(r['is_correct'] for r in results)}/{len(results)})")
        
        return results, random_indices
    
    def plot_training_history(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Loss曲线
        if self.metrics_logger.train_losses:
            steps = [item['step'] for item in self.metrics_logger.train_losses]
            losses = [item['loss'] for item in self.metrics_logger.train_losses]
            axes[0, 0].plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7)
            axes[0, 0].set_xlabel('Training Steps', fontsize=11)
            axes[0, 0].set_ylabel('Loss', fontsize=11)
            axes[0, 0].set_title(f'{self.task_name.upper()} Training Loss', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Learning Rate曲线
        if self.metrics_logger.learning_rates:
            steps = [item['step'] for item in self.metrics_logger.learning_rates]
            lrs = [item['learning_rate'] for item in self.metrics_logger.learning_rates]
            axes[0, 1].plot(steps, lrs, 'g-', linewidth=1.5)
            axes[0, 1].set_xlabel('Training Steps', fontsize=11)
            axes[0, 1].set_ylabel('Learning Rate', fontsize=11)
            axes[0, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Evaluation Metrics
        if self.metrics_logger.eval_metrics:
            epochs = [item['epoch'] for item in self.metrics_logger.eval_metrics]
            for metric_name in self.task_config['metrics']:
                key = f'eval_{metric_name}'
                if key in self.metrics_logger.eval_metrics[0]['metrics']:
                    values = [item['metrics'][key] for item in self.metrics_logger.eval_metrics]
                    axes[1, 0].plot(epochs, values, 'o-', linewidth=2, markersize=8, label=metric_name.upper())
            
            axes[1, 0].set_xlabel('Epoch', fontsize=11)
            axes[1, 0].set_ylabel('Score', fontsize=11)
            axes[1, 0].set_title('Evaluation Metrics', fontsize=12, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Epoch Time
        if self.metrics_logger.epoch_times:
            epochs = [item['epoch'] for item in self.metrics_logger.epoch_times]
            times = [item['duration_seconds'] for item in self.metrics_logger.epoch_times]
            axes[1, 1].bar(epochs, times, color='orange', alpha=0.7)
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('Time (seconds)', fontsize=11)
            axes[1, 1].set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {plot_path}")
        plt.close()
        
        return plot_path
    
    def save_results(self, eval_result, predict_results, random_indices):
        """保存所有实验结果"""
        results_data = {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'task_name': self.task_name,
                'seed': self.seed,
                'timestamp': datetime.datetime.now().isoformat(),
                'total_training_time_seconds': self.train_end_time - self.train_start_time if self.train_end_time else None
            },
            'config': {
                'model_name': self.model_name,
                'seed': self.seed,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'max_length': self.task_config['max_length']
            },
            'dataset_info': {
                'train_size': len(self.dataset['train']),
                'validation_size': len(self.dataset['validation'])
            },
            'final_evaluation': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                                 for k, v in eval_result.items()},
            'prediction_examples': {
                'random_indices': random_indices,
                'samples': predict_results,
                'accuracy': sum(r['is_correct'] for r in predict_results) / len(predict_results)
            },
            'training_history': self.metrics_logger.to_dict()
        }
        
        results_path = os.path.join(self.output_dir, 'experiment_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"完整实验结果已保存到: {results_path}")
        
        # 额外保存一个简化版的结果摘要
        summary = {
            'experiment_id': self.experiment_id,
            'task': self.task_name,
            'final_metrics': {k: v for k, v in eval_result.items() if k.startswith('eval_')},
            'prediction_accuracy': sum(r['is_correct'] for r in predict_results) / len(predict_results),
            'total_time': self.train_end_time - self.train_start_time if self.train_end_time else None
        }
        
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return results_path
    
    def run(self):
        """运行完整的训练流程"""
        print(f"\n开始实验: {self.experiment_id}\n")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 加载模型
        self.load_model()
        
        # 3. 预处理数据
        self.preprocess_data()
        
        # 4. 训练
        train_result, eval_result = self.train()
        
        # 5. 保存模型
        self.save_model()
        
        # 6. 预测示例
        predict_results, random_indices = self.predict_examples()
        
        # 7. 绘制训练曲线
        self.plot_training_history()
        
        # 8. 保存结果
        self.save_results(eval_result, predict_results, random_indices)
        
        print(f"\n{'='*60}")
        print(f"实验完成！")
        print(f"实验ID: {self.experiment_id}")
        print(f"所有结果保存在: {self.output_dir}")
        print(f"  - 模型: {self.model_save_path}")
        print(f"  - 实验结果: experiment_results.json")
        print(f"  - 训练曲线: training_curves.png")
        print(f"  - 检查点: checkpoints/")
        print(f"{'='*60}\n")
        
        return {
            'experiment_id': self.experiment_id,
            'output_dir': self.output_dir,
            'model_path': self.model_save_path,
            'eval_result': eval_result,
            'predict_results': predict_results
        }


if __name__ == '__main__':
    # 测试
    trainer = BertTrainer(task_name='mrpc', seed=42)
    trainer.run()

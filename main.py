"""
BERT微调项目主程序
支持训练和推理功能
"""

import os
import sys
import argparse

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 设置缓存目录到项目目录，避免下载到C盘
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')

from src.trainers.bert_trainer import BertTrainer
from src.inference.predictor import BertPredictor


def print_banner():
    """打印程序标题"""
    print("\n" + "="*60)
    print("       BERT Fine-tuning for NLP Tasks")
    print("       支持任务: MRPC (句子对语义等价) | SST-2 (情感分类)")
    print("="*60 + "\n")


def train_task(args):
    """训练任务"""
    trainer = BertTrainer(
        task_name=args.task,
        seed=args.seed,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    result = trainer.run()
    return result


def predict_task(args):
    """推理预测"""
    predictor = BertPredictor(task_name=args.task)
    predictor.interactive_predict()


def train_all():
    """训练所有任务"""
    print("开始训练所有任务...\n")
    
    results = {}
    
    # 训练MRPC（数据量小，先运行）
    print("\n" + "="*60)
    print("训练 MRPC 任务")
    print("="*60)
    mrpc_trainer = BertTrainer(task_name='mrpc', seed=42)
    results['mrpc'] = mrpc_trainer.run()
    
    # 训练SST-2
    print("\n" + "="*60)
    print("训练 SST-2 任务")
    print("="*60)
    sst2_trainer = BertTrainer(task_name='sst2', seed=42)
    results['sst2'] = sst2_trainer.run()
    
    print("\n" + "="*60)
    print("所有任务训练完成！")
    print("="*60)
    
    for task, result in results.items():
        print(f"\n{task.upper()}:")
        print(f"  实验ID: {result['experiment_id']}")
        print(f"  结果目录: {result['output_dir']}")
    
    return results


def interactive_mode():
    """交互式模式"""
    print_banner()
    
    while True:
        print("\n请选择操作:")
        print("  1. 训练 MRPC 模型")
        print("  2. 训练 SST-2 模型")
        print("  3. 训练所有模型")
        print("  4. MRPC 推理预测")
        print("  5. SST-2 推理预测")
        print("  0. 退出")
        
        choice = input("\n请输入选项 [0-5]: ").strip()
        
        if choice == '0':
            print("再见！")
            break
        elif choice == '1':
            trainer = BertTrainer(task_name='mrpc', seed=42)
            trainer.run()
        elif choice == '2':
            trainer = BertTrainer(task_name='sst2', seed=42)
            trainer.run()
        elif choice == '3':
            train_all()
        elif choice == '4':
            try:
                predictor = BertPredictor(task_name='mrpc')
                predictor.interactive_predict()
            except FileNotFoundError as e:
                print(f"错误: {e}")
                print("请先训练MRPC模型")
        elif choice == '5':
            try:
                predictor = BertPredictor(task_name='sst2')
                predictor.interactive_predict()
            except FileNotFoundError as e:
                print(f"错误: {e}")
                print("请先训练SST-2模型")
        else:
            print("无效选项，请重新输入")


def main():
    parser = argparse.ArgumentParser(description='BERT微调项目')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'all', 'interactive'],
                        default='interactive', help='运行模式')
    parser.add_argument('--task', type=str, choices=['mrpc', 'sst2'],
                        help='任务名称')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=2,
                        help='训练轮数')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'train':
        if not args.task:
            print("请指定任务: --task mrpc 或 --task sst2")
            return
        print_banner()
        train_task(args)
    elif args.mode == 'predict':
        if not args.task:
            print("请指定任务: --task mrpc 或 --task sst2")
            return
        print_banner()
        predict_task(args)
    elif args.mode == 'all':
        print_banner()
        train_all()


if __name__ == '__main__':
    main()

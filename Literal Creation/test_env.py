# test_env.py
import torch
import transformers
from transformers import BertModel, BertTokenizer

# 1. 检查CUDA是否可用
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"当前GPU: {torch.cuda.current_device() if torch.cuda.is_available() else 'None'}")
print(f"GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 2. 测试BERT模型加载
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    if torch.cuda.is_available():
        model = model.to('cuda:0')
    print("BERT模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
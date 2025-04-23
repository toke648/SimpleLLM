import json

# 读取原始数据
with open('c:\\Users\\16673\\Desktop\\ollama 大模型微调\\train.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 转换数据格式
converted_data = []
for line in lines:
    data = json.loads(line)
    for conversation in data['conversation']:
        converted_data.append({
            "instruction": conversation['human'],
            "input": "",
            "output": conversation['assistant']
        })

# 保存为新的JSON文件
with open('c:\\Users\\16673\\Desktop\\ollama 大模型微调\\magic_conch.json', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)
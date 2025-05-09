# import requests
# from playsound import playsound
# url = 'http://cloud-vms.volcengineapi.com?Action=OpenCreateTts&Version=2022-01-01'  # 请根据实际API文档修改
# headers = {
#     'Authorization': 'dQxoR6t4LlXrqYhQ8o5-Rbf3fRP4-Uep',  # 替换为你的访问令牌
#     'Content-Type': 'application/json'
# }
# data = {
#     'text': '你好，世界！',
#     'voice': 'BV001_streaming',  # 替换为所需的声音
#     'output_format': 'mp3'  # 输出格式可以是mp3或其他支持的格式
# }

# response = requests.post(url, headers=headers, json=data)

# if response.status_code == 200:
#     with open('output.mp3', 'wb') as f:
#         f.write(response.content)
#     print("音频已保存为output.mp3")
# else:
#     print("请求失败:", response.text)

# # 指定音频文件的路径
# audio_file = 'output.mp3'  # 或者使用其他音频文件的路径

# # 播放音频
# playsound(audio_file)


# 火山引擎语音服务调用

import requests


# api_key = 'dQxoR6t4LlXrqYhQ8o5-Rbf3fRP4-Uep'
# base_url = 'http://cloud-vms.volcengineapi.com?Action=OpenCreateTts&Version=2022-01-01'

# repose = requests.post(base_url)

# print(repose.text)


import requests  
import datetime  
import hmac  
import hashlib  
import base64  

# 填写必要的身份验证信息  
service_name = "vms"  # 服务名称  
region = "cn-north-1"  # 区域  
access_key = 'dQxoR6t4LlXrqYhQ8o5-Rbf3fRP4-Uep'  # 替换为您的访问密钥  
secret_key = 'HvnvUGpwxhV6RMELMH6DzEyzf4w_73A4'  # 替换为您的安全密钥  

# 生成当前UTC时间  
date = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')  

# 生成签名  
def create_signature(secret_key, date):  
    # 原始字符串  
    string_to_sign = f"HMAC-SHA256 Credential={access_key}/{date}/vms"  
    signature = hmac.new(secret_key.encode(), string_to_sign.encode(), hashlib.sha256).digest()  
    return base64.b64encode(signature).decode()  

# 准备请求的URL和头部  
url = 'http://cloud-vms.volcengineapi.com?Action=OpenCreateTts&Version=2022-01-01'  
headers = {  
    'ServiceName': service_name,  
    'Region': region,  
    'Content-Type': 'application/json',  
    'X-Date': date,  
    'Authorization': f"HMAC-SHA256 Credential={access_key}, Signature={create_signature(secret_key, date)}"  
}  

# 准备请求的数据  
data = {  
    "Name": "ttsTest",  
    "TtsTemplateContent": "你好",  
    "Type": 1  
}  

# 发送POST请求  
response = requests.post(url, headers=headers, json=data)  

# 处理响应  
if response.status_code == 200:  
    print("请求成功:", response.json())  
else:  
    print("请求失败:", response.json())



# # #coding=utf-8

# # '''
# # requires Python 3.6 or later

# # pip install asyncio
# # pip install websockets

# # '''

# # import asyncio
# # import websockets
# # import uuid
# # import json
# # import gzip
# # import copy

# # MESSAGE_TYPES = {11: "audio-only server response", 12: "frontend server response", 15: "error message from server"}
# # MESSAGE_TYPE_SPECIFIC_FLAGS = {0: "no sequence number", 1: "sequence number > 0",
# #                                2: "last message from server (seq < 0)", 3: "sequence number < 0"}
# # MESSAGE_SERIALIZATION_METHODS = {0: "no serialization", 1: "JSON", 15: "custom type"}
# # MESSAGE_COMPRESSIONS = {0: "no compression", 1: "gzip", 15: "custom compression method"}

# # appid = "7318254534"
# # token = "dQxoR6t4LlXrqYhQ8o5-Rbf3fRP4-Uep"
# # cluster = "volcano_tts"
# # voice_type = "volcano_tts"
# # host = "openspeech.bytedance.com"
# # api_url = f"wss://{host}/api/v1/tts/ws_binary"
# # user_id = "2103445088"

# # # version: b0001 (4 bits)
# # # header size: b0001 (4 bits)
# # # message type: b0001 (Full client request) (4bits)
# # # message type specific flags: b0000 (none) (4bits)
# # # message serialization method: b0001 (JSON) (4 bits)
# # # message compression: b0001 (gzip) (4bits)
# # # reserved data: 0x00 (1 byte)
# # default_header = bytearray(b'\x11\x10\x11\x00')

# # request_json = {
# #     "app": {
# #         "appid": appid,
# #         "token": token,
# #         "cluster": cluster
# #     },
# #     "user": {
# #         "uid": user_id
# #     },
# #     "audio": {
# #         "voice_type": voice_type,
# #         "encoding": "mp3",
# #         "speed_ratio": 1.0,
# #         "volume_ratio": 1.0,
# #         "pitch_ratio": 1.0,
# #     },
# #     "request": {
# #         "reqid": "xxx",
# #         "text": "字节跳动语音合成。",
# #         "text_type": "plain",
# #         "operation": "submit",  # 如果你是提交请求，使用 submit 操作
# #     }
# # }


# # async def test_submit():
# #     submit_request_json = copy.deepcopy(request_json)
# #     submit_request_json["audio"]["voice_type"] = voice_type
# #     submit_request_json["request"]["reqid"] = str(uuid.uuid4())
# #     submit_request_json["request"]["operation"] = "submit"
# #     payload_bytes = str.encode(json.dumps(submit_request_json))
# #     payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
# #     full_client_request = bytearray(default_header)
# #     full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
# #     full_client_request.extend(payload_bytes)  # payload
# #     print("\n------------------------ test 'submit' -------------------------")
# #     print("request json: ", submit_request_json)
# #     print("\nrequest bytes: ", full_client_request)
# #     file_to_save = open("test_submit.mp3", "wb")
# #     header = {"Authorization": f"Bearer; {token}"}
# #     async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
# #         await ws.send(full_client_request)
# #         while True:
# #             res = await ws.recv()
# #             done = parse_response(res, file_to_save)
# #             if done:
# #                 file_to_save.close()
# #                 break
# #         print("\nclosing the connection...")


# # async def test_query():
# #     query_request_json = copy.deepcopy(request_json)
# #     query_request_json["audio"]["voice_type"] = voice_type
# #     query_request_json["request"]["reqid"] = str(uuid.uuid4())
# #     query_request_json["request"]["operation"] = "query"
# #     payload_bytes = str.encode(json.dumps(query_request_json))
# #     payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
# #     full_client_request = bytearray(default_header)
# #     full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
# #     full_client_request.extend(payload_bytes)  # payload
# #     print("\n------------------------ test 'query' -------------------------")
# #     print("request json: ", query_request_json)
# #     print("\nrequest bytes: ", full_client_request)
# #     file_to_save = open("test_query.mp3", "wb")
# #     header = {"Authorization": f"Bearer; {token}"}
# #     async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
# #         await ws.send(full_client_request)
# #         res = await ws.recv()
# #         parse_response(res, file_to_save)
# #         file_to_save.close()
# #         print("\nclosing the connection...")


# # def parse_response(res, file):
# #     print("--------------------------- response ---------------------------")
# #     # print(f"response raw bytes: {res}")
# #     protocol_version = res[0] >> 4
# #     header_size = res[0] & 0x0f
# #     message_type = res[1] >> 4
# #     message_type_specific_flags = res[1] & 0x0f
# #     serialization_method = res[2] >> 4
# #     message_compression = res[2] & 0x0f
# #     reserved = res[3]
# #     header_extensions = res[4:header_size*4]
# #     payload = res[header_size*4:]
# #     print(f"            Protocol version: {protocol_version:#x} - version {protocol_version}")
# #     print(f"                 Header size: {header_size:#x} - {header_size * 4} bytes ")
# #     print(f"                Message type: {message_type:#x} - {MESSAGE_TYPES[message_type]}")
# #     print(f" Message type specific flags: {message_type_specific_flags:#x} - {MESSAGE_TYPE_SPECIFIC_FLAGS[message_type_specific_flags]}")
# #     print(f"Message serialization method: {serialization_method:#x} - {MESSAGE_SERIALIZATION_METHODS[serialization_method]}")
# #     print(f"         Message compression: {message_compression:#x} - {MESSAGE_COMPRESSIONS[message_compression]}")
# #     print(f"                    Reserved: {reserved:#04x}")
# #     if header_size != 1:
# #         print(f"           Header extensions: {header_extensions}")
# #     if message_type == 0xb:  # audio-only server response
# #         if message_type_specific_flags == 0:  # no sequence number as ACK
# #             print("                Payload size: 0")
# #             return False
# #         else:
# #             sequence_number = int.from_bytes(payload[:4], "big", signed=True)
# #             payload_size = int.from_bytes(payload[4:8], "big", signed=False)
# #             payload = payload[8:]
# #             print(f"             Sequence number: {sequence_number}")
# #             print(f"                Payload size: {payload_size} bytes")
# #         file.write(payload)
# #         if sequence_number < 0:
# #             return True
# #         else:
# #             return False
# #     elif message_type == 0xf:
# #         code = int.from_bytes(payload[:4], "big", signed=False)
# #         msg_size = int.from_bytes(payload[4:8], "big", signed=False)
# #         error_msg = payload[8:]
# #         if message_compression == 1:
# #             error_msg = gzip.decompress(error_msg)
# #         error_msg = str(error_msg, "utf-8")
# #         print(f"          Error message code: {code}")
# #         print(f"          Error message size: {msg_size} bytes")
# #         print(f"               Error message: {error_msg}")
# #         return True
# #     elif message_type == 0xc:
# #         msg_size = int.from_bytes(payload[:4], "big", signed=False)
# #         payload = payload[4:]
# #         if message_compression == 1:
# #             payload = gzip.decompress(payload)
# #         print(f"            Frontend message: {payload}")
# #     else:
# #         print("undefined message type!")
# #         return True


# # if __name__ == '__main__':
# #     loop = asyncio.get_event_loop()
# #     loop.run_until_complete(test_submit())
# #     loop.run_until_complete(test_query())



# import asyncio
# import websockets
# import json
# import uuid

# # 配置项，请替换成你自己的 appid 和 token
# APP_ID = "appid123"  # 你的 appid
# ACCESS_TOKEN = "dQxoR6t4LlXrqYhQ8o5-Rbf3fRP4-Uep"  # 你的 access_token
# CLUSTER = "volcano_tts"  # 集群信息，通常设置为 "volcano_tts"

# # 音频设置
# VOICE_TYPE = "zh_male_M392_conversation_wvae_bigtts"  # 音色类型
# ENCODING = "mp3"  # 编码格式，可以选择 pcm / mp3 / wav 等
# SPEED_RATIO = 1.0  # 语速，默认 1.0

# # 需要合成的文本
# TEXT = "字节跳动语音合成"  # 你需要转换成语音的文本

# # 请求的唯一标识符
# REQ_ID = str(uuid.uuid4())  # 使用 UUID 生成唯一标识符


# # WebSocket 客户端连接并发送请求
# async def request_tts():
#     # WebSocket 服务器地址
#     api_url = "wss://openspeech.bytedance.com/api/v1/tts/ws_binary"

#     # 请求的 JSON 数据
#     request_data = {
#         "app": {
#             "appid": APP_ID,
#             "token": ACCESS_TOKEN,
#             "cluster": CLUSTER
#         },
#         "user": {
#             "uid": "uid123"  # 用户标识符，可以随意设置
#         },
#         "audio": {
#             "voice_type": VOICE_TYPE,
#             "encoding": ENCODING,
#             "speed_ratio": SPEED_RATIO
#         },
#         "request": {
#             "reqid": REQ_ID,
#             "text": TEXT,
#             "operation": "submit"  # 使用流式合成时，设置为 submit
#         }
#     }

#     # 请求头部
#     headers = {
#         "Authorization": f"Bearer; {ACCESS_TOKEN}"  # Bearer Token 认证
#     }

#     # 连接 WebSocket 并发送请求
#     async with websockets.connect(api_url, extra_headers=headers, ping_interval=None) as ws:
#         # 发送数据请求
#         await ws.send(json.dumps(request_data))

#         # 接收返回的音频数据（base64 编码）
#         response = await ws.recv()
#         response_data = json.loads(response)

#         # 返回数据解析
#         if response_data.get("code") == 3000:
#             # 合成成功，解析音频数据
#             audio_base64 = response_data.get("data")
#             print("合成成功，音频数据：", audio_base64)
#             # 可选择将 base64 编码的音频数据保存为文件
#             with open("output.mp3", "wb") as audio_file:
#                 audio_file.write(bytes(audio_base64, encoding='utf-8'))
#             print("音频已保存为 output.mp3")
#         else:
#             # 如果失败，输出错误信息
#             print(f"合成失败，错误信息：{response_data.get('message')}")

# # 运行请求
# asyncio.run(request_tts())


import asyncio
import websockets
import uuid
import json
import gzip
import copy

# 定义消息类型和标识等
MESSAGE_TYPES = {11: "audio-only server response", 12: "frontend server response", 15: "error message from server"}
MESSAGE_TYPE_SPECIFIC_FLAGS = {0: "no sequence number", 1: "sequence number > 0",
                               2: "last message from server (seq < 0)", 3: "sequence number < 0"}
MESSAGE_SERIALIZATION_METHODS = {0: "no serialization", 1: "JSON", 15: "custom type"}
MESSAGE_COMPRESSIONS = {0: "no compression", 1: "gzip", 15: "custom compression method"}

# 配置信息
appid = "7318254534"
token = "dQxoR6t4LlXrqYhQ8o5-Rbf3fRP4-Uep"
cluster = "volcano_tts"
voice_type = "BV001_streaming"
host = "openspeech.bytedance.com"
api_url = f"wss://{host}/api/v1/tts/ws_binary"
user_id = "2103445088"

# 默认报头，包含协议版本等信息
default_header = bytearray(b'\x11\x10\x11\x00')

# 请求 JSON 模板
request_json = {
    "app": {
        "appid": appid,
        "token": token,
        "cluster": cluster
    },
    "user": {
        "uid": user_id
    },
    "audio": {
        "voice_type": voice_type,
        "encoding": "mp3",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
    },
    "request": {
        "reqid": "xxx",
        "text": "字节跳动语音合成。",
        "text_type": "plain",
        "operation": "submit",  # 默认使用 submit 操作
    }
}

# 提交请求函数
async def submit_request():
    submit_request_json = copy.deepcopy(request_json)
    submit_request_json["request"]["reqid"] = str(uuid.uuid4())
    submit_request_json["request"]["operation"] = "submit"
    payload_bytes = str.encode(json.dumps(submit_request_json))
    payload_bytes = gzip.compress(payload_bytes)  # 压缩请求内容
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # 添加负载大小
    full_client_request.extend(payload_bytes)  # 添加负载数据
    print("\n------------------------ Submit Request -------------------------")
    print("Request JSON: ", submit_request_json)
    print("\nRequest Bytes: ", full_client_request)

    file_to_save = open("test_submit.mp3", "wb")
    header = {"Authorization": f"Bearer; {token}"}

    async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        while True:
            res = await ws.recv()
            done = parse_response(res, file_to_save)
            if done:
                file_to_save.close()
                break
        print("\nClosing the connection...")

# 查询请求函数
async def query_request():
    query_request_json = copy.deepcopy(request_json)
    query_request_json["request"]["reqid"] = str(uuid.uuid4())
    query_request_json["request"]["operation"] = "query"
    payload_bytes = str.encode(json.dumps(query_request_json))
    payload_bytes = gzip.compress(payload_bytes)  # 压缩请求内容
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # 添加负载大小
    full_client_request.extend(payload_bytes)  # 添加负载数据
    print("\n------------------------ Query Request -------------------------")
    print("Request JSON: ", query_request_json)
    print("\nRequest Bytes: ", full_client_request)

    file_to_save = open("test_query.mp3", "wb")
    header = {"Authorization": f"Bearer; {token}"}

    async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        res = await ws.recv()
        parse_response(res, file_to_save)
        file_to_save.close()
        print("\nClosing the connection...")

# 解析响应数据
def parse_response(res, file):
    print("--------------------------- Response ---------------------------")
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0f
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0f
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0f
    reserved = res[3]
    header_extensions = res[4:header_size*4]
    payload = res[header_size*4:]

    print(f"Protocol version: {protocol_version:#x} - version {protocol_version}")
    print(f"Header size: {header_size:#x} - {header_size * 4} bytes")
    print(f"Message type: {message_type:#x} - {MESSAGE_TYPES.get(message_type, 'Unknown')}")
    print(f"Message type specific flags: {message_type_specific_flags:#x} - {MESSAGE_TYPE_SPECIFIC_FLAGS.get(message_type_specific_flags, 'Unknown')}")
    print(f"Message serialization method: {serialization_method:#x} - {MESSAGE_SERIALIZATION_METHODS.get(serialization_method, 'Unknown')}")
    print(f"Message compression: {message_compression:#x} - {MESSAGE_COMPRESSIONS.get(message_compression, 'Unknown')}")
    print(f"Reserved: {reserved:#04x}")

    # 处理音频响应
    if message_type == 0xb:  # audio-only server response
        if message_type_specific_flags == 0:  # no sequence number as ACK
            print("Payload size: 0")
            return False
        else:
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload = payload[8:]
            print(f"Sequence number: {sequence_number}")
            print(f"Payload size: {payload_size} bytes")
        file.write(payload)
        if sequence_number < 0:
            return True
        else:
            return False
    elif message_type == 0xf:  # Error message from server
        code = int.from_bytes(payload[:4], "big", signed=False)
        msg_size = int.from_bytes(payload[4:8], "big", signed=False)
        error_msg = payload[8:]
        if message_compression == 1:
            error_msg = gzip.decompress(error_msg)
        error_msg = str(error_msg, "utf-8")
        print(f"Error code: {code}")
        print(f"Error message size: {msg_size} bytes")
        print(f"Error message: {error_msg}")
        return True
    elif message_type == 0xc:  # Frontend message
        msg_size = int.from_bytes(payload[:4], "big", signed=False)
        payload = payload[4:]
        if message_compression == 1:
            payload = gzip.decompress(payload)
        print(f"Frontend message: {payload}")
    else:
        print("Undefined message type!")
        return True

# 主程序执行
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(submit_request())
    loop.run_until_complete(query_request())

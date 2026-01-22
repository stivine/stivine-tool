from napcat_client import NapCatClient
from message_builder import build_message, build_forward_message

client = NapCatClient(
    api_url="http://127.0.0.1:3000",
    group_id=561410928,
    token=""
)

# msg = build_message(
#     text="你好",
#     image_urls=["http://i2.hdslb.com/bfs/archive/49e8f9db109ea57f43b9c518424fb1faa8786d9f.jpg"],
# )

# client.send_group_msg(msg)

with open('source.txt', 'r', encoding='utf-8') as file:
    content = file.read()
print(content)
print("-----")
msg = build_forward_message(content)
x = client.send_forward_msg(msg)
print(x)
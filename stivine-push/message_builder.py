def build_message(text: str, image_urls: list[str] = None):
    message = []

    if text:
        message.append({
            "type": "text",
            "data": {"text": text}
        })

    if image_urls:
        for url in image_urls:
            message.append({
                "type": "text",
                "data": {"text": "\n"}
            })
            message.append({
                "type": "image",
                "data": {"file": url}
            })

    return message

def build_forward_message(content: str):
    node = {
        "type": "node",
        "data": {
            "nickname": "八千代的十年b站考古计划",
            "user_id": 0,   # 可以填机器人 QQ
            "content": []
        }
    }

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("http"):
            node["data"]["content"].append({
                "type": "image",
                "data": {
                    "file": line
                }
            })
        else:
            node["data"]["content"].append({
                "type": "text",
                "data": {
                    "text": line+"\n"
                }
            })

    return [node]   # ⚠️ messages 是一个数组

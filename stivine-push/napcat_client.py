import requests
import json


class NapCatClient:
    def __init__(self, api_url, group_id, token=None):
        self.api_url = api_url.rstrip("/")
        self.group_id = str(group_id)
        self.token = token

    def send_group_msg(self, message):
        payload = {
            "group_id": self.group_id,
            "message": message
        }

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = requests.post(
            f"{self.api_url}/send_msg",
            headers=headers,
            data=json.dumps(payload)
        )

        resp.raise_for_status()
        return resp.json()
    def send_forward_msg(self, forward_msg):
        payload = {
            "group_id": self.group_id,
            "messages": forward_msg
        }
        

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = requests.post(
            f"{self.api_url}/send_group_forward_msg",
            headers=headers,
            data=json.dumps(payload)
        )

        resp.raise_for_status()
        return resp.json()
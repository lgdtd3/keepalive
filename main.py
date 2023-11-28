import requests
import json
import os
import time
# https://github.com/yese2023/openfrp_signin
# https://github.com/yese2023/openfrp_signin
# https://github.com/yese2023/openfrp_signin
# https://github.com/yese2023/openfrp_signin
# https://github.com/yese2023/openfrp_signin
def perform_user_sign(session, authorization):
    user_sign_url = "https://of-dev-api.bfsea.xyz/frp/api/userSign"

    user_sign_payload = {
        "session": session
    }
    user_sign_headers = {
        "Authorization": authorization,
        "Content-Type": "application/json"
    }

    user_sign_response = requests.post(user_sign_url, data=json.dumps(user_sign_payload), headers=user_sign_headers)

    if user_sign_response.status_code == 200:
        user_sign_data = user_sign_response.json()
        if user_sign_data.get('flag', False):
            # 提取签到信息
            sign_message = user_sign_data.get('data', '')
            return f"签到成功：{sign_message}"
        else:
            return f"签到失败：{user_sign_data.get('msg', '未知错误')}"
    else:
        return f"用户签到请求失败：{user_sign_response.text}"

# 登录请求
def login_user(account, password):
    login_url = "https://of-dev-api.bfsea.xyz/user/login"
    login_payload = {
        "user": account,
        "password": password
    }

    login_headers = {
        "Content-Type": "application/json"
    }

    login_response = requests.post(login_url, data=json.dumps(login_payload), headers=login_headers)

    if login_response.status_code == 200:
        data_response = login_response.json()
        data_value = data_response.get('data', '')
        authorization = login_response.headers.get('Authorization', '')

        if data_value:
            return data_value, authorization
        else:
            return None, None
    else:
        return None, None

# 读取账号密码文件并进行登录和签到
while True:

    for account_info :
        account = "zxz030180@gmail.com"
        password = "ZXZ@030108"

        data_value, authorization = login_user(account, password)

        if data_value:
            print(f"登录成功：账号 {account} 将在14400秒后刷新登录")
            print(f"Authorization头内容：{authorization}")
            print(f"session内容：{data_value}")

            sign_result = perform_user_sign(data_value, authorization)
            print(sign_result)
        else:
            print(f"登录失败：账号 {account}")

    time.sleep(14400)

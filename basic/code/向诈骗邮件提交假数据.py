# 对钓鱼网站进行恶意提交
import requests
import string
import random
import time
import json
print("hello world")
URL = "https://tencent10288.steampworerd.com/admin/reg.php?mb=1"
SLEEPTIME = 0.5

# 装载弱密码

url_list = [
    r'https://raw.githubusercontent.com/rootphantomer/Blasting_dictionary/master/%E5%AD%97%E5%85%B8.txt',
    r'https://raw.githubusercontent.com/rootphantomer/Blasting_dictionary/master/top500%E5%A7%93%E5%90%8D%E7%BB%84%E5%90%88.txt',
    r'https://raw.githubusercontent.com/rootphantomer/Blasting_dictionary/master/%E8%87%AA%E5%B7%B1%E6%94%B6%E9%9B%86%E7%9A%84%E5%AF%86%E7%A0%81.txt']

pass_list = []
[pass_list.extend(requests.get(u).text.split()) for u in url_list]

success = 0
fail = 0

print('向', URL, '提交假数据')

while True:

    # generate
    user = ''.join(random.choices(
        string.digits, k=random.randint(8, 13)))+'@qq.com'
    password = random.choice(pass_list)
    tid = ''.join(random.choices(string.digits, k=4))
    data = {
        'user': user,
        'pass': password,
        'mb': 1,
        'tid': tid,
    }
    try:
        r = requests.post(URL, data=data)
        if r.status_code == 200:
            res = json.loads(r.content.decode(r.apparent_encoding))
            if res['status']:
                success += 1
            else:
                fail += 1
    except:
        fail += 1

    print('\r成功次数:', success, ',', '失败次数', fail, '[', 'fake_user', user,
          'fake_pass', password, ']', end='                      ')

    time.sleep(SLEEPTIME)
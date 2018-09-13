# -*- coding: utf-8 -*-

import sys
import getopt
from time import sleep


import requests
import psycopg2


# 获获取命令行参数
from_mbid = ''
opts, args = getopt.getopt(sys.argv[1:], 'hf:', ('help', 'from='))
for op, value in opts:
    if op in ('-h', '--help'):
        print('-h,\t--help:\tshow help')
        print('-f,\t--from:\tfetch from a given mbid')
        exit()
    elif op in ('-f', '--from'):
        from_mbid = value

# 打开数据库连接，获取所有数据
conn = psycopg2.connect("dbname=musicbrainz_db user=postgres")
cur = conn.cursor()
cur.execute('SELECT mbid FROM done.done;')
mbids = cur.fetchall()

# 检查是否从指定 mbid 开始获取
from_index = 0
if from_mbid:
    for i, (mbid,) in enumerate(mbids):
        if from_mbid == mbid:
            from_index = i
            break
    if mbids[from_index][0] != from_mbid:
        print('cannot found given mbid')

# 开始找
total = len(mbids) - from_index
request_failed_count_in_succession = 0
for i, (mbid,) in enumerate(mbids[from_index:]):
    if i % 100 == 0:
        conn.commit()
        print('committed')

    print('(%d/%d)fetching %s' % (i, total, mbid))

    # 发出请求，自动重试
    request_success = False
    while not request_success:
        request_success = True
        try:
            r = requests.get('https://coverartarchive.org/release-group/%s/front-250' % (mbid,))
        except:
            request_success = False
            print('Request failed. Wait 10 seconds then retry...')
            sleep(10)

    if r.status_code == 200:
        cur.execute('UPDATE done.done SET cover=%s WHERE mbid=%s', (psycopg2.Binary(r.content), mbid))
        request_failed_count_in_succession = 0
    elif r.status_code == 404:
        print('no cover')
        request_failed_count_in_succession = 0
    else:
        print('error!', r.status_code, r.reason)
        err_message = mbid + ': ' + str(r.status_code) + '  ' + r.reason + '\n'
        with open('error_mbids', 'a') as f:
            f.write(err_message)
        # 如果连续 5 条数据不明原因失败，停止拉取，需手动检查 API 可用性
        request_failed_count_in_succession += 1
        if request_failed_count_in_succession >=5:
            print('request failed 5 times in succession, stoped')
            break

    sys.stdout.flush()

conn.commit()
print('all done!')
cur.close()
conn.close()

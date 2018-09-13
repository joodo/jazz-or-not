import random
import sys
from StringIO import StringIO

import psycopg2
from PIL import Image, ImageFile
import numpy as np
import paddle.fluid as fluid
import paddle

from smote import Smote


ImageFile.LOAD_TRUNCATED_IMAGES = True

# 训练、测试、验证数据比例
DATA_PROPORTION = {'train': 8, 'test': 1, 'verify': 1}  # (train, test, verify)
# 风格关键词
TAG_KEYWORD = 'jazz'
# 图片缩放后大小
IMAGE_RESIZE_SIZE = 128
# 人工数据倍数
OVER_SIMPLING_N = 6

conn = psycopg2.connect("dbname=musicbrainz_db user=postgres")
cur = conn.cursor()

print('Loading database...')
sys.stdout.flush()
data_all = []
positive_covers = []
cur.execute('SELECT cover,tags FROM done.done;')
#cur.execute('SELECT cover,tags FROM done.done LIMIT 1000;')
for cover, tag in cur.fetchall():
    cover = Image.open(StringIO(cover))
    cover = cover.convert(mode='RGB')
    cover = cover.resize((IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE),
                         Image.ANTIALIAS)
    cover = np.array(cover).astype(np.float32)
    cover = cover / 255.0
    # The storage order of the loaded image is W(width),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them
    cover = cover.transpose((2, 0, 1))

    data_all.append((cover, TAG_KEYWORD in tag))

data = {}
t = len(data_all) / sum(DATA_PROPORTION.values())
train_count = int(DATA_PROPORTION['train'] * t)
test_count = int(DATA_PROPORTION['test'] * t)
data['train'] = data_all[: train_count]
data['test'] = data_all[train_count : train_count+test_count]
data['verify'] = data_all[train_count+test_count :]

print('Creating more simples by SMOTE...')
sys.stdout.flush()
cover_shape = (3, IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE)
positive_covers = [record[0].reshape(-1) for record in data['train'] if record[1]]
smote = Smote(np.array(positive_covers), N=OVER_SIMPLING_N, k=5)
over_simples = smote.over_sampling()
data['train'] += [(np.reshape(simple, cover_shape), True) for simple in over_simples]

print('Shuffling data...')
sys.stdout.flush()
for d in data.values():
    random.shuffle(d)

print('Exporting data.info ...')
sys.stdout.flush()
with open('data.info', 'w') as f:
    f.write(str({ k: len(v) for k,v in data.items() }))

print('Preparing export recordio file...')
sys.stdout.flush()
data_shape = [-1, 3, IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE]
image_layer = fluid.layers.data(name='cover',
                                shape=data_shape, dtype='float32')
label_layer = fluid.layers.data(name='label', shape=[1], dtype='float32')
feeder = fluid.DataFeeder(place=fluid.CPUPlace(),
                          feed_list=[image_layer, label_layer])

for k in data.keys():
    print('Exporting %s.recordio ...' % (k,))
    sys.stdout.flush()

    def reader_creator():
        def reader():
            for record in data[k]:
                yield (record[0], 1.0 if record[1] else 0.0)
        return reader
    reader = paddle.batch(reader_creator(), batch_size=1)
    fluid.recordio_writer.convert_reader_to_recordio_file(
        filename = k + '.recordio',
        reader_creator = reader,
        feeder = feeder,
    )

print('All done!')


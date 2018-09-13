import sys
import math

import paddle
import paddle.fluid as fluid
from sklearn import metrics, preprocessing
import numpy as np

from resnet import resnet
from log_loss import log_loss


USE_CUDA = True
IMAGE_RESIZE_SIZE = 128
BATCH_SIZE = 32
PASS_NUM = 100

# Layers
def network(is_test):
    data_shape = [-1, 3, IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE]

    if not is_test:
        file_obj = fluid.layers.open_files(
              filenames=['train.recordio'],
              shapes=[data_shape, [-1, 1]],
              lod_levels=[0, 0],
              dtypes=['float32', 'float32'],
              pass_num=PASS_NUM,
        )
        file_obj = fluid.layers.double_buffer(file_obj)
        file_obj = fluid.layers.shuffle(file_obj, buffer_size=8192)
        file_obj = fluid.layers.batch(file_obj, batch_size=BATCH_SIZE)
    else:
        file_obj = fluid.layers.open_files(
              filenames=['test.recordio'],
              shapes=[data_shape, [-1, 1]],
              lod_levels=[0, 0],
              dtypes=['float32', 'float32'],
        )
    with fluid.unique_name.guard():
        image_layer, label_layer = fluid.layers.read_file(file_obj)
        prediction_layer = resnet(image_layer)
        cost_layer = log_loss(input=prediction_layer, label=label_layer)
        avg_cost_layer = fluid.layers.mean(cost_layer)
    return avg_cost_layer, prediction_layer, label_layer

# Train program
train_cost_layer, train_prediction_layer, train_label_layer = network(is_test=False)
optimizer = fluid.optimizer.Adam(
    learning_rate=0.001,
    regularization=fluid.regularizer.L2Decay(0.8),
)
optimizer.minimize(train_cost_layer)

# Test program
test_program = fluid.Program()
test_startup = fluid.Program()
with fluid.program_guard(test_program, test_startup):
    test_cost_layer, test_prediction_layer, test_label_layer = network(is_test=True)

# Executor
place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
exe.run(test_startup)

# Train!!
with open('data.info') as f:
    data_info = eval(f.read())

batch_count = int(math.ceil(data_info['train'] / BATCH_SIZE))

binarizer = preprocessing.Binarizer(threshold=0.5)
def more_info(prediction, label):
    print('=== MORE INFO ===')
    y_true = np.array([1 if x > 0.5 else 0 for x in label])
    y_score = np.array(prediction)
    #print('label:')
    #print(y_true)
    #print('prediction:')
    #print(y_score)
    print('label mean: %f' % (np.mean(y_true), ))
    print('prediction mean: %f' % (np.mean(y_score), ))
    acc = metrics.accuracy_score(y_true, binarizer.transform(y_score))
    print('Accuracy: %f' % (acc, ))
    roc_auc = metrics.roc_auc_score(y_true, y_score)
    print('ROC AUC: %f' % (roc_auc, ))
    return acc, roc_auc


for pass_id in xrange(PASS_NUM):
    for batch_id in xrange(batch_count):
        cost, prediction, label = exe.run(fetch_list=[train_cost_layer, train_prediction_layer, train_label_layer])
        if batch_id % 25 == 0:
            if batch_id % 500 == 0:
                print('\nBatch %d / %d, loss %f' % (batch_id, batch_count, cost))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

    # save model
    param_path = './saved_models/pass_%d' % (pass_id,)
    fluid.io.save_params(executor=exe, dirname=param_path, main_program=None)
    print('Model saved: ' + param_path)

    # run test program
    test_label_list = []
    test_prediction_list = []
    test_cost_total = 0
    for iteration_id in xrange(data_info['test']):
        test_cost, test_prediction, test_label = exe.run(program=test_program, fetch_list=[test_cost_layer, test_prediction_layer, test_label_layer])
        test_prediction_list.append(test_prediction[0][0])
        test_label_list.append(test_label[0][0])
        test_cost_total += test_cost
    print('Test loss: %f' % (test_cost_total / data_info['test'],))
    acc, roc_auc = more_info(test_prediction_list, test_label_list)
    # Stop early
    if acc > 0.7 and roc_auc > 0.7:
        print('Good enough! Stop early.')
        break

print('Done!')


from paddle import fluid as fluid


def conv_bn(input, ch_out, filter_size, stride, padding, act='relu',
            bias_attr=False):
    conv_layer = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr,
    )
    return fluid.layers.batch_norm(input=conv_layer, act=act)


def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn(input, ch_out, 1, stride, 0, None)   // change channel by conv 1x1
    else:
        return input


def basic_block(input, ch_in, ch_out, stride):
    conv1 = conv_bn(input, ch_out, 3, stride, 1)
    conv2 = conv_bn(conv1, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=conv2, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    layer = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        layer = block_func(layer, ch_out, ch_out, 1)
    return layer


def resnet(input, depth=32):
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6

    conv1 = conv_bn(input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basic_block, conv1, 16, 16, n, 1)
    res2 = layer_warp(basic_block, res1, 16, 32, n, 2)
    res3 = layer_warp(basic_block, res2, 32, 64, n, 2)
    #drop = fluid.layers.dropout(x=res3, dropout_prob=0.5)
    pool = fluid.layers.pool2d(
        input=res3,
        pool_size=8,
        pool_type='avg',
        pool_stride=1,
    )
    predict = fluid.layers.fc(input=pool, size=1, act='sigmoid')
    return predict

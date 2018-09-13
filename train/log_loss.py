from paddle.fluid.layer_helper import LayerHelper

def log_loss(input, label, epsilon=1e-4):
    """
    **Negative Log Loss Layer**
    This layer accepts input predictions and target label and returns the
    negative log loss.
    .. math::
        Out = -label * log(X + epsilon) - (1 - label) * log(1 - X + epsilon)
    Args:
        input:  a 2-D tensor with shape [N x 1], where N is the batch size.
                This input is a probability computed by the previous operator.
        label:  the ground truth which is a 2-D tensor with shape [N x 1],
                where N is the batch size.
        epsilon: epsilon
    Returns:
         A 2-D tensor with shape [N x 1], the negative log loss.
    Examples:
        .. code-block:: python
          prob = fluid.layers.sigmoid(net)
          cost = fluid.layers.log_loss(input=prob, label=label)
    """
    helper = LayerHelper('log_loss', **locals())
    loss = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='log_loss',
        inputs={'Predicted': [input],
                'Labels': [label]},
        outputs={'Loss': [loss]},
        attrs={'epsilon': epsilon})
    return loss

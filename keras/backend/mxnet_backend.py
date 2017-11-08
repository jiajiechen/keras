import mxnet as mx
from mxnet import gluon
import numpy as np
from .common import _FLOATX, floatx, _EPSILON, image_data_format


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        sparse: Boolean, whether the placeholder should have a sparse type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2, 4, 5))
        >>> input_ph.shape
        (2, 4, 5)
        >>> input_ph
        <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
    ```
    """
    if dtype is None:
        dtype = floatx()
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    else:
        shape = tuple([0 if x is None else x for x in shape])
    x = mx.sym.Variable(shape=shape, name=name, dtype=dtype)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


def variable(value, dtype=None, name=None, constraint=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
        constraint: Optional projection function to be
            applied to the variable after an optimizer update.

    # Returns
        A variable instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> kvar.eval()
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    if dtype is None:
        dtype = floatx()
    if hasattr(value, 'tocoo'):
        sparse_coo = value.tocoo()
        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                  np.expand_dims(sparse_coo.col, 1)), 1)
        v = tf.SparseTensor(indices=indices,
                            values=sparse_coo.data,
                            dense_shape=sparse_coo.shape)
        v._keras_shape = sparse_coo.shape
        v._uses_learning_phase = False
        return v
    v = tf.Variable(value, dtype=tf.as_dtype(dtype), name=name)
    if isinstance(value, np.ndarray):
        v._keras_shape = value.shape
    elif hasattr(value, 'get_shape'):
        v._keras_shape = int_shape(value)
    v._uses_learning_phase = False
    # TODO: move to Variable constructor when supported in public release.
    try:
        v.constraint = constraint
    except AttributeError:
        v._constraint = constraint
    return v
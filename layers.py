import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def conv2d(input, filters, kernel_size, stride, padding, name, act=tf.nn.relu):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding, activation=act,
                                  kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05),
                                  bias_initializer=tf.constant_initializer(value=0.1),
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                  bias_regularizer=tf.keras.regularizers.l2(l=0.1),
                                  activity_regularizer=tf.keras.regularizers.l2(l=0.1), name=name)(input)

def max_pool(input, kernel_size, stride, padding, name):
    return tf.keras.layers.MaxPooling2D(kernel_size, stride, padding=padding, name=name)(input)

def deconv2d(input, size, name):
    return tf.keras.layers.UpSampling2D(size=(size, size), name=name)(input)

def build_UNet(input, is_training, n_class, drop_rate=0, f=32, krl=3, pad="same"):
    """Create U-Net with
	:param input: input images
	:param is_training: False for inference, True if training
	:param n_class: number of classes considered
	:param drop_rate: dropout rate between 0 and 1
	:param f: initial number of filters
	:param krl: kernel size (symmetric) of convolution layers
	:param pad: same or valid
	"""
    conv11 = conv2d(input=input, filters=f, kernel_size=krl, stride=1, padding=pad, name='conv11')
    conv12 = conv2d(conv11, f, krl, 1, pad, 'conv12')

    drop1 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(conv12, training=is_training)
    pool1 = max_pool(drop1, (2, 2), (2, 2), pad, 'pool1')

    conv21 = conv2d(pool1, f * 2, krl, 1, pad, 'conv21')
    conv22 = conv2d(conv21, f * 2, krl, 1, pad, 'conv22')

    drop2 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(conv22, training=is_training)
    pool2 = max_pool(drop2, (2, 2), (2, 2), pad, 'pool2')

    conv31 = conv2d(pool2, f * 4, krl, 1, pad, 'conv31')
    conv32 = conv2d(conv31, f * 4, krl, 1, pad, 'conv32')

    drop3 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(conv32, training=is_training)
    pool3 = max_pool(drop3, (2, 2), (2, 2), pad, 'pool3')

    conv41 = conv2d(pool3, f * 8, krl, 1, pad, 'conv41')
    conv42 = conv2d(conv41, f * 8, krl, 1, pad, 'conv42')

    drop4 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(conv42, training=is_training)
    pool4 = max_pool(drop4, (2, 2), (2, 2), pad, 'pool4')
    #
    conv51 = conv2d(pool4, f * 16, 2, 1, pad, 'conv51')
    conv52 = conv2d(conv51, f * 16, 2, 1, pad, 'conv52')

    drop5 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(conv52, training=is_training)
    deconv1 = deconv2d(input=drop5, size=2, name='deconv1')

    merge11 = tf.concat(values=[conv42, deconv1], axis=-1, name='merge11')
    drop6 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(merge11, training=is_training)

    conv61 = conv2d(drop6, f * 8, krl, 1, pad, 'conv61')
    conv62 = conv2d(conv61, f * 8, krl, 1, pad, 'conv62')

    deconv2 = deconv2d(conv62, 2, 'deconv2')

    merge12 = tf.concat(values=[conv32, deconv2], axis=-1, name='merge12')
    drop7 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(merge12, training=is_training)

    conv71 = conv2d(drop7, f * 4, krl, 1, pad, 'conv71')
    conv72 = conv2d(conv71, f * 4, krl, 1, pad, 'conv72')

    deconv3 = deconv2d(conv72, 2, 'deconv3')

    merge13 = tf.concat(values=[conv22, deconv3], axis=-1, name='merge13')
    drop8 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(merge13, training=is_training)

    conv81 = conv2d(drop8, f * 2, krl, 1, pad, 'conv81')
    conv82 = conv2d(conv81, f * 2, krl, 1, pad, 'conv82')

    deconv4 = deconv2d(conv82, 2, 'deconv4')

    merge14 = tf.concat(values=[conv12, deconv4], axis=-1, name='merge14')
    drop9 = tf.keras.layers.Dropout(drop_rate, noise_shape=None, seed=None)(merge14, training=is_training)

    conv91 = conv2d(drop9, f, krl, 1, pad, 'conv91')
    conv92 = conv2d(conv91, f, krl, 1, pad, 'conv92')

    layers_out = []
    for clss in range(n_class):
        name = 'conv94_0_{}'.format(clss)
        conv_ = conv2d(input=conv92, filters=2, kernel_size=1, stride=1, padding=pad, name=name, act=None)
        layers_out.append(conv_)
    return layers_out

from models import *


model_zoo = ['DCGAN', 'LSGAN', 'WGAN', 'WGAN-GP', 'EBGAN', 'BEGAN', 'DRAGAN', 'CoulombGAN']

def get_model(mtype, name, training,image_shape=[64,64,3],batch_norm=True):
    model = None
    if mtype == 'DCGAN' or mtype == 'DCGAN_SIM':
        model = dcgan.DCGAN
    elif mtype == 'LSGAN':
        model = lsgan.LSGAN
    elif mtype == 'WGAN':
        model = wgan.WGAN
    elif mtype == 'WGAN-GP':
        model = wgan_gp.WGAN_GP
    elif mtype == 'EBGAN':
        model = ebgan.EBGAN
    elif mtype == 'BEGAN':
        model = began.BEGAN
    elif mtype == 'DRAGAN' or mtype == 'DRAGAN_BN':
        model = dragan.DRAGAN
        return model(name=name, training=training,image_shape=image_shape,batch_norm=batch_norm)
    elif mtype == 'COULOMBGAN':
        model = coulombgan.CoulombGAN
    elif mtype == 'DCGAN-GP':
        model = dcgan_gp.DCGAN_GP
    elif mtype == 'WGAN-DRAGAN':
        model = wgan_dragan.WGAN_DRAGAN
    elif mtype == 'DCGAN-CONS' or mtype == 'DCGAN-CONS_BN':
        model = dcgan_consensus.DCGAN_CONS
        return model(name=name,training=training,image_shape=image_shape,batch_norm=batch_norm)

    elif mtype == 'DCGAN-LOCAL':
        model = dcgan_local.DCGAN_LOCAL
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(name=name, training=training,image_shape=image_shape)


def get_dataset(dataset_name):
    celebA_64 = './data/celebA_tfrecords/*.tfrecord'
    lsun_bedroom_128 = './data/lsun/bedroom_128_tfrecords/*.tfrecord'
    lsun_bedroom_64 = './data/lsun/bedroom_64_tfrecords/*.tfrecord'
    mnist = './data/mnist_tfrecords/*.tfrecord'

    if dataset_name == 'celeba':
        path = celebA_64
        n_examples = 201599 #1000 images held out for testing
        n_channels = 3
    elif dataset_name == 'lsun':
        path = lsun_bedroom_64
        n_examples = 3033042
        n_channels = 3
    elif dataset_name == 'mnist':
        path = mnist
        n_examples = 69000
        n_channels = 1
    else:
        raise ValueError('{} is does not supported. dataset must be celeba or lsun.'.format(dataset_name))

    return path, n_examples,n_channels


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")


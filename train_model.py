import jax_resnet
import jax
from flax import linen as nn
import jax.numpy as jnp
import optax

def get_loss(*, logits, labels, num_classes, l2_reg=False, params=None):
    labels_one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
    if l2_reg:
        return optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot).mean() + l2_loss(params=params, alpha=0.00004)

@jax.jit
def l2_loss(params, alpha):
    loss = 0.0
    for i in jax.tree_leaves(params):
        loss += alpha * jax.lax.square(i).mean()
    return loss



# -----RESNET50-----

def get_resnet(no_params=False, num_classes=10):
    model = jax_resnet.ResNet50(n_classes=num_classes)  # this is good implementation no need to make from start
    if no_params:
        return model
    else:
        key = jax.random.PRNGKey(0)
        params = model.init(key, jnp.ones((1,32,32,3)))
        return params, model



# -----INCEPTION V4-----

class conv2d_bnorm(nn.Module):
    nb_filter: int
    num_row: int
    num_col: int
    padding = "SAME" 
    strides = (1,1)
    use_bias = False

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.nb_filter, (self.num_row, self.num_col), 
        strides=self.strides, 
        padding=self.padding, 
        use_bias=self.use_bias, 
        kernel_init=jax.nn.initializers.variance_scaling(scale=2.0,mode="fan_in",distribution="normal")
        )(x)
        x = nn.BatchNorm(axis=-1, momentum=0.9997, use_scale=False)(x)
        return nn.relu(x)

class block_inception_a(nn.Module):

    @nn.compact
    def __call__(self, x):
        branch_0 = conv2d_bnorm(nb_filter=96, num_row=1, num_col=1)(x)

        branch_1 = conv2d_bnorm(nb_filter=64, num_row=1, num_col=1)(x)
        branch_1 = conv2d_bnorm(nb_filter=96, num_row=3, num_col=3)(branch_1)

        branch_2 = conv2d_bnorm(nb_filter=64, num_row=1, num_col=1)(x)
        branch_2 = conv2d_bnorm(nb_filter=96, num_row=3, num_col=3)(branch_2)
        branch_2 = conv2d_bnorm(nb_filter=96, num_row=3, num_col=3)(branch_2)

        branch_3 = nn.avg_pool(x, window_shape=(3,3), strides=(1,1), padding="SAME")
        branch_3 = conv2d_bnorm(nb_filter=96, num_row=1, num_col=1)(branch_3)

        x = jax.lax.concatenate([branch_0,branch_1,branch_2,branch_3])
        return x

class block_reduction_a(nn.Module):
    @nn.compact
    def __call__(self, x):
        branch_0 = conv2d_bnorm(nb_filter=384, num_row=3, num_col=3, strides=(2,2), padding="VALID")(x)

        branch_1 = conv2d_bnorm(nb_filter=192, num_row=1, num_col=1)(x)
        branch_1 = conv2d_bnorm(nb_filter=244, num_row=3, num_col=3)(branch_1)
        branch_1 = conv2d_bnorm(nb_filter=256, num_row=3, num_col=3, strides=(2,2), padding="VALID")(branch_1)

        branch_2 = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="VALID")

        x = jax.lax.concatenate([branch_0,branch_1,branch_2])
        return x

class block_inception_b(nn.Module):
    @nn.compact
    def __call__(self, x):
        branch_0 = conv2d_bnorm(nb_filter=384, num_row=1, num_col=1)(x)

       
        branch_1 = conv2d_bnorm(nb_filter=192, num_row=1, num_col=1)(branch_1)
        branch_1 = conv2d_bnorm(nb_filter=244, num_row=1, num_col=7)(branch_1)
        branch_1 = conv2d_bnorm(nb_filter=256, num_row=7, num_col=1)(branch_1)

        branch_2 = conv2d_bnorm(nb_filter=192, num_row=1, num_col=1)(x)
        branch_2 = conv2d_bnorm(nb_filter=192, num_row=7, num_col=1)(branch_2)
        branch_2 = conv2d_bnorm(nb_filter=244, num_row=1, num_col=7)(branch_2)
        branch_2 = conv2d_bnorm(nb_filter=244, num_row=7, num_col=1)(branch_2)
        branch_2 = conv2d_bnorm(nb_filter=256, num_row=1, num_col=7)(branch_2)

        branch_3 =  nn.avg_pool(x, window_shape=(3,3), strides=(1,1), padding="SAME")
        branch_3 = conv2d_bnorm(nb_filter=128, num_row=1, num_col=1)(branch_3)

        x = jax.lax.concatenate([branch_0,branch_1,branch_2,branch_3])
        return x

class block_reduction_b(nn.Module):
    @nn.compact
    def __call__(self,x):

        branch_0 = conv2d_bnorm(nb_filter=192, num_row=1, num_col=1)(x)
        branch_0 = conv2d_bnorm(nb_filter=192, num_row=3, num_col=3, strides=(2,2), padding="VALID")(branch_0)

        branch_1 = conv2d_bnorm(nb_filter=256, num_row=1, num_col=1)(x)
        branch_1 = conv2d_bnorm(nb_filter=256, num_row=1, num_col=7)(branch_1)
        branch_1 = conv2d_bnorm(nb_filter=320,num_row=7,num_col=1)(branch_1)
        branch_1 = conv2d_bnorm(nb_filter=320, num_row=3, num_col=3, strides=(2,2), padding="VALID")(branch_1)

        branch_2 = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="VALID")

        x = jax.lax.concatenate([branch_0,branch_1,branch_2])
        return x

class block_inception_c(nn.Module):
    @nn.compact
    def __call__(self,x):

        branch_0 = conv2d_bnorm(nb_filter=256, num_row=1, num_col=1)(x)
        
        branch_1 = conv2d_bnorm(nb_filter=384, num_row=1, num_col=1)(x)
        branch_1_0 = conv2d_bnorm(nb_filter=256, num_row=1, num_col=3)(branch_1)
        branch_1_1 = conv2d_bnorm(nb_filter=256, num_row=3, num_col=1)(branch_1)
        branch_1 = jax.lax.concatenate([branch_1_0,branch_1_1])

        branch_2 = conv2d_bnorm(nb_filter=384, num_row=1, num_col=1)(x)
        branch_2 = conv2d_bnorm(nb_filter=448, num_row=3, num_col=1)(branch_2)
        branch_2 = conv2d_bnorm(nb_filter=512, num_row=1, num_col=3)(branch_2)
        branch_2_0 = conv2d_bnorm(nb_filter=256, num_row=1, num_col=3)(branch_2)
        branch_2_1 = conv2d_bnorm(nb_filter=256, num_row=3, num_col=1)(branch_2)
        branch_2 = jax.lax.concatenate([branch_2_0,branch_2_1])

        branch_3 = nn.avg_pool(x, window_shape=(3,3), strides=(1,1), padding="SAME")
        branch_3 = conv2d_bnorm(nb_filter=256, num_row=1, num_col=1)(branch_3)

        x = jax.lax.concatenate([branch_0,branch_1,branch_2,branch_3])
        return x


class inception_v4_base(nn.Module):
    @nn.compact
    def __call__(self,x):
        x = conv2d_bnorm(nb_filter=32, num_row=3, num_col=3, strides=(2,2), padding="VALID")(x)
        x = conv2d_bnorm(nb_filter=32, num_row=3, num_col=3, padding="VALID")(x)
        x = conv2d_bnorm(nb_filter=64, num_row=3, num_col=3)(x)

        branch_0 = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="VALID")
        branch_1 = conv2d_bnorm(nb_filter=96, num_row=3, num_col=3, strides=(2,2), padding="VALID")(x)
        x = jax.lax.concatenate([branch_0,branch_1])

        branch_0 = conv2d_bnorm(nb_filter=64, num_row=1, num_col=1)(x)
        branch_0 = conv2d_bnorm(nb_filter=96, num_row=3, num_col=3, padding="VALID")(branch_0)

        branch_1 = conv2d_bnorm(nb_filter=64, num_row=1, num_col=1)(x)
        branch_1 = conv2d_bnorm(nb_filter=64, num_row=1, num_col=7)(branch_1)
        branch_1 = conv2d_bnorm(nb_filter=64, num_row=7, num_col=1)(branch_1)
        branch_1 = conv2d_bnorm(nb_filter=96, num_row=3, num_col=3, padding="VALID")(branch_1)

        x = jax.lax.concatenate([branch_0,branch_1])

        branch_0 = conv2d_bnorm(nb_filter=192, num_row=3, num_col=3, strides=(2,2), padding="VALID")
        branch_1 = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="VALID")

        x = jax.lax.concatenate([branch_0,branch_1])

        for _ in range(4):
            x = block_inception_a()(x)

        x = block_reduction_a()(x)

        for _ in range(7):
            x = block_inception_b()(x)

        x = block_reduction_b(x)

        for _ in range(3):
            x = block_inception_c()(x)

        return x

class inception_v4(nn.Module):
    num_classes: int
    dropout_keep: float

    @nn.compact
    def __call__(self, x):
        x = inception_v4_base()(x)
        x = nn.avg_pool(x, window_shape=(8,8), padding="VALID")
        x = nn.Dropout(rate = self.dropout_keep)(x)
        x = x.resahpe((x.shape[0], -1))  # should flatten
        x = nn.Dense(features=self.num_classes)(x)
        x = nn.softmax(x)
        return x

def get_inception(no_params=False, num_classes=10):
    model = inception_v4(num_classes=num_classes, dropout_keep=0.2)  # this is good implementation no need to make from start
    if no_params:
        return model
    else:
        key = jax.random.PRNGKey(0)
        params = model.init(key, jnp.ones((1,32,32,3))) # will cause bug as has to be shape of image
        return params, model


# -----VGG16-----

class VGG16(nn.Module):
    num_classes:int
    @nn.compact
    def __call__(self,x):
        x = nn.Conv(64, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, pool_size=(2,2), strides=(2,2))
        x = nn.Conv(128, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(128, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, pool_size=(2,2), strides=(2,2))
        x = nn.Conv(256, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(256, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(256, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, pool_size=(2,2), strides=(2,2))
        x = nn.Conv(512, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(512, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(512, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, pool_size=(2,2), strides=(2,2))
        x = nn.Conv(512, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(512, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(512, kernal_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, pool_size=(2,2), strides=(2,2))
        x = x.resahpe((x.shape[0], -1))
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        x = nn.softmax(x)
        return x




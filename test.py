import tensorflow as tf
from absl import app
from pysc2.lib import actions,features
import utils as U

def main(unused_argv):
    #config = tf.ConfigProto(allow_soft_placement=True)
    #print(type(config))
    #print(config.gpu_options)
    print(type(actions.FUNCTIONS))
    print(len(actions.FUNCTIONS))
    print(len(features.MINIMAP_FEATURES))
    print('minimap channel is', U.minimap_channel())
    print('screen channel is', U.screen_channel())
    print('player id index in minimap features is', features.MINIMAP_FEATURES.player_id.index)


if __name__ == "__main__":
    app.run(main);
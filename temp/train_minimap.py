import os
import sys
from absl import flags
import datetime
from baselines import deepq
import tensorflow as tf

from pysc2.env import sc2_env
from pysc2.env import available_actions_printer

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards", "Name of the map to play.")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "a2c", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.0005, "Learning rate")
flags.DEFINE_integer("num_agents", 4, "number of RL agents for A2C")
flags.DEFINE_integer("num_scripts", 4, "number of script agents for A2C")
flags.DEFINE_integer("nsteps", 20, "number of batch steps for A2C")
flags.DEFINE_integer("step_mul", 8, "number of steps per agent step")

# get current date and time
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print(start_time)

# get current file directory
full_path = os.path.realpath(__file__)
PROJECT_DIR, FILENAME = os.path.split(full_path)
#print(PROJECT_DIR)
LOGGING_DIR = PROJECT_DIR + "/Logging"
#print(LOGGING_DIR)
max_mean_reward = 0
last_filename = ""

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def deepq_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10
        and locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(os.path.join(PROJECT_DIR, 'models/deepq/'))):
        try:
          os.mkdir(os.path.join(PROJECT_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJECT_DIR, 'models/deepq/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

    #   max_mean_reward = locals['mean_100ep_reward']
    #   act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
    #   act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

    #   filename = os.path.join(
    #     PROJECT_DIR,
    #     'models/deepq/mineral_x_%s.pkl' % locals['mean_100ep_reward'])
    #   act_x.save(filename)
    #   filename = os.path.join(
    #     PROJECT_DIR,
    #     'models/deepq/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
    #   act_y.save(filename)
    #   print("save best mean_100ep_reward model to %s" % filename)
    #   last_filename = filename


def main():
    print(tf.ConfigProto)
    # FLAGS(sys.argv)
    # #env = gym.make("CartPole-v0")
    # env = sc2_env.SC2Env(
    #     map_name=FLAGS.map,
    #     step_mul=FLAGS.step_mul,
    #     visualize=True,
    #     screen_size_px=(16, 16),
    #     minimap_size_px=(16, 16))
    # #env = available_actions_printer.AvailableActionsPrinter(env)
    # #print(env.observation_spec)
    # #print(env.action_spec)
    # #model = deepq.models.mlp([64])
    # model = deepq.models.cnn_to_mlp(convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)
    # act = deepq.learn(
    #     env,
    #     q_func=model,
    #     num_actions=16,
    #     lr=1e-3,
    #     max_timesteps=100000,
    #     buffer_size=50000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     print_freq=10,
    #     callback=callback
    # )
    # print("Saving model to collectmineralshards.pkl")
    # act.save("collectmineralshards.pkl")


if __name__ == '__main__':
    main()

# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AlphaZero Bot implemented in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np
import math

import tensorflow.compat.v1 as tf

import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import masked_softmax
from open_spiel.python.algorithms import dqn

MCTSResult = collections.namedtuple("MCTSResult",
                                    "state_feature target_value target_policy")

LossValues = collections.namedtuple("LossValues", "total policy value l2")


class AlphaZero(object):
  """AlphaZero implementation following the pseudocode AlphaZero implementation
  given in the paper with DOI 10.1126/science.aar6404."""

  def __init__(self,
               game,
               bot,
               replay_buffer_capacity=int(1e6),
               action_selection_transition=30,
               num_self_play_games=5000,
               batch_size=4096,
               random_state=None):
    """
    Args:
      game: a pyspiel.Game object
      bot: an MCTSBot object.
      replay_buffer_capacity: the size of the replay buffer in which the results 
        of self-play games are stored.
      action_selection_transition: an integer representing the move number in a 
        game of self-play when greedy action selection is used. Before this,
        actions are sampled from the MCTS policy.
      num_self_play_games: the number of self-play games to play before each
        training round.
      batch_size: the number of examples used for a single training update. Note
        that this batch size must be small enough for the neural net training 
        update to fit into device memory.
      random_state: An optional numpy RandomState to make it deterministic.

    Raises:
      ValueError: if incorrect inputs are supplied.
    """

    game_info = game.get_type()
    if game.num_players() != 2:
      raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
      raise ValueError("The game must be a Deterministic one, not {}".format(
          game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
      raise ValueError(
          "The game must be a perfect information one, not {}".format(
              game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("The game must be turn-based, not {}".format(
          game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
      raise ValueError("The game must be 0-sum, not {}".format(game.utility))
    if game.num_players() != 2:
      raise ValueError("Game must have exactly 2 players.")

    self.bot = bot
    self.game = game
    self.replay_buffer = dqn.ReplayBuffer(replay_buffer_capacity)
    self.num_self_play_games = num_self_play_games
    self._action_selection_transition = action_selection_transition
    self.batch_size = batch_size

  def update(self):
    data = self.replay_buffer.sample(self.batch_size, replace=True)
    (total_loss, policy_loss, value_loss,
     l2_loss) = self.bot.evaluator.update(data)
    return LossValues(total=total_loss,
                      policy=policy_loss,
                      value=value_loss,
                      l2=l2_loss)

  def self_play(self):
    # optim = a0.bot.evaluator.optimizer
    # tf.variables_initializer(optim.variables())
    return [self._self_play_single() for _ in range(self.num_self_play_games)]

  def _self_play_single(self):
    state = self.game.new_initial_state()
    policy_targets, state_features = [], []

    while not state.is_terminal():
      root_node = self.bot.mcts_search(state)
      state_features.append(self.bot.evaluator.feature_extractor(state))
      target_policy = np.zeros(self.game.num_distinct_actions(),
                               dtype=np.float32)
      for child in root_node.children:
        target_policy[child.action] = child.explore_count
      target_policy /= sum(target_policy)
      policy_targets.append(target_policy)

      action = self._select_action(root_node.children, len(state.history()))
      state.apply_action(action)

    terminal_rewards = state.rewards()
    for i, (feature, pol) in enumerate(zip(state_features, policy_targets)):
      value = terminal_rewards[i % 2]
      self.replay_buffer.add(
          MCTSResult(state_feature=feature,
                     target_policy=pol,
                     target_value=value))

    return terminal_rewards

  def _select_action(self, children, game_history_len):
    explore_counts = [(child.explore_count, child.action) for child in children]
    if game_history_len < self._action_selection_transition:
      probs = np_softmax(np.array([i[0] for i in explore_counts]))
      action_index = np.random.choice(range(len(probs)), p=probs)
      action = explore_counts[action_index][1]
    else:
      _, action = max(explore_counts)
    return action


def alpha_zero_ucb_score(child, parent_explore_count, params):
  c_init, c_base = params
  if child.outcome is not None:
    return child.outcome[child.player]

  c = math.log((parent_explore_count + c_base + 1) / c_base) + c_init
  c *= math.sqrt(parent_explore_count) / (child.explore_count + 1)

  prior_score = c * child.prior
  value_score = child.explore_count and child.total_reward / child.explore_count

  return prior_score + value_score


def np_softmax(logits):
  max_logit = np.amax(logits, axis=-1, keepdims=True)
  exp_logit = np.exp(logits - max_logit)
  return exp_logit / np.sum(exp_logit, axis=-1, keepdims=True)


class AlphaZeroKerasEvaluator(mcts.TrainableEvaluator):
  """Implements a 
  """

  def __init__(
      self,
      keras_model,
      l2_regularization=1e-4,
      #TODO: generalize optimizer
      optimizer=tf.train.MomentumOptimizer(2e-1, momentum=0.9),
      device='cpu',
      feature_extractor=None,
      cache_size=None):
    """
    Args:
      keras_model: a Keras Model object.
      l2_regularization: the amount of l2 regularization to use during training.
      optimizer: the number of self-play games to play before each
        training round.
      device: The device used to run the keras_model during evaluation and 
        training. Possible values are 'cpu', 'gpu', or a tf.device(...) object.
      feature_extractor: a function which takes as argument the game state and 
        returns a numpy tensor which the keras_model can accept as input. If 
        None, then the default features will be used, which is the 
        observation_tensor() state method, reshaped to match the keras_model 
        input shape (if possible). The keras_model is always evaluated on the
        output of this function.
        cache_size: Whether to cache the result of the net evaluation. Calling 
        the update method automatically resets the cache. Set to 0 to turn it 
        off, and None for an unbounded cache size.
    Raises:
      ValueError: if incorrect inputs are supplied.
    """

    super().__init__(cache_size=cache_size)

    self.model = keras_model

    # TODO: validate user-supplied keras_model
    self.input_shape = list(self.model.input_shape)
    self.input_shape[0] = 1  # Keras sets the batch dim to None
    _, (_, self.num_actions) = self.model.output_shape

    self.l2_regularization = l2_regularization
    self.optimizer = optimizer

    if device == 'gpu':
      if not tf.test.is_gpu_available():
        raise ValueError("GPU support is unavailable.")
      self.device = tf.device("gpu:0")
    elif device == 'cpu':
      self.device = tf.device("cpu:0")
    else:
      self.device = device

    if feature_extractor == None:
      self.feature_extractor = _create_default_feature_extractor(
          self.input_shape)
    else:
      self.feature_extractor = feature_extractor

  def value_and_prior(self, state):
    state_feature = self.feature_extractor(state)
    with self.device:
      value, policy = self.model(state_feature)

    # renormalize policy over legal actions
    policy = np.array(policy)[0]
    mask = np.array(state.legal_actions_mask())
    policy = masked_softmax.np_masked_softmax(policy, mask)
    policy = [(action, policy[action]) for action in state.legal_actions()]

    # value is required to be array over players
    value = value[0, 0].numpy()
    value = np.array([value, -value])

    return (value, policy)

  def update(self, training_examples):
    state_features = np.vstack([f for (f, _, _) in training_examples])
    value_targets = np.vstack([v for (_, v, _) in training_examples])
    policy_targets = np.vstack([p for (_, _, p) in training_examples])

    with self.device:
      with tf.GradientTape() as tape:
        values, policy_logits = self.model(state_features, training=True)
        loss_value = tf.losses.mean_squared_error(
            values, tf.stop_gradient(value_targets))
        loss_policy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=policy_logits, labels=tf.stop_gradient(policy_targets))
        loss_policy = tf.reduce_mean(loss_policy)
        loss_l2 = 0
        for weights in self.model.trainable_variables:
          loss_l2 += self.l2_regularization * tf.nn.l2_loss(weights)
        loss = loss_policy + loss_value + loss_l2

      grads = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(
          zip(grads, self.model.trainable_variables),
          global_step=tf.train.get_or_create_global_step())

    return LossValues(total=float(loss),
                      policy=float(loss_policy),
                      value=float(loss_value),
                      l2=float(loss_l2))


def _create_default_feature_extractor(shape):

  def feature_extractor(state):
    obs = state.observation_tensor()
    return np.array(obs, dtype=np.float32).reshape(shape)

  return feature_extractor


def keras_resnet(input_shape,
                 num_actions,
                 num_residual_blocks=19,
                 num_filters=256,
                 value_head_hidden_size=256,
                 activation='relu'):
  """
  This ResNet implementation copies as closely as possible the
  description found in the Methods section of the AlphaGo Zero Nature paper.
  It is mentioned in the AlphaZero Science paper supplementary material that
  "AlphaZero uses the same network architecture as AlphaGo Zero". Note that
  this implementation only supports flat policy distributions.

  Arguments:
    input_shape: A tuple of 3 integers specifying the non-batch dimensions of 
      input tensor shape.
    num_actions: The determines the output size of the policy head.
    num_residual_blocks: The number of residual blocks. Can be 0.
    num_filters: the number of convolution filters to use in the residual blocks.
    value_head_hidden_size: the number of hidden units in the value head dense layer.
    activation: the activation function to use in the net. Does not affect the 
      final tanh activation in the value head.
  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  inputs = tf.keras.Input(shape=input_shape, name='input')
  body = _resnet_body(inputs,
                      num_filters=num_filters,
                      num_residual_blocks=num_residual_blocks,
                      kernel_size=3,
                      activation=activation)
  value_head = _resnet_value_head(body,
                                  hidden_size=value_head_hidden_size,
                                  activation=activation)
  policy_head = _resnet_mlp_policy_head(body,
                                        num_actions,
                                        activation=activation)
  return tf.keras.Model(inputs=inputs, outputs=[value_head, policy_head])


def _residual_layer(inputs, num_filters, kernel_size, activation):
  x = inputs
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  return tf.keras.layers.BatchNormalization(axis=-1)(x)


def _residual_tower(inputs, num_res_blocks, num_filters, kernel_size,
                    activation):
  x = inputs
  for _ in range(num_res_blocks):
    y = _residual_layer(x, num_filters, kernel_size, activation)
    y = _residual_layer(x, num_filters, kernel_size, activation)
    x = tf.keras.layers.add([x, y])
    x = tf.keras.layers.Activation(activation)(x)

  return x


def _resnet_body(inputs, num_residual_blocks, num_filters, kernel_size,
                 activation):
  x = inputs
  x = tf.keras.layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             padding='same',
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = _residual_tower(x, num_residual_blocks, num_filters, kernel_size,
                      activation)
  return x


def _resnet_value_head(inputs, hidden_size, activation):
  x = inputs
  x = tf.keras.layers.Conv2D(filters=1,
                             kernel_size=1,
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(hidden_size,
                            activation=activation,
                            kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.Dense(1,
                            activation='tanh',
                            kernel_initializer='he_uniform',
                            name='value')(x)
  return x


def _resnet_mlp_policy_head(inputs, num_classes, activation):
  x = inputs
  x = tf.keras.layers.Conv2D(filters=2,
                             kernel_size=1,
                             strides=1,
                             kernel_initializer='he_uniform')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(num_classes,
                            kernel_initializer='he_uniform',
                            name='policy')(x)
  return x


def keras_mlp(input_size,
              num_actions,
              num_layers=2,
              num_hidden=128,
              activation='relu'):
  """
  A simple MLP implementation with both a value and policy head.

  Arguments:
    input_size: An integer specifying the size of the input vector.
    num_actions: The determines the output size of the policy head.
    num_layers: The number of dense layers before the policy and value heads.
    num_hidden: the number of hidden units in the dense layers.
    activation: the activation function to use in the net. Does not affect the 
      final tanh activation in the value head.

  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  inputs = tf.keras.Input(shape=(input_size,), name='input')
  x = inputs
  for _ in range(num_layers):
    x = tf.keras.layers.Dense(num_hidden,
                              kernel_initializer='he_uniform',
                              activation=activation)(x)
  policy = tf.keras.layers.Dense(num_actions,
                                 kernel_initializer='he_uniform',
                                 name='policy')(x)
  value = tf.keras.layers.Dense(1,
                                kernel_initializer='he_uniform',
                                activation='tanh',
                                name='value')(x)
  return tf.keras.Model(inputs=inputs, outputs=[value, policy])

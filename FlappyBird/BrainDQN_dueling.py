import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import random
from collections import deque

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 500000000.  # timesteps to observe before training. too large : for test
EXPLORE = 1000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100


class BrainDQN:

    def __init__(self, actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.w21, self.b21,self.w22,self.b22 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.w21T, self.b21T,self.w22T,self.b22T = self.createQNetworkT()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.w21T.assign(self.w21), self.b21T.assign(self.b21),
                                            self.w22T.assign(self.w22), self.b22T.assign(self.b22)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("------成功载入---------------------:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])
        # --------------dueling方法需要重写的第二层全连接层权重
        # W_fc2 = self.weight_variable([512, self.actions])
        # b_fc2 = self.bias_variable([1, self.actions])

        # input layer

        stateInput = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers

        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        # 更改的地方-------------------------------------------------------------
        w21 = self.weight_variable([512, 1])
        b21 = self.bias_variable([1])
        V = tf.matmul(h_fc1, w21) + b21
        w22 = self.weight_variable([512, self.actions])
        b22 = self.bias_variable([1, self.actions])
        A = tf.matmul(h_fc1, w22) + b22
        # Q Value layer
        QValue = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        # 更改的地方-------------------------------------------------------------
        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, w21,b21,w22,b22

    def createQNetworkT(self):
        # network weights
        W_conv1T = self.weight_variable([8, 8, 4, 32])
        b_conv1T = self.bias_variable([32])

        W_conv2T = self.weight_variable([4, 4, 32, 64])
        b_conv2T = self.bias_variable([64])

        W_conv3T = self.weight_variable([3, 3, 64, 64])
        b_conv3T = self.bias_variable([64])

        W_fc1T = self.weight_variable([1600, 512])
        b_fc1T = self.bias_variable([512])
        # --------------dueling方法需要重写的第二层全连接层权重
        # W_fc2T = self.weight_variable([512, self.actions])
        # b_fc2T = self.bias_variable([1, self.actions])

        # input layer

        stateInputT = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers

        h_conv1T = tf.nn.relu(self.conv2d(stateInputT, W_conv1T, 4) + b_conv1T)
        h_pool1T = self.max_pool_2x2(h_conv1T)

        h_conv2T = tf.nn.relu(self.conv2d(h_pool1T, W_conv2T, 2) + b_conv2T)

        h_conv3T = tf.nn.relu(self.conv2d(h_conv2T, W_conv3T, 1) + b_conv3T)

        h_conv3_flatT = tf.reshape(h_conv3T, [-1, 1600])
        h_fc1T = tf.nn.relu(tf.matmul(h_conv3_flatT, W_fc1T) + b_fc1T)

        w21T = self.weight_variable([512, 1])
        b21T = self.bias_variable([1])
        V1T = tf.matmul(h_fc1T, w21T) + b21T

        w22T = self.weight_variable([512, self.actions])
        b22T = self.bias_variable([1, self.actions])
        A1T = tf.matmul(h_fc1T, w22T) + b22T
        # Q Value layer
        QValueT = V1T + (A1T - tf.reduce_mean(A1T, axis=1, keepdims=True))

        return stateInputT, QValueT, W_conv1T, b_conv1T, W_conv2T, b_conv2T, W_conv3T, b_conv3T, W_fc1T, b_fc1T, w21T, b21T, w22T, b22T

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

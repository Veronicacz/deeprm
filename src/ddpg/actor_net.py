import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.0001
BATCH_SIZE = 1000
TAU = 0.001


class ActorNet:
    """ Actor Network Model of DDPG Algorithm """

    def __init__(self, num_states, num_actions):
        self.g = tf.Graph()  # A TensorFlow computation, represented as a dataflow graph.
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            # tf.Session() and tf.InteractiveSession differences:
            # https://www.cnblogs.com/cvtoEyes/p/9035047.html

            # actor network model parameters:
            self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a,\
                self.actor_state_in, self.actor_model = self.create_actor_net(num_states, num_actions)  # self.W4_a, self.B4_a,\

            # target actor network model parameters:
            self.t_W1_a, self.t_B1_a, self.t_W2_a, self.t_B2_a, self.t_W3_a, self.t_B3_a,\
                self.t_actor_state_in, self.t_actor_model = self.create_actor_net(num_states, num_actions)  # self.t_W4_a, self.t_B4_a,


            # self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a,self.W4_a, self.B4_a,\
            #     self.actor_state_in, self.actor_model = self.create_actor_net(num_states, num_actions)  # self.W4_a, self.B4_a,\

            # # target actor network model parameters:
            # self.t_W1_a, self.t_B1_a, self.t_W2_a, self.t_B2_a, self.t_W3_a, self.t_B3_a,self.t_W4_a, self.t_B4_a,\
            #     self.t_actor_state_in, self.t_actor_model = self.create_actor_net(num_states, num_actions)  # self.t_W4_a, self.t_B4_a,

 
            # cost of actor network:
            # gets input from action_gradient computed in critic network file
            self.q_gradient_input = tf.placeholder("float", [None, num_actions])
            self.actor_parameters = [self.W1_a, self.B1_a,
                                     self.W2_a, self.B2_a, self.W3_a, self.B3_a] #, self.W4_a, self.B4_a]


            # self.actor_parameters = [self.W1_a, self.B1_a,
            #                          self.W2_a, self.B2_a, self.W3_a, self.B3_a, self.W4_a, self.B4_a] #, self.W4_a, self.B4_a]

            self.parameters_gradients = tf.gradients(
                self.actor_model, self.actor_parameters, -self.q_gradient_input)  # /BATCH_SIZE)
            # tf.gradients: Constructs symbolic derivatives of sum of ys w.r.t. x in xs.
            # https://www.tensorflow.org/api_docs/python/tf/gradients

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
                zip(self.parameters_gradients, self.actor_parameters))
            # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer


            # initialize all tensor variable parameters:
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

            file = "/home/veronica/Desktop/model/actor_net_model_zz0.ckpt"
            self.saver.restore(self.sess, file)

            self.update_target_actor_op = [
                self.t_W1_a.assign(TAU * self.W1_a + (1 - TAU) * self.t_W1_a),
                self.t_B1_a.assign(TAU * self.B1_a + (1 - TAU) * self.t_B1_a),
                self.t_W2_a.assign(TAU * self.W2_a + (1 - TAU) * self.t_W2_a),
                self.t_B2_a.assign(TAU * self.B2_a + (1 - TAU) * self.t_B2_a),
                self.t_W3_a.assign(TAU * self.W3_a + (1 - TAU) * self.t_W3_a),
                self.t_B3_a.assign(TAU * self.B3_a + (1 - TAU) * self.t_B3_a)]

            # self.update_target_actor_op = [
            #     self.t_W1_a.assign(TAU * self.W1_a + (1 - TAU) * self.t_W1_a),
            #     self.t_B1_a.assign(TAU * self.B1_a + (1 - TAU) * self.t_B1_a),
            #     self.t_W2_a.assign(TAU * self.W2_a + (1 - TAU) * self.t_W2_a),
            #     self.t_B2_a.assign(TAU * self.B2_a + (1 - TAU) * self.t_B2_a),
            #     self.t_W3_a.assign(TAU * self.W3_a + (1 - TAU) * self.t_W3_a),
            #     self.t_B3_a.assign(TAU * self.B3_a + (1 - TAU) * self.t_B3_a),
            #     self.t_W4_a.assign(TAU * self.W4_a + (1 - TAU) * self.t_W4_a),
            #     self.t_B4_a.assign(TAU * self.B4_a + (1 - TAU) * self.t_B4_a)]
            # To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters


            self.sess.run([
                self.t_W1_a.assign(self.W1_a),
                self.t_B1_a.assign(self.B1_a),
                self.t_W2_a.assign(self.W2_a),
                self.t_B2_a.assign(self.B2_a),
                self.t_W3_a.assign(self.W3_a),
                self.t_B3_a.assign(self.B3_a)])

            # self.sess.run([
            #     self.t_W1_a.assign(self.W1_a),
            #     self.t_B1_a.assign(self.B1_a),
            #     self.t_W2_a.assign(self.W2_a),
            #     self.t_B2_a.assign(self.B2_a),
            #     self.t_W3_a.assign(self.W3_a),
            #     self.t_B3_a.assign(self.B3_a),
            #     self.t_W4_a.assign(self.W4_a),
            #     self.t_B4_a.assign(self.B4_a)])

    def create_actor_net(self, num_states=2520, num_actions=6):
        """ Network that takes states and return action """
        N_HIDDEN_1 = 400 # 1500 # 400
        N_HIDDEN_2 =  300 # 1000 # 300
        N_HIDDEN_3 = 600
        num_acts = 6 #1
        actor_state_in = tf.placeholder("float", [None, num_states])
        # tf.placeholder(
    #    dtype,
#       shape=None,
#       name=None
#       )
###################07/15################
        # actor_state_in = tf.reshape(tem_actor_state_in, shape=[-1, 126, 126, 1])

      # return : A Tensor that may be used as a handle for feeding a value, but not evaluated directly.
        W1_a = tf.Variable(tf.random_uniform(
           [num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
       # tf.variable(tf.random_uniform()): https://www.aiworkbox.com/lessons/generate-a-random-tensor-in-tensorflow
        B1_a = tf.Variable(tf.random_uniform(
           [N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        W2_a = tf.Variable(tf.random_uniform(
           [N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        B2_a = tf.Variable(tf.random_uniform(
           [N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        W3_a = tf.Variable(tf.random_uniform([N_HIDDEN_2, num_acts], -0.01, 0.01)) # -0.003, 0.003))
        B3_a = tf.Variable(tf.random_uniform([num_acts], -0.01, 0.01)) # -0.003, 0.003))



        # tf.math.softplus(    features,    name=None): Computes softplus: log(exp(features) + 1).
        # tf.matmul: matrix multiplication
        # https://www.aiworkbox.com/lessons/multiply-two-matricies-using-tensorflow-matmul
        # note: differences between it and tf.multiply: tf.multiply(X, Y) does element-wise multiplication; not matrix multiplication
        
        H1_a = tf.nn.softplus(tf.matmul(actor_state_in, W1_a) + B1_a)
        # tf.nn.tanh: activation function, Computes hyperbolic tangent of x element-wise.
        H2_a = tf.nn.tanh(tf.matmul(H1_a, W2_a) + B2_a)
        H3_a = tf.nn.softmax(tf.matmul(H2_a, W3_a) + B3_a)
        actor_model = H3_a # tf.argmax(H3_a, 1)  # tf.matmul(H2_a, W3_a) + B3_a
        return W1_a, B1_a, W2_a, B2_a, W3_a, B3_a, actor_state_in, actor_model

        # W1_a = tf.Variable(tf.random_uniform(
        #     [num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        # # tf.variable(tf.random_uniform()): https://www.aiworkbox.com/lessons/generate-a-random-tensor-in-tensorflow
        # B1_a = tf.Variable(tf.random_uniform(
        #     [N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        # W2_a = tf.Variable(tf.random_uniform(
        #     [N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        # B2_a = tf.Variable(tf.random_uniform(
        #     [N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        # W3_a = tf.Variable(tf.random_uniform(
        #     [N_HIDDEN_2, N_HIDDEN_3], -1 / math.sqrt(N_HIDDEN_2), 1 / math.sqrt(N_HIDDEN_2)))
        # B3_a = tf.Variable(tf.random_uniform(
        #     [N_HIDDEN_3], -1 / math.sqrt(N_HIDDEN_2), 1 / math.sqrt(N_HIDDEN_2)))
        # W4_a = tf.Variable(tf.random_uniform([N_HIDDEN_3, num_actions], -0.01, 0.01)) # -0.003, 0.003))
        # B4_a = tf.Variable(tf.random_uniform([num_actions], -0.05, 0.05)) # -0.01, 0.01)) # -0.003, 0.003))



        # H0_a = tf.nn.relu(tf.matmul(actor_state_in, W1_a) + B1_a)
        # H1_a = tf.nn.relu(tf.matmul(H0_a, W2_a) + B2_a) # (tf.matmul(actor_state_in, W1_a) + B1_a)
        # # tf.nn.tanh: activation function, Computes hyperbolic tangent of x element-wise.
        # H2_a = tf.nn.relu(tf.matmul(H1_a, W3_a) + B3_a) # (tf.matmul(H1_a, W2_a) + B2_a)
        # actor_model = tf.matmul(H2_a, W4_a) + B4_a #tf.nn.softmax(tf.matmul(H2_a, W4_a) + B4_a) #  tf.matmul(H2_a, W3_a) + B3_a
        # return W1_a, B1_a, W2_a, B2_a, W3_a, B3_a, W4_a, B4_a,actor_state_in, actor_model

    def evaluate_actor(self, state_t):
        return self.sess.run(self.actor_model, feed_dict={self.actor_state_in: state_t})

    def evaluate_target_actor(self, state_t_1):
        return self.sess.run(self.t_actor_model, feed_dict={self.t_actor_state_in: state_t_1})

    def train_actor(self, actor_state_in, q_gradient_input):
        self.sess.run(self.optimizer, feed_dict={
                      self.actor_state_in: actor_state_in, self.q_gradient_input: q_gradient_input})

    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)

    def save_model(self, ep):
        file = "/home/veronica/Desktop/model/actor_net_model_zz" + str(ep) + ".ckpt"
        save_path = self.saver.save(self.sess, file)

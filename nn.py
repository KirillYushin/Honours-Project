import tensorflow.compat.v1 as tf

class NeuralNetwork():
	def __init__(self, inputSize, outputSize, scopeName):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.hidden1Size = 512 #1024
		self.hidden2Size = 256 #512
		self.streamSize = self.hidden2Size // 2
		self.learningRate = 0.01
		
		with tf.variable_scope(scopeName):
			# Input state and target Q values
			self.inputs = tf.placeholder("float", [None, self.inputSize])
			self.targets = tf.placeholder("float", [None, self.outputSize])
			# Weights
			self.w_h1 = self.init_weights([self.inputSize, self.hidden1Size])
			self.w_h2 = self.init_weights([self.hidden1Size, self.hidden2Size])
			self.w_val = self.init_weights([self.streamSize, 1])
			self.w_adv = self.init_weights([self.streamSize, self.outputSize])
			#self.w_o = self.init_weights([self.hidden2Size, self.outputSize])
			# Biases
			self.b_h1 = self.init_weights([self.hidden1Size])
			self.b_h2 = self.init_weights([self.hidden2Size])
			self.b_val = self.init_weights([1])
			self.b_adv = self.init_weights([self.outputSize])
			#self.b_o = self.init_weights([self.outputSize])
		
		self.all_trainable_variables = tf.trainable_variables()
		self.trainable_variables = [var for var in self.all_trainable_variables if scopeName in var.name]

		with tf.device("/gpu:0"):
			self.predictions = self.model(self.inputs)
			self.loss = tf.reduce_mean(tf.square(tf.subtract(self.targets, self.predictions)))
			self.train_op = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
			self.predict_op = tf.argmax(self.predictions, 1)
		
	def init_weights(self, shape):
		return tf.Variable(tf.random_normal(shape, stddev=0.5))
	
	def model(self, X):
		with tf.device("/gpu:0"):
			h1 = tf.nn.relu(tf.matmul(X, self.w_h1) + self.b_h1)
			h2 = tf.nn.relu(tf.matmul(h1, self.w_h2) + self.b_h2)
			# Split into separate advantage and value streams (Dueling DQN)
			streamVal, streamAdv = tf.split(h2, num_or_size_splits=2, axis=1)
			value = tf.matmul(streamVal, self.w_val) + self.b_val
			advantage = tf.matmul(streamAdv, self.w_adv) + self.b_adv
			# Combine streams together to get the final Q-values
			Qvalues = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
			return Qvalues
			#return tf.matmul(h2, W_o) + B_o
			
	
	# Copy weights from main Q network (Double DQN)
	def copy_weights(self, session, mainQN):
		with tf.device("/gpu:0"):
			for targetWeight, mainWeight in zip(self.trainable_variables, mainQN.trainable_variables):
				session.run(targetWeight.assign(mainWeight))

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")
print(f"TensorFlow Hub Version: {hub.__version__}")

# # Create some Tensors
# string = tf.Variable("this is a string", tf.string)
# number = tf.Variable(324, tf.int16)
# floating = tf.Variable(3.567, tf.float64)
#
# print(f"String Tensor: {string}")
# print(f"Number Tensor: {number}")
# print(f"Floating Point Tensor: {floating}")
#
# # Create Ranked Tensors
# rank1_tensor = tf.Variable(["test"], tf.string)
# rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
# rank3_tensor = tf.Variable([[["a", "b"], ["c", "d"], ["e", "f"]], [["g", "h"], ["i", "j"], ["k", "l"]]], tf.string)
#
# # Determine Tensor Rank
# print(f"Rank 1 Tensor: {tf.rank(rank1_tensor)}")
# print(f"Rank 2 Tensor: {tf.rank(rank2_tensor)}")
#
# # Determine Tensor Shape:
# print(f"Rank 1 Tensor Shape: {rank1_tensor.shape}")
# print(f"Rank 2 Tensor Shape: {rank2_tensor.shape}")
# print(f"Rank 3 Tensor Shape: {rank3_tensor.shape}")
#
# # Change Tensor Shape
# tensor1 = tf.ones([3, 1, 2])
# tensor2 = tf.reshape(tensor1, [2, 3, 1])
# tensor3 = tf.reshape(tensor2, [3, -1])
#
# print(tensor1)
# print(tensor2)
# print(tensor3)
# # Get Tensor Value
# tensor_value = tensor3.numpy()
# print(tensor_value)

t = tf.zeros([5,5,5,5])
t = tf.reshape(t, [625])
print(t)

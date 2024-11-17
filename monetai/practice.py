import tensorflow as tf
import time
import sys
import os



import tensorflow as tf
# Create large random tensors
matrix_size = 5000
a = tf.random.normal([matrix_size, matrix_size])
b = tf.random.normal([matrix_size, matrix_size])

# Measure GPU performance
start_time = time.time()
with tf.device('/GPU:0'):
    c = tf.matmul(a, b)
    tf.print("Result computed on GPU.")
end_time = time.time()

print(f"Time taken for matrix multiplication of size {matrix_size}x{matrix_size}: {end_time - start_time:.2f} seconds")

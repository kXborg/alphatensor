from audioop import mul
from ctypes import util
import unittest

from absl.testing import absltest
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np

from benchmarking import factorizations
from benchmarking import utils

config.update('jax_enable_x64', True)
factors = factorizations.get_4x4x4_alphatensor_gpu()
matrix_multiplication_algorithm = utils.algorithm_from_factors(factors)

rng1, rng2 = jax.random.split(jax.random.PRNGKey(42))
# print('RNG1 : ', rng1, 'RNG2 : ', rng2)

full_a = jax.random.uniform(rng1, (16, 16), dtype=jnp.float64)
full_b = jax.random.uniform(rng2, (16, 16), dtype=jnp.float64)

np_a = np.asarray(full_a)
np_b = np.asarray(full_b)

# print(full_a)
# print('___'*50)
# print(np_a)

a = utils.block_split(full_a, 4, 4)
b = utils.block_split(full_b, 4, 4)

block_np_a = utils.np_block_split(np_a, 4, 4)
block_np_b = utils.np_block_split(np_b, 4, 4)

print(len(a), len(a[0]), len(a[0][0]))
print('___'*50)
print(len(block_np_a), len(block_np_a[0]), len(block_np_a[0][0]))

# print('______________________')
# print(b)
# print(len(a[0][0]))

alpha_multi = matrix_multiplication_algorithm(a, b)
np_multi = np_a @ np_b
jax_multi = full_a @ full_b

print('Success')
# print(alpha_multi)

# Verify if jax multiplication is same as np_multi.

ver_jax_np = np.asarray(jax_multi)
ver_alpha_multi = np.asarray(alpha_multi)

print(np_multi.shape, ver_jax_np.shape, ver_alpha_multi.shape)

print(ver_alpha_multi)
print('---'*50)
print(ver_jax_np)
print('---'*50)
print(np_multi)

'''
Now check how to convert 4x4x4x4 matrix to 16x16.

Then compare the outputs.

Check time taken
'''

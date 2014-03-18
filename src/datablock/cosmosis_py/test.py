#some tests, very incomplete

from . import block
import unittest
from . import errors

class TestBlockFunctions(unittest.TestCase):
	def test_int(self):
		# make sure the shuffled sequence does not lose any elements
		b = block.Block()
		section = 'test'
		b.put_int(section,'a',1)
		assert b.get_int(section,'a')==1
		self.assertRaises(errors.BlockNameNotFound, b.get_int, 'test', 'b')
		b.replace_int(section,'a',2)
		assert b.get_int(section,'a')==2
		self.assertRaises(errors.BlockNameAlreadyExists, b.put_int, 'test', 'a', 3)

	def test_string(self):
		b = block.Block()
		section = 'test'
		b.put_string(section, 's', 'my_string')
		assert b.get_string(section,'s')=='my_string'
		self.assertRaises(errors.BlockNameNotFound, b.get_string, 'test', 't')
		self.assertRaises(errors.BlockNameAlreadyExists, b.put_string, 'test', 's', 'my_string')


	def test_int_array(self):
		b = block.Block()
		section='test'
		b.put_int_array_1d(section, 'x', [1,2,3])
		r = b.get_int_array_1d(section, 'x')
		assert (r==[1,2,3]).all()

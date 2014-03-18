#some tests, very incomplete

from . import block
import unittest

class TestBlockFunctions(unittest.TestCase):
	def test_int(self):
		# make sure the shuffled sequence does not lose any elements
		b = block.Block()
		section = 'test'
		b.put_int(section,'a',1)
		assert b.get_int(section,'a')==1
		self.assertRaises(block.BlockError, b.get_int, 'test', 'b')
		b.replace_int(section,'a',2)
		assert b.get_int(section,'a')==2
		self.assertRaises(block.BlockError, b.put_int, 'test', 'a', 3)

	def test_int_array(self):
		b = block.Block()
		section='test'
		b.put_int_array_1d(section, 'x', [1,2,3])
		r = b.get_int_array_1d(section, 'x')
		assert (r==[1,2,3]).all()

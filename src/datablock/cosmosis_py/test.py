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

	def test_double_array(self):
		b = block.Block()
		section='test'
		b.put_double_array_1d(section, 'x', [1.4,2.1,3.6])
		r = b.get_double_array_1d(section, 'x')
		assert (r==[1.4,2.1,3.6]).all()	

	def test_has(self):
		b = block.Block()
		assert not b.has_section("cats")		
		b.put("cats", "n", 14)
		assert b.has_section("cats")
		assert not b.has_section("dogs")
		assert b.has_value("cats", "n")
		assert not b.has_value("cats", "m")
		assert not b.has_value("dogs", "n")

	def test_special(self):
		b = block.Block()
		section = 'test'
		b.put(section, 'a', 4)
		assert b[section, 'a']==4
		b[section,'b'] = 5
		assert b[section,"b"]==5
		b[section,'b'] = 6
		assert b[section,"b"]==6
		assert (section,'b') in b

	def test_generic(self):
		b = block.Block()
		section = 'test'
		b.put(section, 'a', 4)
		assert b.get_int(section, 'a')==4
		b.put(section, 'b', 2.0)
		assert b.get_double(section, 'b')==2.0
		b.put(section, 'c', 'hello')
		assert b.get_string(section, 'c')=='hello'
		self.assertRaises(errors.BlockNameAlreadyExists, b.put, 'test', 'c', 'my_string')
		b.put(section, 'd', [1,2,3,4])
		assert all(b.get_int_array_1d(section, 'd') == [1,2,3,4])
		assert b.get(section, 'a')==4
		assert b.get(section, 'b')==2.0
		assert b.get(section, 'c')=="hello"

#some tests, very incomplete

from . import block
import unittest
from . import errors

class TestBlockFunctions(unittest.TestCase):

	def test_delete(self):
		b = block.DataBlock()
		section = 'test'
		b.put_bool(section, 'b', True)
		b._delete_section(section)
		assert len(list(b.keys()))==0
		self.assertRaises(errors.BlockSectionNotFound, b._delete_section, section)

	def test_bool(self):
		b = block.DataBlock()
		section = 'test'
		b.put_bool(section, 'b', True)
		assert b.get_bool(section, 'b')==True
		assert b.get_bool(section, 'c', default=False)==False
		assert b.get_bool(section, 'c', default=True)==True
		self.assertRaises(errors.BlockSectionNotFound, b.get_bool, 'ddd', 'c')

	def test_default(self):
		b = block.DataBlock()
		section = 'test'
		assert b.get_int(section, 'a', default=14)==14
		self.assertRaises(errors.BlockSectionNotFound, b.get_int, section, 'a')
		assert b.get_double(section, 'a', default=1.0)==1.0
		assert b.get_string(section, 'a', default="QQQ")=="QQQ"

	def test_complex(self):
		pass

	def test_int(self):
		# make sure the shuffled sequence does not lose any elements
		b = block.DataBlock()
		section = 'test'
		b.put_int(section,'a',1)
		assert b.get_int(section,'a')==1
		self.assertRaises(errors.BlockNameNotFound, b.get_int, 'test', 'b')
		b.replace_int(section,'a',2)
		assert b.get_int(section,'a')==2
		self.assertRaises(errors.BlockNameAlreadyExists, b.put_int, 'test', 'a', 3)

	def test_string(self):
		b = block.DataBlock()
		section = 'test'
		b.put_string(section, 's', 'my_string')
		assert b.get_string(section,'s')=='my_string'
		self.assertRaises(errors.BlockNameNotFound, b.get_string, 'test', 't')
		self.assertRaises(errors.BlockNameAlreadyExists, b.put_string, 'test', 's', 'my_string')


	def test_int_array(self):
		b = block.DataBlock()
		section='test'
		b.put_int_array_1d(section, 'x', [1,2,3])
		r = b.get_int_array_1d(section, 'x')
		assert (r==[1,2,3]).all()

	def test_double_array(self):
		b = block.DataBlock()
		section='test'
		b.put_double_array_1d(section, 'x', [1.4,2.1,3.6])
		r = b.get_double_array_1d(section, 'x')
		assert (r==[1.4,2.1,3.6]).all()	

	def test_keys(self):
		b = block.DataBlock()
		section='dogs'
		b.put(section, 'x', [1.4,2.1,3.6])
		b.put(section, "n", 14)
		b.put(section, 's', 'my_string')
		section='other'
		b.put(section, 'a', 98)
		b.put(section, "b", 1.4)
		b.put_string(section, 's', 'my_string')
		keys = list(b.keys())
		assert sorted(keys) == sorted([('dogs','x'), ('dogs','n'),('dogs','s'),('other','a'),('other','b'),('other','s')])
		for k in keys:
			assert k in b


	def test_has(self):
		b = block.DataBlock()
		assert not b.has_section("cats")		
		b.put("cats", "n", 14)
		assert b.has_section("cats")
		assert not b.has_section("dogs")
		assert b.has_value("cats", "n")
		assert not b.has_value("cats", "m")
		assert not b.has_value("dogs", "n")

	def test_special(self):
		b = block.DataBlock()
		section = 'test'
		b.put(section, 'a', 4)
		assert b[section, 'a']==4
		b[section,'b'] = 5
		assert b[section,"b"]==5
		b[section,'b'] = 6
		assert b[section,"b"]==6
		assert (section,'b') in b
		self.assertRaises(errors.BlockNameNotFound, b.get, section, 'fff',)


	def test_generic(self):
		b = block.DataBlock()
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


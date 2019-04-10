#!/usr/bin/env ruby

require 'erb'

ENUMERATORS = {
  "bool" => "DBT_BOOL",
  "int" => "DBT_INT",
  "double" => "DBT_DOUBLE",
  "complex" => "DBT_COMPLEX",
  "string" => "DBT_STRING"
}

class TestContext
  def initialize(typ, def_val, values, uninitialized_val)
    scalar_types = [ ["bool", 'false'],
                     ["int", 1],
                     ["double", 2.5],
                     ["string", '"wombat"'],
                     ["complex", "3.5-2.5*_Complex_I"] ]
    @current_type = typ
    @current_type_simplified = typ
    @current_type_simplified = "complex" if typ == "double complex"
    @default_val = def_val
    @values = values
    @uninitialized_val = uninitialized_val
    @get_fun = "c_datablock_get_#{@current_type_simplified}"
    @get_def_fun = "c_datablock_get_#{@current_type_simplified}_default"
    @put_fun = "c_datablock_put_#{@current_type_simplified}"
    @replace_fun = "c_datablock_replace_#{@current_type_simplified}"
    @other_types = scalar_types.select { |t,v| t != @current_type_simplified }
    @assert_sym = typ == "string" ? :assert_equal_string : :assert_equal_simple
    @cleanup_sym = typ == "string" ? :cleanup_string : :cleanup_simple
    @type_enumerator = ENUMERATORS[@current_type_simplified]
  end

  def cleanup_simple(*)
    "// no cleanup needed "
  end

  def cleanup_string(varname)
    "free(#{varname})"
  end

  def assert_equal_simple(varname, val)
  "assert(#{varname} == #{val})"
  end

  def assert_equal_string(varname, val)
    "assert(strlen(#{varname}) == strlen(#{val}));\n  assert(strcmp(#{varname}, #{val}) == 0)"
  end

  def get_binding
    binding
  end
end

template = ERB.new(File.read("test_c_datablock_scalars.template"), nil, "<>")

bool_context    = TestContext.new("bool", false, [true, false, false, true], false)
int_context     = TestContext.new("int", 0, [5, -4, 10, -11], 0)
double_context  = TestContext.new("double", 0.0, [5.5, -4.5, 10.25, -11.5], 0.0)
complex_context = TestContext.new("double complex", 0.0,
                                  ["1.25-2.5*_Complex_I",
                                   "-1.0e-6+10.5*_Complex_I",
                                   "1.75e10+3.0*_Complex_I",
                                   "5.5*_Complex_I"],
                                  0.0)
string_context  = TestContext.new("string", 'NULL', ['"cow"', '"a dog"', '"\tmoose\nbat"', '"pointy thing"'],
                                  '""')

contexts = [bool_context, int_context, double_context, complex_context, string_context]

File.open("test_c_datablock_scalars.h", "w") do |f|
  contexts.each do |c|
    f.write(template.result(c.get_binding))
  end
end



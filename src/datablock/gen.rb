#!/usr/bin/env ruby

require 'erb'


class TestContext
  def initialize(typ, def_val, values)
    scalar_types = [ ["int", 1],
                     ["double", 2.5],
                     ["string", '"wombat"'],
                     ["complex", "3.5-2.5*_Complex_I"] ]
    @current_type = typ
    @default_val = def_val
    @values = values
    @get_fun = "c_datablock_get_#{typ}"
    @get_def_fun = "c_datablock_get_#{typ}_default"
    @put_fun = "c_datablock_put_#{typ}"
    @replace_fun = "c_datablock_replace_#{typ}"
    @other_types = scalar_types.select { |t,v| t != typ }
    @assert_sym = typ == "string" ? :assert_equal_string : :assert_equal_simple
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

int_context     = TestContext.new("int", 0, [5, -4, 10, -11])
double_context  = TestContext.new("double", 0.0, [5.5, -4.5, 10.25, -11.5])
complex_context = TestContext.new("complex", 0.0,
                                  ["1.25-2.5*_Complex_I",
                                   "-1.0e-6+10.5*_Complex_I",
                                   "1.75e10+3.0*_Complex_I",
                                   "5.5*_Complex_I"])
string_context  = TestContext.new("string", '""', ['"cow"', '"a dog"', '"\tmoose\nbat"', '"pointy thing"'])

contexts = [int_context, double_context, complex_context, string_context]

File.open("test_c_datablock_scalars.h", "w") do |f|
  contexts.each do |c|
    f.write(template.result(c.get_binding))
  end
  # f.write(template.result(int_context.get_binding))
  # f.write(template.result(double_context.get_binding))

end



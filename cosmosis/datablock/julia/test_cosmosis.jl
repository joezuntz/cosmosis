push!(LOAD_PATH,".")
import cosmosis
block=cosmosis.make_datablock()
cosmosis.put_int(block, "my_section", "my_int", 17)
x = cosmosis.get_int(block, "my_section", "my_int")

cosmosis.put_double(block, "my_section", "my_double", 12.99)
y = cosmosis.get_double(block, "my_section", "my_double")

cosmosis.put_bool(block, "my_section", "my_bool", 1)
z = cosmosis.get_bool(block, "my_section", "my_bool")

cosmosis.put_string(block, "my_section", "my_str", "hat")
w = cosmosis.get_string(block, "my_section", "my_str")


a = fill(19.4, 2, 3)
cosmosis.put_double_array(block, "my_section", "myarr", a)
b= cosmosis.get_double_array(block, "my_section", "myarr")

c = fill(Int32(19), 2, 3)
cosmosis.put_int_array(block, "my_section", "myarr2", c)
d = cosmosis.get_int_array(block, "my_section", "myarr2")

println("x = ", x)
println("y = ", y)
println("z = ", z)
println("w = ", w)
println("b = ", b)
println("d = ", d)


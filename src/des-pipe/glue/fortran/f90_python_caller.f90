module f90_py_caller
	USE ISO_C_BINDING
	USE ISO_C_UTILITIES
	USE dynamic_loading
	implicit none
	
	interface
		!int call_python_interface(char *module_name, char *function_name, size_t n);
		function wrap_call_python_interface(module_name, function_name, handle)  &
		        bind(C, name='call_python_interface')
			use iso_c_binding
			implicit none
			integer (c_int) :: wrap_call_python_interface
			integer (c_size_t), value :: handle
			character(c_char) :: module_name, function_name		
		end function wrap_call_python_interface

	end interface
	
	contains
	
	function call_python_interface(module_name, function_name, handle) result(status)
	implicit none
		integer status
		integer (c_size_t) :: handle
		character(*) :: module_name, function_name
		status = wrap_call_python_interface(trim(module_name)//C_NULL_CHAR, & 
			trim(function_name)//C_NULL_CHAR, handle)
	end function call_python_interface
	

end module f90_py_caller
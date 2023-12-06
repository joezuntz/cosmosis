module cosmosis
import ForwardDiff


# double, array, grid

datablock_status = Int32
datablock = Ptr{Cvoid}

option_section = "module_options"
lib_name = ENV["COSMOSIS_SRC_DIR"] * "/datablock/libcosmosis.so"


struct BlockError <: Exception
    code::datablock_status
    msg::String
    
    function BlockError(code::datablock_status)
        new(code, error_names[code])
    end
    function BlockError(code::datablock_status, msg)
        new(code, error_names[code + 1] * ": " * msg)
    end
end
Base.showerror(io::IO, e::BlockError) = print(io, e.msg)

error_names = [
    "CosmoSIS: Success",
    "CosmoSIS: Datablock is Null",
    "CosmoSIS: Section name is null",
    "CosmoSIS: Section not found",
    "CosmoSIS: Name is Null",
    "CosmoSIS: Name not found",
    "CosmoSIS: Name already exists",
    "CosmoSIS: Value is null",
    "CosmoSIS: Wrong value type",
    "CosmoSIS: Memory allocation error",
    "CosmoSIS: Size is null",
    "CosmoSIS: Size not positive",
    "CosmoSIS: Size is insufficient",
    "CosmoSIS: Dimensionality not positive",
    "CosmoSIS: Number of dimensions too large",
    "CosmoSIS: Dimensions do notmarch",
    "CosmoSIS: Extents are null",
    "CosmoSIS: Extents do not match dimension",
    "CosmoSIS: Internal logic error: please open an issue"
]

cosmosis_dtype_int = 0
cosmosis_dtype_double = 1
cosmosis_dtype_complex = 2
cosmosis_dtype_string = 3
cosmosis_dtype_int1d = 4
cosmosis_dtype_double1d = 5
cosmosis_dtype_complex1d = 6
cosmosis_dtype_string1d = 7
cosmosis_dtype_bool = 8
cosmosis_dtype_intnd = 9
cosmosis_dtype_doublend = 10
cosmosis_dtype_complexnd = 11
cosmosis_dtype_unknown = 12

enum_size_ptr = cglobal((:cosmosis_enum_size, lib_name), Cint)
enum_size = unsafe_load(enum_size_ptr)
Cenum = Dict(
    1=>Int8,
    2=>Int16,
    4=>Int32,
    8=>Int64,
)[enum_size]



function make_datablock()
    res = ccall((:make_c_datablock, lib_name), datablock, ())
    res
end

function get_type(block::datablock, section, name)
    value = Ref{Cenum}()
    res = ccall((:c_datablock_get_type, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Cenum}), block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * "name was " * string(name) * "value was " * string(value)))        
    end
    value[]
end

function put!(block::datablock, section, name, value::Int32)
    res = ccall((:c_datablock_put_int, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Int32), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * "name was " * string(name) * "value was " * string(value)))
    end
    res
end

function put!(block::datablock, section, name, value_::Int64)
    value = Int32(value_)
    res = ccall((:c_datablock_put_int, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Int32), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * "name was " * string(name) * "value was " * string(value)))
    end
    res
end


function get_int(block, section, name)
    value = Ref{Int32}()
    res = ccall((:c_datablock_get_int, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Int32}), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    value[]
end



function put!(block::datablock, section, name, value::Float64)
    res = ccall((:c_datablock_put_double, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Cdouble), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    res
end


function get_double(block, section, name)
    value = Ref{Cdouble}()
    res = ccall((:c_datablock_get_double, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Cdouble}), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name) * ", value was " * string(value) ))
    end
    value[]
end


function put!(block::datablock, section, name, value::Bool)
    if value
        uint_value = UInt8(1)
    else
        uint_value = UInt8(0)
    end
    res = ccall((:c_datablock_put_bool, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, UInt8), 
            block, section, name, uint_value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name) * ", value was " * string(value) ))
    end
    res
end


function get_bool(block, section, name)
    value = Ref{UInt8}()
    res = ccall((:c_datablock_get_bool, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{UInt8}), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    value[]
end


function get_string(block, section, name)
    value = Ref{Cstring}()
    res = ccall((:c_datablock_get_string, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Cstring}), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    unsafe_string(value[])
end


function put!(block, section, name, value::String)
    res = ccall((:c_datablock_put_string, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Cstring), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    res
end


function put!(block, section, name, value::Array{Float64})

    ndim = ndims(value)
    extents = convert(Array{Int32}, collect(size(value)))

    res = ccall((:c_datablock_put_double_array, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ptr{Cdouble}, Int32, Ptr{Int32}), 
            block, section, name, vec(value), ndim, extents)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    res
end


function get_double_array_nd(block, section, name)

    ndim_ptr = Ref{Int32}()
    res = ccall((:c_datablock_get_array_ndim, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Int32}),
            block, section, name, ndim_ptr)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    ndim = ndim_ptr[]    

    extents = Array{Int32}(undef, ndim)

    res = ccall((:c_datablock_get_double_array_shape, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Int32, Ptr{Int32}), 
            block, section, name, ndim, extents)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end


    t = tuple(extents...)

    value = Array{Float64}(undef, t)

    res = ccall((:c_datablock_get_double_array, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ptr{Cdouble}, Int32, Ptr{Int32}), 
            block, section, name, value, ndim, extents)


    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    value

end



function put!(block, section, name, value::Array{Int32})

    ndim = ndims(value)
    extents = convert(Array{Int32}, collect(size(value)))

    res = ccall((:c_datablock_put_int_array, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ptr{Int32}, Int32, Ptr{Int32}), 
            block, section, name, vec(value), ndim, extents)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    res
end



function get_int_array_nd(block, section, name)

    ndim_ptr = Ref{Int32}()
    res = ccall((:c_datablock_get_array_ndim, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Int32}),
            block, section, name, ndim_ptr)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    ndim = ndim_ptr[]    

    extents = Array{Int32}(undef, ndim)

    res = ccall((:c_datablock_get_int_array_shape, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Int32, Ptr{Int32}), 
            block, section, name, ndim, extents)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end


    t = tuple(extents...)

    value = Array{Int32}(undef, t)

    res = ccall((:c_datablock_get_int_array, lib_name), 
            datablock_status,
            (datablock, Cstring, Cstring, Ptr{Int32}, Int32, Ptr{Int32}), 
            block, section, name, value, ndim, extents)


    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    value

end


function put!(block, section, name, value::Vector{Float64})

    sz = Int32(length(value))

    res = ccall((:c_datablock_put_double_array_1d, lib_name), 
        datablock_status,
        (datablock, Cstring, Cstring, Ptr{Float64}, Int32), 
        block, section, name, value, sz)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end


end


function get_double_array_1d(block, section, name)


    sz = ccall((:c_datablock_get_array_length, lib_name), 
        Int32,
        (datablock, Cstring, Cstring), 
        block, section, name)

    if (sz<0)
        throw(BlockError(Int32(8), "section was " * string(section) * ", name was " * string(name)))
    end

    sz_ptr = Ref{Int32}()
    value = Array{Float64}(undef, sz)

    res = ccall((:c_datablock_get_double_array_1d_preallocated, lib_name), 
        datablock_status,
        (datablock, Cstring, Cstring, Ptr{Float64}, Ref{Int32}, Int32), 
        block, section, name, value, sz_ptr, sz)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    value
end

function get_int_array_1d(block, section, name)


    sz = ccall((:c_datablock_get_array_length, lib_name), 
        Int32,
        (datablock, Cstring, Cstring), 
        block, section, name)

    if (sz<0)
        throw(BlockError(Int32(8), "section was " * string(section) * ", name was " * string(name)))
    end

    sz_ptr = Ref{Int32}()
    value = Array{Int32}(undef, sz)

    res = ccall((:c_datablock_get_int_array_1d_preallocated, lib_name), 
        datablock_status,
        (datablock, Cstring, Cstring, Ptr{Int32}, Ref{Int32}, Int32), 
        block, section, name, value, sz_ptr, sz)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    value
end

function get(block, section, name)
    dtype = get_type(block, section, name)

    if dtype == cosmosis_dtype_int
        return get_int(block, section, name)
    elseif dtype == cosmosis_dtype_double
        return get_double(block, section, name)
    elseif dtype == cosmosis_dtype_bool
        return get_bool(block, section, name)
    elseif dtype == cosmosis_dtype_string
        return get_string(block, section, name)
    elseif dtype == cosmosis_dtype_int1d
        return get_int_array_1d(block, section, name)
    elseif dtype == cosmosis_dtype_double1d
        return get_double_array_1d(block, section, name)
    elseif dtype == cosmosis_dtype_intnd
        return get_int_array_nd(block, section, name)
    elseif dtype == cosmosis_dtype_doublend
        return get_double_array_nd(block, section, name)
    else
        throw(BlockError(Int32(8), "section was " * string(section) * ", name was " * string(name) *  "type code was " * string(dtype) ))
    end
        

end

function stack_tracer_wrapper(f)
    name = String(Symbol(f))
    return function wrapped_function(a...)
        try
            return f(a...)
        catch e
            s = stacktrace(catch_backtrace())
            println("Function ", name, " failed with error: ", e)
            println("Stack trace:")
            for i in 1 : length(s)
                println("    ", s[i])
            end
            rethrow(e)
        end
    end
end

"""
This function takes a Julia-format cosmosis execute function, which
maps dictionary -> dictionary and gives you a function which instead
maps vector -> vector.  This makes it suitable for automatic
differentiation.

We need to know the input names and lengths to start with
"""
function make_differentiable_wrapper(func, input_names, input_lengths, input_types, config)
    n_input = length(input_names)

    function wrapper(x::Vector{T})::Vector{T} where T

        # turn our vector of inputs into a dict
        d_in = Dict{Tuple{String,String},   Union{T, Array{T}}  }()
        s = 1
        for i in 1:n_input
            l = input_lengths[i]
            section, name = input_names[i]
            if input_types[i] == "scalar"
                d_in[section, name] = x[s]
            else
                d_in[section, name] = x[s:s+l-1]
            end
            s += l
        end


        # run the user execute function
        d_out = func(d_in, config)

        output_lengths = [length(d_out[name]) for name in keys(d_out)]
        total_output_length = sum(output_lengths)
        
        out = zeros(T, total_output_length)
        s = 1
        for (section, name) in keys(d_out)
            l = length(d_out[section, name])
            out[s:s+l-1] .= d_out[section, name]
            s += l
        end
        
        return out
    end 
end


function make_final_execute_function(execute, select_inputs)
    function final_execute(block, config)
        inputs = select_inputs(config)
        d_in = Dict{Tuple{String,String}, Union{Float64, Array{Float64}}}()
        n_input = length(inputs)
        input_lengths = []
        input_types = []

        for (section, name) in inputs
            inp = get(block, section, name)
            d_in[section, name] = inp
            push!(input_lengths, length(inp))
            if isa(inp, Number)
                push!(input_types, "scalar")
            else
                push!(input_types, "array")
            end
        end

        results = execute(d_in, config)

        n_output = length(results)
        output_names = collect(keys(results))
        output_lengths = [length(results[name]) for name in output_names]
        total_output_length = sum(output_lengths)
        output_types = []
        # Fill in results into the data block
        for (sec, name) in output_names
            v = results[sec, name]
            put!(block, sec, name, v)
            if isa(v, Number)
                push!(output_types, "scalar")
            else
                push!(output_types, "array")
            end
        end

        # recompile if results have changed

        wrapper = make_differentiable_wrapper(execute, inputs, input_lengths, input_types, config)


        packed_inputs = zeros(Float64, sum(input_lengths))
        s = 1
        for i in 1:n_input
            section, name = inputs[i]
            l = input_lengths[i]
            packed_inputs[s:s+l-1] .= d_in[section, name]
            s += l
        end    

        # xx = wrapper(packed_inputs)

        # return 0

        jacobian_matrix = ForwardDiff.jacobian(wrapper, packed_inputs)

        # split up the jacobian matrix more carefully by both input
        # parameter row and output parameter column
        s1 = 1
        for i in 1:n_input
            e1 = s1 + input_lengths[i] - 1
            s2 = 1
            (sec1, name1) = inputs[i]
            for j in 1:n_output
                (sec2, name2) = output_names[j]
                e2 = s2 + output_lengths[j] - 1
                key = "$(name2)_wrt_$(sec1)-$(name1)"
                if output_types[j] == "scalar"
                    val = Float64(jacobian_matrix[s2, s1])
                else
                    val = reshape(jacobian_matrix[s2:e2, s1:e1], output_lengths[j], input_lengths[i])
                end
                put!(block, sec2 * "_derivative", key, val)
                s2 = e2 + 1
            end
            s1 = e1 + 1
        end

        return 0
    end
    return final_execute

end

function make_final_execute_function(execute, inputs::Vector{Tuple{String, String}})
    select_inputs(config) = inputs
    return make_final_execute_function(execute, select_inputs)
end


function make_julia_module1(m)
    setup = stack_tracer_wrapper(m.setup)
    if isdefined(m, :select_inputs)
        execute = stack_tracer_wrapper(make_final_execute_function(m.execute, m.select_inputs))
    else
        execute = stack_tracer_wrapper(make_final_execute_function(m.execute, m.inputs))
    end
    if hasproperty(m, :cleanup)
        cleanup = stack_tracer_wrapper(m.cleanup)
    else
        cleanup = function(config) return 0 end
    end
    return (setup, execute, cleanup)
end

make_julia_module = stack_tracer_wrapper(make_julia_module1)

end
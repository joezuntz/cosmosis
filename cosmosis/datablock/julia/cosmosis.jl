module cosmosis


# double, array, grid

datablock_status = Int32
datablock = Ptr{Cvoid}

option_section = "module_options"



struct BlockError <: Exception
    code::datablock_status
    msg::String
    
    function BlockError(code::datablock_status)
        new(code, error_names[code])
    end
    function BlockError(code::datablock_status, msg)
        new(code, error_names[code] * ": " * msg)
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




function make_datablock()
    res = ccall((:make_c_datablock, "libcosmosis.so"), datablock, ())
    res
end


function put_int(block::datablock, section, name, value)
    res = ccall((:c_datablock_put_int, "libcosmosis.so"), 
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
    res = ccall((:c_datablock_get_int, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Int32}), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    value[]
end



function put_double(block::datablock, section, name, value)
    res = ccall((:c_datablock_put_double, "libcosmosis.so"), 
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
    res = ccall((:c_datablock_get_double, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Cdouble}), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name) * ", value was " * string(value) ))
    end
    value[]
end


function put_bool(block::datablock, section, name, value)
    if ((value!=0) && (value!=1))
        throw(BlockError("Bool values in Julia must be 0 or 1"))
    end
    if (value==0)
        uint_value = UInt8(0)
    elseif (value==1)
        uint_value = UInt8(1)
    else
        throw(ArgumentError("Bool values in Julia must be 0 or 1. You passed: " * string(value)))
    end
    res = ccall((:c_datablock_put_bool, "libcosmosis.so"), 
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
    res = ccall((:c_datablock_get_bool, "libcosmosis.so"), 
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
    res = ccall((:c_datablock_get_string, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Cstring}), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    unsafe_string(value[])
end


function put_string(block, section, name, value)
    res = ccall((:c_datablock_put_string, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Cstring), 
            block, section, name, value)
    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end
    res
end


function put_double_array_nd(block, section, name, value)

    ndim = ndims(value)
    extents = convert(Array{Int32}, collect(size(value)))

    res = ccall((:c_datablock_put_double_array, "libcosmosis.so"), 
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
    res = ccall((:c_datablock_get_array_ndim, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Int32}),
            block, section, name, ndim_ptr)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    ndim = ndim_ptr[]    

    extents = Array{Int32}(undef, ndim)

    res = ccall((:c_datablock_get_double_array_shape, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Int32, Ptr{Int32}), 
            block, section, name, ndim, extents)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end


    t = tuple(extents...)

    value = Array{Float64}(undef, t)

    res = ccall((:c_datablock_get_double_array, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Ptr{Cdouble}, Int32, Ptr{Int32}), 
            block, section, name, value, ndim, extents)


    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    value

end



function put_int_array_nd(block, section, name, value)

    ndim = ndims(value)
    extents = convert(Array{Int32}, collect(size(value)))

    res = ccall((:c_datablock_put_int_array, "libcosmosis.so"), 
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
    res = ccall((:c_datablock_get_array_ndim, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Ref{Int32}),
            block, section, name, ndim_ptr)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    ndim = ndim_ptr[]    

    extents = Array{Int32}(undef, ndim)

    res = ccall((:c_datablock_get_int_array_shape, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Int32, Ptr{Int32}), 
            block, section, name, ndim, extents)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end


    t = tuple(extents...)

    value = Array{Int32}(undef, t)

    res = ccall((:c_datablock_get_int_array, "libcosmosis.so"), 
            datablock_status,
            (datablock, Cstring, Cstring, Ptr{Int32}, Int32, Ptr{Int32}), 
            block, section, name, value, ndim, extents)


    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    value

end


function put_double_array_1d(block, section, name, value)

    sz = Int32(length(value))

    res = ccall((:c_datablock_put_double_array_1d, "libcosmosis.so"), 
        datablock_status,
        (datablock, Cstring, Cstring, Ptr{Float64}, Int32), 
        block, section, name, value, sz)

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end


end


function get_double_array_1d(block, section, name)


    sz = ccall((:c_datablock_get_array_length, "libcosmosis.so"), 
        Int32,
        (datablock, Cstring, Cstring), 
        block, section, name)

    println("size = ", sz)
    if (sz<0)
        throw(BlockError(8, "section was " * string(section) * ", name was " * string(name)))
    end

    sz_ptr = Ref{Int32}()
    value = Array{Float64}(undef, sz)

    println("calling")
    res = ccall((:c_datablock_get_double_array_1d_preallocated, "libcosmosis.so"), 
        datablock_status,
        (datablock, Cstring, Cstring, Ptr{Float64}, Ref{Int32}, Int32), 
        block, section, name, value, sz_ptr, sz)
    println("out")

    if (res!=0)
        throw(BlockError(res, "section was " * string(section) * ", name was " * string(name)))
    end

    value
end


end



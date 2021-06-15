import ctypes

def activate_segfault_handling():
    try:
        import faulthandler
    except ImportError:
        raise ImportError("You must install the python faulthandler package to use experimental fault handling")
    from ..datablock.cosmosis_py import enable_cosmosis_segfault_handler
    enable_cosmosis_segfault_handler()
    print("Experimental fault handling enabled")
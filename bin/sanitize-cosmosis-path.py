import os
paths = os.environ['DYLD_LIBRARY_PATH'].split(":")
paths = [p for p in paths if "/ups/gcc/v4_8_2/Darwin64bit+13/" not in p]
print ":".join(paths)
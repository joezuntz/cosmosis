from builtins import object
import collections
import types


enum_names = [
"DBS_SUCCESS",
"DBS_DATABLOCK_NULL",
"DBS_SECTION_NULL",
"DBS_SECTION_NOT_FOUND",
"DBS_NAME_NULL",
"DBS_NAME_NOT_FOUND",
"DBS_NAME_ALREADY_EXISTS",
"DBS_VALUE_NULL",
"DBS_WRONG_VALUE_TYPE",
"DBS_MEMORY_ALLOC_FAILURE",
"DBS_SIZE_NULL",
"DBS_SIZE_NONPOSITIVE",
"DBS_SIZE_INSUFFICIENT",
"DBS_NDIM_NONPOSITIVE",
"DBS_NDIM_OVERFLOW",
"DBS_NDIM_MISMATCH",
"DBS_EXTENTS_NULL",
"DBS_EXTENTS_MISMATCH",
"DBS_LOGIC_ERROR"
]


class EnumGenerator(object):
    """This is a slightly insane hack to generate an enum into the local
    namespace from a list of names.  This avoids a nested namespace.
    Once the list of errors has settled down so we do not need to keep
    retyping we can just hard-code them.
    """
    def __init__(self, namespace):
        self.namespace=namespace
        self.value = 0
    def add(self, name):
        self.namespace[name] = self.value
        self.value += 1

enum = EnumGenerator(locals())
for name in enum_names:
    enum.add(name)

ERROR_MESSAGES = collections.defaultdict(lambda:  "{status}: Uknown error; possibly cosmosis internal failure.  Please contact cosmosis team")
ERROR_MESSAGES.update({
    DBS_DATABLOCK_NULL: "{status}: Null (zero) datablock passed into function (section was {section}, name was {name})",
    DBS_SECTION_NULL: "{status}: Null name passed into function (section was {section}",
    DBS_SECTION_NOT_FOUND: "{status}: Could not find section called {section} (name was {name})",
    DBS_NAME_NULL: "{status}: Null name passed into function (section was {section})",
    DBS_NAME_NOT_FOUND: "{status}: Could not find name {name} in section {section}",
    DBS_NAME_ALREADY_EXISTS: "{status}: Tried to overwrite {name} in section {section}.  Use the replace functions to over-write",
    DBS_VALUE_NULL: "{status}: Passed a null value into function (section was {section}, name was {name})",
    DBS_WRONG_VALUE_TYPE: "{status}: Wrong value type for {name} in section {section}",
    DBS_MEMORY_ALLOC_FAILURE: "{status}: Failed to allocate memory for {name} in section {section}.",
    DBS_SIZE_NULL: "{status}: Null parameter for returned size passed into function (section was {section}, name was {name})",
    DBS_SIZE_NONPOSITIVE: "{status}: Non-positive maximum size passed into function (section was {section}, name was {name})",
    DBS_SIZE_INSUFFICIENT: "{status}: Size of passed in array not large enough for what is needed (section was {section}, name was {name})",
    DBS_NDIM_NONPOSITIVE: "{status}: Number of dimensions not positive (section was {section}, name was {name})",
    DBS_NDIM_OVERFLOW: "{status}: Number of dimensions in array is larger than can be represented in an 'int' (section was {section}, name was {name})",
    DBS_NDIM_MISMATCH: "{status}: Number of dimensions supplied for array does not match the array dimensions (section was {section}, name was {name})",
    DBS_EXTENTS_NULL: "{status} Null value passed for array extents (section was {section}, name was {name})",
    DBS_EXTENTS_MISMATCH: "{status} Supplied array extents do not match the extents of the stored array (section was {section}, name was {name})",
    DBS_LOGIC_ERROR: "{status}: Internal cosmosis logical error.  Please contact cosmosis team (section was {section}, name was {name})",
})

ERROR_CLASSES = {}

class CosmosisError(Exception):
    pass

 #Might want to create more of a subclass structure here.
 #Have them all inherit from BlockError
class BlockError(CosmosisError):
    def __init__(self, status, section, name):
        self.name=name
        self.status=status
        self.section=section

    @staticmethod
    def exception_for_status(status, section, name):
        return ERROR_CLASSES[status](status, section, name)
    
    def __str__(self):
        return ERROR_MESSAGES[self.status].format(status=self.status, name=self.name,section=self.section)


def underscore_to_camelcase(value):
    return "".join(str.capitalize(x) if x else '_' for x in value.split("_"))

#for each status except the success one we create an error class
for name in enum_names[1:]:
    class_name = 'Block'+underscore_to_camelcase(name[4:].lower())
    cls = type(class_name, (BlockError,), {})
    status = locals()[name]
    locals()[class_name] = cls
    ERROR_CLASSES[status] = cls 


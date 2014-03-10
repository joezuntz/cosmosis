import collections

DBS_SUCCESS = 0
DBS_DATABLOCK_NULL = 1
DBS_SECTION_NAME_NULL = 2
DBS_SECTION_NOT_FOUND = 3
DBS_NAME_NULL = 4
DBS_NAME_NOT_FOUND = 5
DBS_NAME_ALREADY_EXISTS = 6
DBS_VALUE_NULL = 7
DBS_WRONG_VALUE_TYPE = 8
DBS_MEMORY_ALLOC_FAILURE = 9

ERROR_MESSAGES = collections.defaultdict(lambda:  "{status}: Uknown error; probably cosmosis internal failure.  Please contact cosmosis team")
ERROR_MESSAGES.update({
	DBS_DATABLOCK_NULL: "{status}: Null (zero) datablock passed into function (section was {section}, name was {name})",
	DBS_SECTION_NAME_NULL: "{status}: Null name passed into function (section was {section}",
	DBS_SECTION_NOT_FOUND: "{status}: Could not find section called {section} (name was {name})",
	DBS_NAME_NULL: "{status}: Null name passed into function (section was {section})",
	DBS_NAME_NOT_FOUND: "{status}: Could not find name {name} in section {section}",
	DBS_NAME_ALREADY_EXISTS: "{status}: Tried to overwrite {name} in section {section}.  Use the replace functions to over-write",
	DBS_VALUE_NULL: "{status}: Passed a null value into function (section was {section}, name was {name})",
	DBS_WRONG_VALUE_TYPE: "{status}: Tried to overwrite {name} in section {section} with a value of the wrong type",
	DBS_MEMORY_ALLOC_FAILURE: "{status}: Failed to allocate memory for {name} in section {section}.",
})

class BlockError(Exception):
	def __init__(self, status, section, name):
		self.name=name
		self.status=status
		self.section=section
	def __str__(self):
		return ERROR_MESSAGES[self.status].format(status=self.status, name=self.name,section=self.section)



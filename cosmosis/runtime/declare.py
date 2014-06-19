
from cosmosis.datablock import option_section
# from cosmosis.datablock.cosmosis_py.block import option_section
    
def mksetup(cls):
    def setup(config):
        return cls(config,option_section)
    return setup

def execute(block, state):
    return state.execute(block)

def cleanup(state):
    return state.cleanup()
    
def declare_module(class_obj):
    tmp = __import__(class_obj.__dict__['__module__'])
    tmp.__dict__['execute']=execute
    tmp.__dict__['setup']=mksetup(class_obj)
    tmp.__dict__['cleanup']=cleanup
    tmp.__dict__['pymodule']=class_obj

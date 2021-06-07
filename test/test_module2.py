from cosmosis.runtime import register_new_parameter

def setup(options):
    # example of how to register a new parameter in the pipeline
    # from where.
    register_new_parameter(
        options, # the options datablock we were just passed
        "new_parameters", # section
        "p3",  # name
        -1.0,  # min value
        0.0,   # start value
        1.0,   # max value
        "normal",  # prior name, optional, defaults to uniform or delta if min==max.  see prior.py
        [0.1, 0.2] # prior parameters, optional, in this case [mu, sigma], see prior.py
    )
    register_new_parameter(
        options, # the options datablock we were just passed
        "new_parameters", # section
        "delta",  # name
        1.0,  # min value
        1.0,   # start value
        1.0,   # max value
    )
    return {}

def execute(block, config):
    p1 = block['parameters', 'p1']
    p2 = block['parameters', 'p2']
    p3 = block['new_parameters', 'p3']
    delta = block['new_parameters', 'delta']
    assert delta == 1.0
    return 0

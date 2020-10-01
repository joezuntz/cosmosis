from cosmosis.runtime import register_new_parameter
def setup(options):

    register_new_parameter(
        options,
        "new_parameters",
        "p3",
        -1.0,  # min value
        0.0,   # start value
        1.0,   # max value
        "normal",  # prior name, optional, defaults to uniform or delta if min==max
        [0.1, 0.2] # prior parameters, in this case mu, sigma, see prior.py
    )
    return {}

def execute(block, config):
    p1 = block['parameters', 'p1']
    p2 = block['parameters', 'p2']
    p3 = block['new_parameters', 'p3']
    return 0

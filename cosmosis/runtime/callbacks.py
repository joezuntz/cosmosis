

PRESCRIPT_RUN = "pre-script-run"
OUTPUT_READY = "output-ready"
SAMPLER_CONFIGURED = "sampler-configured"
SAMPLER_RESUMED = "sampler-resumed"
SAMPLER_CONVERGED = "sampler-converged"
SAMPLER_EXECUTING = "sampler-executing"
SAMPLER_EXECUTED = "sampler-executing"
SAMPLER_WORKING = "sampler-working"
POSTSCRIPT_RUN = "post-script-run"
MODULE_SET_UP = "module-set-up"
MODULE_RUN_START = "module-run-start"
MODULE_RUN_FAIL = "module-run-fail"
MODULE_RUN_SUCCESS = "module-run-success"
ALL_MODULES_SET_UP = "all-modules-set-up"
PIPELINE_RUN_START = "pipeline-run-start"
PIPELINE_RUN_SUCCESS = "pipeline-run-success"
PIPELINE_RESULTS_SUCCESS = "pipeline-results-success"
PIPELINE_RUN_FAIL = "pipeline-run-fail"

def null_callback(event, details):
    pass

def print_callback(event, details):
    print(event, details)

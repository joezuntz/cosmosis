import random
import config
import prior as priors # to avoid breaking other stuff below
import numpy as np

class Parameter(object):
    def __init__(self, section, name, start, limits=None, prior=None):
        if not limits:
            self.limits = (start, start)
        else:
            self.limits = limits
            if (limits[1]<limits[0]):
                raise ValueError("Your ini file specified that "
                    "parameter %s in section %s had upper limit "
                    "< lower limit"% (name, section))

        self.section = section
        self.name = name
        self.start = start

        if prior is None:
            if limits is None:
                # Parameter has no setting in the priors file and is fixed
                prior = priors.DeltaFunctionPrior(start)
            else:
                # Parameter has no setting in the priors file and is variable
                prior = priors.UniformPrior(limits[0], limits[1])
        else:
            if limits is None:
                # Parameter does have setting in the priors file but is fixed - just fix value
                prior = priors.DeltaFunctionPrior(start)
            else:
                # Parameter does have setting in the priors file and is variable - truncate prior to limits
                prior = prior.truncate(limits[0], limits[1])

        self.prior = prior


        # TODO: check consistency of prior with limits

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            try:
                section, name = other
            except ValueError:
                return False
            return (self.section == section and
                    self.name == name)
        elif isinstance(other, Parameter):
            return (self.section == other.section and
                    self.name == other.name)

    def __str__(self):
        return self.section + "--" + self.name

    def __repr__(self):
        return self.__str__()

    def is_fixed(self):
        return self.limits[0] == self.limits[1]

    def is_varied(self):
        return not self.is_fixed()

    def in_range(self, p):
        return self.limits[0] <= p <= self.limits[1]

    def width(self):
        return self.limits[1] - self.limits[0]

    def random_point(self):
        return self.prior.sample()

    def normalize(self, p):
        if self.is_fixed():
            return 0.0
        else:
            return (p - self.limits[0]) / (self.limits[1] - self.limits[0])

    def denormalize(self, p, raise_exception=True):
        if 0.0 <= p <= 1.0:
            return p*(self.limits[1]-self.limits[0]) + self.limits[0]
        elif not raise_exception:
            return p*(self.limits[1]-self.limits[0]) + self.limits[0]
        else:
            raise ValueError("parameter value not normalized")

    def denormalize_from_prior(self, p):
        if 0.0 <= p <= 1.0:
            return self.prior.denormalize_from_prior(p)
        else:
            raise ValueError("parameter value not normalized")


    def evaluate_prior(self, p):
        if p < self.limits[0] or p > self.limits[1]:
            return -np.inf
        elif self.prior:
            return self.prior(p)
        else:
            return 0.0

    @staticmethod
    def load_parameters(value_file, priors_files=None, override=None):
        values_ini = config.Inifile(value_file)

        if priors_files:
            priors_data = priors.Prior.load_priors(priors_files)
        else:
            priors_data = {}

        parameters = []
        for (section, name), value in values_ini:
            #override if available
            if (override is not None) and (section, name) in override:
                value = override[(section,name)]

            # parse value line
            start, limits = Parameter.parse_parameter(value)

            # check for prior
            pri = priors_data.get((section, name), None)


            parameters.append(Parameter(section, name,
                                        start, limits, pri))

        return parameters

    @staticmethod
    def parse_parameter(line):
        try:
            values = [float(p) for p in line.split()]
            if len(values) == 1:
                if values[0]==int(values[0]):
                    try:
                        v = int(line)
                        return v, None
                    except ValueError:
                        return values[0], None
                return values[0], None
            elif len(values) == 2:
                return 0.5*(values[0]+values[1]), tuple(values)
            elif len(values) == 3:
                return values[1], (values[0], values[2])
            else:
                raise ValueError("Was expecting 1-3 values for "
                                 "parameter value %s" % (line,))
        except ValueError as error:
            raise ValueError("Unable to parse numeric value for "
                             "parameter value %s, error %s" %
                             (line, error))

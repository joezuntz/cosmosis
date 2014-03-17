import random
import config
import prior


class Parameter(object):
    def __init__(section, name, start, limits=None, prior=None):
        if not limits:
            self.limits = (start, start)
        else:
            self.limits = limits

        self.section = section
        self.name = name
        self.start = start

        self.prior = prior
        # check consistency of prior with range

    def __eq__(self, other):
        if instance(other, (list, tuple)):
            try:
                section, name = other
            except ValueError:
                return False
            return (self.section == section and
                    self.name == name)
        elif instanceof(other, Parameter):
            return (self.section == other.section and
                    self.name == other.name)

    def is_fixed(self):
        return self.limits[0] == self.limits[1]

    def is_varied(self):
        return not self.is_fixed()

    def in_range(self, p):
        return self.limits[0] <= p <= self.limits[1]

    def random_point(self):
        return random.uniform(*self.limits)

    def normalize(self, p):
        if self.is_fixed(p):
            return 0.0
        else:
            return (p - self.limits[0]) / (self.limits[1] - self.limits[0])

    def denormalize(self, p):
        if 0.0 <= p <= 1.0:
            return p*(self.limits[1]-self.limits[0]) + self.limits[0]
        else:
            raise ValueError("parameter value not normalized")

    def prior(self, p):
        if self.prior:
            return self.prior(p)
        else:
            return 0.0

    @staticmethod
    def load_parameters(value_file, priors_files=None):
        values_ini = config.Inifile(value_file)

        if priors_files:
            priors = Prior.load_priors(priors_files)
        else:
            priors = {}

        parameters = []
        for section, name, value in values_ini:
            # parse value line
            start, limits = Parameter.parse_parameter(value)

            # check for prior
            prior = priors.get((section, name), None)

            parameters.append(Parameter(section, name,
                                        start, limits, prior))

        return parameters

    @staticmethod
    def parse_parameter(line):
        try:
            values = [float(p) for p in line.split()]
            if len(values) == 1:
                return start, None
            elif len(values) == 2:
                return 0.5*(values[0]+values[1]), tuple(values)
            elif len(values) == 3:
                return values[1], (values[0], values[2])
            else:
                raise ValueError("Was expecting 1-3 values for parameter value %s" % (line,))
        except ValueError as error:
            raise ValueError("Unable to parse numeric value for parameter value %s, error %s" % (line, error))

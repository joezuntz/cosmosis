#coding: utf-8


u"""Definition of the :class:`Inifile` class."""


from future import standard_library
standard_library.install_aliases()
import os
import sys
import collections
import warnings
import configparser

class CosmosisConfigurationError(configparser.Error):
    u"""Something to throw when there is an error in a .ini file particular to Cosmosis.

    The (underlying) object simply carries a string providing information
    to the user.

    """
    pass

class IncludingConfigParser(configparser.ConfigParser):
    u"""Extension of built-in python :class:`ConfigParser` to \%include other files.

    Use the line: %include filename.ini This is assumed to end a section,
    and the last section in the included file is assumed to end as well

    Note that the caller of the `read()` method may set a
    `no_expand_includes` attribute on this object, to cause any %include
    lines to *not* actually be actioned (they will be regarded as
    comments, but still delineate sections).
    """

    def _read(self, fp, fpname):
        u"""Parse a sectioned setup file.

        The sections in setup file contains a title line at the top,
        indicated by a name in square brackets (`[]'), plus key/value
        options lines, indicated by `name: value' format lines.
        Continuations are represented by an embedded newline then
        leading whitespace.  Blank lines, lines beginning with a '#',
        and just about everything else are ignored.

        """
        cursect = None                        # None, or a dictionary
        optname = None
        lineno = 0
        e = None                              # None, or an exception
        while True:
            line = fp.readline()
            if not line:
                break
            lineno = lineno + 1
            # comment or blank line?
            if line.strip() == '' or line[0] in '#;':
                continue
            if line.split(None, 1)[0].lower() == 'rem' and line[0] in "rR":
                # no leading whitespace
                continue
            # continuation line?
            if line[0].isspace() and cursect is not None and optname:
                value = line.strip()
                if value:
                    cursect[optname].append(value)
            # a section header or option header?
            else:
                #JAZ add environment variable expansion
                if not getattr(self, 'no_expand_vars', False):
                    line = os.path.expandvars(line)
                # is it a section header?
                mo = self.SECTCRE.match(line)
                if mo:
                    sectname = mo.group('header')
                    if sectname in self._sections:
                        cursect = self._sections[sectname]
                    elif sectname == configparser.DEFAULTSECT:
                        cursect = self._defaults
                    else:
                        cursect = self._dict()
                        self._sections[sectname] = cursect
                    # So sections can't start with a continuation line
                    optname = None
                # no section header in the file?
                elif line.lower().startswith('%include'):
                    if  not  getattr (self, 'no_expand_includes', False):
                        include_statement, filename = line.split()
                        filename = filename.strip('"').strip("'")
                        sys.stdout.write("Reading included ini file: `"
                                                           + filename + "'\n")
                        if not os.path.exists(filename):
                            # TODO: remove direct sys.stderr writes
                            sys.stderr.write("Tried to include non-existent "
                                             "ini file: `" + filename + "'\n")
                            raise IOError("Tried to include non-existent "
                                          "ini file: `" + filename + "'\n")
                        self.read(filename)
                    cursect = None
                elif cursect is None:
                    raise configparser.MissingSectionHeaderError(fpname,
                                                                 lineno,
                                                                 line)
                # an option line?
                else:
                    mo = self._optcre.match(line)
                    if mo:
                        optname, vi, optval = mo.group('option', 'vi', 'value')
                        optname = self.optionxform(optname.rstrip())
                        # This check is fine because the OPTCRE cannot
                        # match if it would set optval to None
                        if optval is not None:
                            if vi in ('=', ':') and ';' in optval:
                                # ';' is a comment delimiter only if it
                                # follows a spacing character
                                pos = optval.find(';')
                                if pos != -1 and optval[pos-1].isspace():
                                    optval = optval[:pos]
                            optval = optval.strip()
                            # allow empty values
                            if optval == '""':
                                optval = ''
                            cursect[optname] = [optval]
                        else:
                            # valueless option handling
                            cursect[optname] = optval
                    else:
                        # a non-fatal parsing error occurred.  set up the
                        # exception but keep going. the exception will be
                        # raised at the end of the file and will contain
                        # a list of all bogus lines
                        if not e:
                            e = configparser.ParsingError(fpname)
                        e.append(lineno, repr(line))
        # if any parsing errors occurred, raise an exception
        if e:
            raise e

        # join the multi-line values collected while reading
        all_sections = [self._defaults]
        all_sections.extend(list(self._sections.values()))
        for options in all_sections:
            for name, val in list(options.items()):
                if isinstance(val, list):
                    options[name] = '\n'.join(val)




class Inifile(IncludingConfigParser):

    u"""A dictionary of `(section, name) -> value` pairs corresponding to entries in a .ini file.

    The class is designed to hide the details of parsing .ini files, and
    then for creating :class:`DataBlock` objects (which wrap C objects
    and can be passed down a processing pipeline which may include modules
    written in C) via the :class:`Pipeline` and then :class:`Module`
    constructors (see `Pipeline.__init__()`).

    The values are all stored as strings, with methods provided to locate
    and interpret them as integers, floating-point numbers, booleans, or
    arrays of integers or floating-point numbers.

    """

    def __init__(self, filename, defaults=None, override=None):
        u"""Read in a configuration from `filename`.

        The `defaults` will be applied if a parameter is not specified in
        the file (or \%included descendants), and the `override`s will be
        imposed on the regardless of whether those parameters have
        assigned values or not.

        Where supplied, `defaults` and `override` should be dictionary
        mappings of `(section, name) -> value`.

        """

        IncludingConfigParser.__init__(self,
                                       defaults=defaults,
                                       dict_type=collections.OrderedDict,
                                       strict=False)
        # default read behaviour is to ignore unreadable files which
        # is probably not what we want here
        if filename is not None:
            if isinstance(filename,str) and not os.path.exists(filename):
                raise IOError("Unable to open configuration file `" + filename + "'")
            self.read(filename)

        # override parameters
        if override:
            for section, name in override:
                if section=="DEFAULT":
                    self._defaults[name] = override[(section,name)]
                else:
                    if not self.has_section(section):
                        self.add_section(section)
                    self.set(section, name, override[(section, name)])



    def __iter__(self):
        u"""Iterate over all the parameters.

        The value of the iterator is `((section, name), value)`.

        """
        return (((section, name), value) for section in self.sections()
                for name, value in self.items(section))



    def items(self, section, raw=False, vars=None, defaults=True):
        u"""Return a list of pairs (key, value) from all the options in a given `section`.

        If raw is set, do not replace values which are set using the ini file 
        interpolation syntax %(name)s.

        If vars is set to a dictionary, use it as an additional source of options.

        If defaults is True (the default), parameters in the [DEFAULT] section are included
        in all other sections.


        """
        if defaults:
            return IncludingConfigParser.items(self, section, raw=raw, vars=vars)
        else:
            d = collections.OrderedDict()
            try:
                d.update(self._sections[section])
            except KeyError:
                if section != configparser.DEFAULTSECT:
                    raise configparser.NoSectionError(section)
            # Update with the entry specific variables
            if vars:
                for key, value in list(vars.items()):
                    d[self.optionxform(key)] = value
            options = list(d.keys())
            if raw:
                return [(option, d[option])
                        for option in options]
            else:
                return [(option, self._interpolate(section, option, d[option], d))
                        for option in options]


    def get(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):

        u"""Get a value as a string, or `default` if the value is not in the dictionary.

        If the `default` is not set and is needed, an error with a message
        to the user will be raised. (`None` is *not* acceptable as a
        default).

        """
        try:
            return IncludingConfigParser.get(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for an option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            else:
                return fallback

    # these functions override the default parsers to allow for extra formats
    def getint(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):
        u"""Get a value as an integer, or return `default` if the value is not found.
        
        If the `default` is not set and is needed, an error with a message
        to the user will be raised. (`None` is *not* acceptable as a
        default).

        """
        try:
            return IncludingConfigParser.getint(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, CosmosisConfigurationError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for an integer option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            elif not isinstance(fallback, int):
                raise TypeError("Default not integer")
            else:
                return fallback

    def getfloat(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):
        u"""Get a floating-point value from the dictionary, with `default`.

        If the value is not found in the dictionary and `default` is
        specified, then `default` will be returned.  Otherwise an error
        will be thrown with a useful message for the user.

        """
        try:
            return IncludingConfigParser.getfloat(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, CosmosisConfigurationError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for a float option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            elif not isinstance(fallback, float):
                raise TypeError("Default not float")
            else:
                return fallback

    def getboolean(self, section, option, raw=False, vars=None, fallback=configparser._UNSET):
        u"""Interpret a parameter as a boolean, including symbolic values.

        This essentially allows a configuration file to represent boolean
        values in the most convenient manner (‘true’, ‘n’, etc) as well as
        with zero/non-zero numerical values.

        If the parameter is not found in the dictionary, then `default`
        will be returned, which will itself default to `False` if not
        specified.

        """
        try:
            return IncludingConfigParser.getboolean(self, section, option, raw=raw, vars=vars, fallback=fallback)
        except ValueError:
            # additional options t/y/n/f
            value = self.get(section, option).lower()
            if value in ['y', 'yes', 't','true']:
                return True
            elif value in ['n', 'no', 'f', 'false']:
                return False
            else:
                raise ValueError("Unable to parse parameter "
                                 "%s--%s = %s into boolean form"
                                 % (section, option, value))
        except (configparser.NoSectionError, configparser.NoOptionError, CosmosisConfigurationError) as e:
            if fallback is configparser._UNSET:
                raise CosmosisConfigurationError("CosmoSIS looked for a boolean (T/F) option called '%s' in the '[%s]' section, but it was not in the ini file"%(option,section))
            elif not isinstance(fallback, bool):
                raise TypeError("Default not boolean")
            else:
                return fallback



    def gettyped(self, section, name):
        u"""Best-guess the type of a parameter and return it as that type.

        The method will try parsing the value as, in this order:
            an integer or list of integers,
            a float or list of floats,
            a complex number or list of complex numbers,
            a boolean,
            a string.

        The string value is the fallback if all else fails.
        """

        import re

        value = IncludingConfigParser.get(self, section, name)
        value = value.strip()
        # There isn't really a sensible thing to return for this,
        # so we just need to set it to None.
        if not value: return None
        # try quoted string
        m = re.match(r"^(['\"])(.*?)\1$", value)
        if m is not None:
            return m.group(2)

        value_list = value.split()

        # Try to match integer array.  This will fail whenever a decimal
        # point occurs anywhere in the list of values.
        try:
            parsed = [int(s) for s in value_list]
            if len(parsed) == 1:
                return parsed[0]
            else:
                return parsed
        except ValueError:
            pass

        # try to match float array
        try:
            parsed = [float(s) for s in value_list]
            if len(parsed) == 1:
                return parsed[0]
            else:
                return parsed
        except ValueError:
            pass

        # try to match complex array
        try:
            parsed = [complex(s) for s in value_list]
            if len(parsed) == 1:
                return parsed[0]
            else:
                return parsed
        except ValueError:
            pass

        # try to match boolean (no array support)

        try:
            return self.getboolean(section, name)
        except ValueError:
            pass

        # default to string
        return value

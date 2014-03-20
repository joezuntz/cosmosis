import os
import sys
import collections
import warnings
import ConfigParser


class IncludingConfigParser(ConfigParser.ConfigParser):
    """ Extension of ConfigParser to \%include other files.
        Use the line:
        %include filename.ini
        This is assumed to end a section, and the last section
        in the included file is assumed to end as well
    """
    def _read(self, fp, fpname):
        """Parse a sectioned setup file.

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
                # is it a section header?
                mo = self.SECTCRE.match(line)
                if mo:
                    sectname = mo.group('header')
                    if sectname in self._sections:
                        cursect = self._sections[sectname]
                    elif sectname == ConfigParser.DEFAULTSECT:
                        cursect = self._defaults
                    else:
                        cursect = self._dict()
                        cursect['__name__'] = sectname
                        self._sections[sectname] = cursect
                    # So sections can't start with a continuation line
                    optname = None
                # no section header in the file?
                elif line.lower().startswith('%include'):
                    include_statement, filename = line.split()
                    filename = filename.strip('"')
                    filename = filename.strip("'")
                    sys.stdout.write("Reading included ini file: " % (filename,))
                    if not os.path.exists(filename):
                        sys.stderr.write("Tried to include non-existent ini file: %s\n" % (filename,))
                        raise IOError("Tried to include non-existent ini file: %s\n" % (filename,))
                    self.read(filename)
                    cursect = None
                elif cursect is None:
                    raise ConfigParser.MissingSectionHeaderError(fpname, lineno, line)
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
                                # ';' is a comment delimiter only if it follows
                                # a spacing character
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
                        # raised at the end of the file and will contain a
                        # list of all bogus lines
                        if not e:
                            e = ConfigParser.ParsingError(fpname)
                        e.append(lineno, repr(line))
        # if any parsing errors occurred, raise an exception
        if e:
            raise e

        # join the multi-line values collected while reading
        all_sections = [self._defaults]
        all_sections.extend(self._sections.values())
        for options in all_sections:
            for name, val in options.items():
                if isinstance(val, list):
                    options[name] = '\n'.join(val)


class Inifile(IncludingConfigParser):
    def __init__(self, filename, defaults=None, override=None):
        IncludingConfigParser.__init__(self,
                                       defaults=defaults,
                                       dict_type=collections.OrderedDict)
        self.read(filename)

        # override parameters
        if override:
            for section, name in override:
                if not self.has_section(section):
                    self.add_section(section)
                self.set(section, name, override[(section, name)])

    def __iter__(self):
        return (((section, name), value) for section in self.sections()
                for name, value in self.items(section))

    def get(self, section, name, default=None):
        try:
            return IncludingConfigParser.get(self, section, name)
        except (ConfigParser.NoSectionError,ConfigParser.NoOptionError) as e:
            if default is None:
                raise e
            else:
                return default

    # these functions override the default parsers to allow for extra formats
    def getint(self, section, name, default=None):
        try:
            return IncludingConfigParser.getint(self, section, name)
        except (ConfigParser.NoSectionError,ConfigParser.NoOptionError) as e:
            if default is None:
                raise e
            elif not isinstance(default, int):
                raise TypeError("Default not integer")
            else:
                return default

    def getfloat(self, section, name, default=None):
        try:
            return IncludingConfigParser.getfloat(self, section, name)
        except (ConfigParser.NoSectionError,ConfigParser.NoOptionError) as e:
            if default is None:
                raise e
            elif not isinstance(default, float):
                raise TypeError("Default not float")
            else:
                return default       

    def getboolean(self, section, name, default=False):
        try:
            return IncludingConfigParser.getboolean(self, section, name)
        except (ConfigParser.NoSectionError,ConfigParser.NoOptionError) as e:
            if default is None:
                raise e
            elif not isinstance(default, bool):
                raise TypeError("Default not boolean")
            else:
                return default
        except ValueError:
            # additional options t/y/n/f
            value = self.get(section, name).lower()
            if value.startswith('y') or value.startswith('t'):
                return True
            elif value.startswith('n') or value.startswith('f'):
                return False
            else:
                raise ValueError("Unable to parse parameter %s--%s = %s into boolean form"
                                 % (section, name, value))

    def gettyped(self, section, name):
        import re

        value = IncludingConfigParser.get(self, section, name)

        # try quoted string
        m = re.match(r"^(['\"])(.*?)\1$", value)
        if m is not None:
            return m.group(2)

        value_list = value.split()

        # try to match integer array
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

"""
Access credentials stored in credential.yaml in the home directory.
"""

import os
import json

class Credentials(object):
    

    creds = None

    def __init__(self, filepath):
        self.filepath = filepath

    def get_filepath(self):
        return self.filepath

    def get(self, keys=None):
        """Returns a specific key. Can be nested.

        Parameters
        ----------
        keys: list or tuple, keys used to traverse a nested graph.  For one-layer searches just enter a one-element iterable.
        """

        if self.creds is None:
            self.__load_file()

        return DictQuery(self.creds).get(keys) if keys else self.creds

    def __load_file(self):
        _, ext = os.path.splitext(self.filepath)

        if ext == '.json':
            with open(self.filepath, 'r') as _f:
                self.creds = json.load(_f)
        else:
            raise UnknownFileType(self.filepath)


class DictQuery(dict):
    def get(self, keys, default=None):
        val = None

        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [ v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(self, key, default)

            if not val:
                break;

        return val


class UnknownFileType(Exception):
    """File type is not expected."""
    pass
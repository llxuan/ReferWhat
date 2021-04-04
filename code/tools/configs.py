''' A config class that reads YAML config file and tracks the changes/accesses '''

import yaml

UPGRADING_VALUEERR = ValueError("Up-grading a single value to a dict is forbiden")
DOWNGRADING_VALUEERR = ValueError("Down-grading a dict to a single value is forbidden")

class Field(object):
    def __init__(self, v=None):
        self.val = v
        self.is_overwritten = False
        self.is_used = False
    def overwrite(self, value):
        self.is_overwritten = True
        self.val = value
        return value
    def set_used(self):
        self.is_used = True
    def __repr__(self):
        return self.val
    def __str__(self):
        return str(self.val)
    
class Config(object):
    def __init__(self, filepath = None):
        self.filepath = filepath
        self.fields = dict()
        if self.filepath is not None:
            self.load_from_file(self.filepath)

    def load_from_file(self, filepath):
        with open(filepath, 'r') as f:
            self.load_from_dict(yaml.load(f))
        return self

    def load_from_dict(self, d):
       def _convert(d_ref , d):
           for k, v in d.items():
               if type(v) is dict:
                   if k not in d_ref:
                       d_ref[k] = Config().load_from_dict(v)
                   else:
                       if type(d_ref[k]) is Field:
                           raise UPGRADING_VALUEERR
                       else:
                           ''' merge two dicts '''
                           _convert(d_ref[k].fields, v)
               else:
                   if k in d_ref:
                       if type(d_ref[k]) is Config:
                           raise DOWNGRADING_VALUEERR
                       else:
                           d_ref[k].overwrite(v)
                   else:
                       d_ref[k] = Field(v)
       _convert(self.fields, d)
       return self

    def dump_yaml(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.as_dict(), f)

    def as_dict(self):
        def _convert(d):
            ret = dict()
            for k, v in d.items():
                if type(v) is Config:
                    ret[k] = _convert(v.fields)
                else:
                    ret[k] = v.val
            return ret
        return _convert(self.fields)

    def _walk(self, leaf_pred = lambda x:x):
        def walker(d, acc):
            for k, f in d.items():
                if type(f) is Config:
                    return walker(f.fields, acc)
                elif leaf_pred(f):
                    acc.append(k)
            return acc
        return walker(self.fields, [])

    def get_overwritten_fields(self):
        return self._walk(lambda f:f.is_overwritten)

    def get_unused_fields(self):
        return self._walk(lambda f: not f.is_used)

    def __str__(self):
        return str(self.as_dict())

    def __getattr__(self, k):
        if type(self.fields[k]) is Field:
            self.fields[k].set_used()
            return self.fields[k].val
        else:
            return self.fields[k]

    # NOTE this one may contain protaintial bug.
    def __setattr__(self, k , v ):
        if k in ['filepath', 'fields']:
            self.__dict__[k] = v
        elif type(v) is dict: # k is not normal, v is dict.
            if type(self.fields[k]) is Config:
                self.fields[k].load_from_dict(v)
            else: # is Field
                raise UPGRADING_VALUEERR
        else: # v is not dict.
            if type(self.fields[k]) is Config:
                raise DOWNGRADING_VALUEERR
            else: # is Field
                self.fields[k].overwrite(v)

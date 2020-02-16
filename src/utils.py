class ObjectDict(dict):
    """
    Interface similar to an argparser
    """
    def __init__(self):
        pass
    
    def __setattr__(self, attr, value):
        self[attr] = value
        return self[attr]
    
    def __getattr__(self, attr):
        if attr.startswith('_'):
            # https://stackoverflow.com/questions/10364332/how-to-pickle-python-object-derived-from-dict
            raise AttributeError
        return dict(self)[attr]
    
    @property
    def __dict__(self):
        return dict(self)

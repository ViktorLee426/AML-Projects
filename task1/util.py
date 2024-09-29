def make_serializable(f, name):
    f.__name__ = name
    return f
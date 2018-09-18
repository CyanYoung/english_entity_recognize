def sent2label(quaples):
    label = list()
    for quaple in quaples:
        label.append(quaple['label'])
    return label


def map_pos(pos):
    if pos[0] == 'J':
        return 'a'
    elif pos[0] == 'V':
        return 'v'
    elif pos[0] == 'R':
        return 'r'
    else:
        return 'n'


def map_path(name, paths):
    if name in paths:
        return paths[name]
    else:
        raise KeyError


def map_func(name, funcs):
    if name in funcs:
        return funcs[name]
    else:
        raise KeyError


def map_model(name, models):
    if name in models:
        return models[name]
    else:
        raise KeyError

pos_dict = {'J': 'a',
            'V': 'v',
            'R': 'r'}


def map_pos(pos):
    if pos[0] in pos_dict:
        return pos_dict[pos[0]]
    else:
        return 'n'


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError

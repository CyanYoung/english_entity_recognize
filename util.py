pos_dict = {'J': 'a',
            'V': 'v',
            'R': 'r'}


def map_pos(pos):
    if pos[0] in pos_dict:
        return pos_dict[pos]
    else:
        return 'n'


def sent2label(quaples):
    label = list()
    for quaple in quaples:
        label.append(quaple['label'])
    return label


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError

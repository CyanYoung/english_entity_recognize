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


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError

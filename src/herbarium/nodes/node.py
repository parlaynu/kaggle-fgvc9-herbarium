
class Node:
    def __init__(self, inode):
        self.inode = inode
    
    @property
    def fullname(self):
        klass = self.__class__
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__
        return module + '.' + klass.__qualname__


def get_root(node):
    while node.inode is not None:
        node = node.inode
    return node


def iter_rev(node):
    while node is not None:
        yield node
        node = node.inode


def iter_fwd(node):
    inode = node.inode
    if inode is not None:
        yield from iter_fwd(inode)
    yield node


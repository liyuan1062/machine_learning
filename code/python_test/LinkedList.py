# coding=utf-8
class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        print("type is Node, data={}".format(self.data))

class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None
        self.current = self.head

    def append(self, data):
        node = Node(data)
        if self.head is None:
            self.head = node
            self.tail = node
            self.current = node
        else:
            self.tail.next = node
            self.tail = node

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is not None:
            data = self.current.data
            self.current = self.current.next
            return data
        else:
            raise StopIteration()

    def removeFisrt(self, data):
        if self.head is None:
            return Exception("Empty linked list!")
        remove = 0
        cur = self.head
        pre_cur = None
        while cur.next is not None:
            if cur.data == data:
                if pre_cur is None:
                    self.head = cur.next
                else:
                    pre_cur.next = cur.next
                remove = 1
            else:
                pre_cur = cur
            cur = cur.next
        self.resetIter()
        if remove == 0:
            raise Exception("data {} not found".format(data))

    def resetIter(self):
        self.current = self.head

    def reverse(self):
        if self.head is None or self.head.next is None:
            return self
        cur = self.head
        pre = None
        # self.head, self.tail = self.tail, self.head
        self.tail = self.head
        while cur is not None:
            self.head = cur
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        self.resetIter()



if __name__ == '__main__':
    ls = LinkedList()
    ls.append(1)
    ls.append(3)
    ls.append(5)
    # ls.append(1)
    # ls.removeFisrt(3)
    ls.reverse()
    for i in ls:
        print(i)





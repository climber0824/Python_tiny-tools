class TrieNode:
    
    def __init__(self):
        self.children = {}
        self.isEnd = False


class phoneBook:

    def __init__(self):
        self.phoneDic = {}
        self.root = TrieNode() 

    def add(self, name, phone):
        node = self.root
        if name not in self.phoneDic:
            self.phoneDic[name] = phone
        else:
            print(name, 'exist')
        
        for char in name:
            if char in node.children:
                node = node.children[char]
            else:
                node.children[char] = TrieNode()
                node = node.children[char]
     
    def search(self, name):
        if name in self.phoneDic:
            print(name, self.phoneDic[name])
        else:
            print(name, 'not exist')

    def searchPrefix(self, prefix):
        node = self.root
        res = []
        for char in prefix:
            if char not in node.children:
                print('not match')
                break
            node = node.children[char]
        #print('prefix', prefix, node.children, len(node.children))
        #if node and node.children:
        #    print('node', node.children)
        #    node = node.children
        


if __name__ == '__main__':
    phone_book = phoneBook()
    phone_book.add('Jack', '0912123123')
    phone_book.add('Jac', '0912123124')
    phone_book.add('Jak', '0912123125')
    phone_book.add('PPPa', '0912123126')
    phone_book.add('AAack', '0912123127')
    phone_book.add('Jac', '0912123125')
    phone_book.add('Jaj', '0912123155')
    phone_book.add('Jjj', '0912123199')
    phone_book.search('Jak')
    phone_book.search('Jaa')
    phone_book.search('Joo')
    phone_book.search('Jkk')
    phone_book.searchPrefix('Ja')
    phone_book.searchPrefix('J')


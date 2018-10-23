class TriesNode:
    def __init__(self):
        self.trie = {}
        self.size =0

    
    def contains(self, key):
        p = self.trie
        key = key.lstrip()
        for s in key:
            if not s in p:
                return False
            p = p[s]
        if 'val' in p:
            return True
        return False


    def get(self, key):
        p = self.trie
        key = key.lstrip()
        for s in key:
            if not s in p:
                return False
            p = p[s]
        if 'val' in p:
            return p['val']
        return False
        

    def put(self, key, val=0):
        p = self.trie
        key = key.strip()
        for s in key:
            if not s in p:
                p[s] = {}
            p = p[s]
        if key != 'val':
            p['val']= val
        

    # 查找query最长匹配前缀
    def longestPrefixOf(self, query):
        p = self.trie
        query = query.strip()
        pre = ''
        if query != '':
            for s in query:
                if s in p: 
                   pre = pre+s
                   p=p[s]
                else: 
                   return pre
        else:
            return None

    # 递归所有的词
    def word_recur(self, p, word, word_list):
        for key in p:
            if key=='val':
                word_list.append(word)
            else:
                tem = p[key]
                word_t = word + key
                word_list = self.word_recur(tem, word_t, word_list)
        return word_list
    
    # 返回trie树所有词语(迭代器)
    def keys(self):
        p = self.trie
        word_list =[]
        word = ''
        word_list = self.word_recur(p, word, word_list)
        return word_list


    # 返回所有前缀匹配prefix的词语(迭代器)
    def keysWithPrefix(self, prefix):
        p = self.trie
        word_list =[]
        word = prefix
        for s in prefix:
            if not s in p:
                return None
            p=p[s]
        word_list = self.word_recur(p, word, word_list)
        return word_list

    def delete(self,key):
        p = self.trie
        key = key.strip()
        num = []
        if (key !='') and (self.contains(key)) :
            for s in key:
                num.append(len(p[s]))
                p=p[s]
        l =len (key)
        num_i =self.num_get(num,l)
        p = self.trie
        for i in range(0,num_i+1):
            p=p[key[i]]
        del p[key[num_i+1]]

    def num_get(self,num,l):
        for i in range(l-1,0,-1):
            if num[i]>1:
                 return i




if __name__ == '__main__':
    tries_obj=TriesNode()
    tries_obj.put('出去玩')
    tries_obj.put('出去香港玩',20)
    tries_obj.delete('出去香港玩')
    c = tries_obj.keys()
    tries_obj.put('国庆玩')
    a=tries_obj.contains('出去玩')
    tries_obj.longestPrefixOf('出去香港玩')
    b=tries_obj.get('国庆玩')
    tries_obj.longestPrefixOf('出去香港玩')


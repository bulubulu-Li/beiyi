# encoding: utf-8
# a = input("please input a number:")
# print("hello world")

from collections import defaultdict

class Solution():
    def myfun(self,n,p_matrix,q_matrix):
        def dfs(node1,node2):#可能加visited
            if node2 in graph[node1]:
                self.res.append(1)
                self.flag = 1
                return
            else:
                for node in list(graph[node1]):
                    dfs(node,node2)
                    if self.flag==1:
                        return

        graph = [defaultdict(set)]*n
        self.res = []
        visited = [0]*n

        for vector in p_matrix:
            node1, node2 = vector[0],vector[1]
            graph[node1].add(node2)
            graph[node2].add(node1)
        for vector in q_matrix:
            self.flag = 0
            dfs(vector[0],vector[1])
            if self.flag == 0:
                self.res.append(0)
            else:
                self.flag = 0
        return self.flag

P = Solution()
print(P.myfun(10,[[0,1],[1,2]],[[0,2],[0,3]]))
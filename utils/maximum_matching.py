import numpy as np

class MaximumMatching:
    def __init__(self,graph): 
          
        # residual graph 
        self.graph = graph  
        self.ppl = len(graph) 
        self.jobs = len(graph[0]) 
  
    # A DFS based recursive function 
    # that returns true if a matching  
    # for vertex u is possible 
    def bpm(self, u, matchR, seen): 
  
        # Try every job one by one 
        for v in range(self.jobs): 
  
            # If applicant u is interested  
            # in job v and v is not seen 
            if self.graph[u][v] and seen[v] == False: 
                  
                # Mark v as visited 
                seen[v] = True 
  
                '''If job 'v' is not assigned to 
                   an applicant OR previously assigned  
                   applicant for job v (which is matchR[v])  
                   has an alternate job available.  
                   Since v is marked as visited in the  
                   above line, matchR[v]  in the following 
                   recursive call will not get job 'v' again'''
                if matchR[v] == -1 or self.bpm(matchR[v],  
                                               matchR, seen): 
                    matchR[v] = u 
                    return True
        return False
  
    # Returns maximum number of matching  
    def maxBPM(self): 
        '''An array to keep track of the  
           applicants assigned to jobs.  
           The value of matchR[i] is the  
           applicant number assigned to job i,  
           the value -1 indicates nobody is assigned.'''
        matchR = [-1] * self.jobs 
          
        # Count of jobs assigned to applicants 
        result = 0 
        for i in range(self.ppl): 
              
            # Mark all jobs as not seen for next applicant. 
            seen = [False] * self.jobs 
              
            # Find if the applicant 'u' can get a job 
            if self.bpm(i, matchR, seen): 
                result += 1
        return result, matchR
    
class HopcroftKarp():
      def __init__(self, n, adj):
            self.match = np.zeros(n + 1)
            self.adj = adj
            self.n = n
            self.dist = np.zeros(n + 1)
      
      def bfs(self):
            queue = []
            for i in range(1, self.n + 1):
                  if (self.match[i] == 0):
                        self.dist[i] = 0
                        queue.append(i)
                  else :
                        self.dist[i] = 1e9

            self.dist[0] = 1e9

            while (len(queue) > 0):
                  u = queue[0]; queue.pop(0)
                  if (u != 0):
                        for v in self.adj[u]:
                              if (self.dist[self.match[v]] == 1e9):
                                    self.dist[self.match[v]] = self.dist[u] + 1
                                    queue.append(self.match[v])
            
            return (self.dist[0] != 1e9)

      def dfs(self, u):
            if (u != 0):
                  for v in self.adj[u]:
                        if (self.dist[self.match[v]] == self.dist[u] + 1):
                              if (self.dfs(self.match[v])):
                                    self.match[v] = u
                                    self.match[u] = v
                                    return True
                  self.dist[u] = 1e9
                  return False
            return True

      def hopcroft_karp(self):
            result = 0
            while (self.bfs()):
                  for i in range(1, self.n+1):
                        if (self.match[i] == 0 and self.dfs(i)):
                              result = result + 1
            return result, self.match[1:]
      
#a = Hopcroft_karp()


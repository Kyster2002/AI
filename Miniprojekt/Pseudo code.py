from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C']
}


#DFS(graph, node, visited):
  #  If node is not in visited:
   #     Print node  // Process the node
    #    Add node to visited
     #   For each neighbor in graph[node]:
      #      DFS(graph, neighbor, visited)

def dfs(graph, node, visited):
    if node not in visited:
        print(node)  # Process the node
        visited.add(node)
        for neighbor in graph[node]:  # Visit all neighbors
            dfs(graph, neighbor, visited)

visited = set()
dfs(graph, 'A', visited)


#BFS(graph, start):
 #   Create an empty set visited
  #  Create a queue and enqueue start
#
 #   While queue is not empty:
  #      node ← dequeue from queue
   #     If node is not in visited:
    #        Print node  // Process the node
     #       Add node to visited
      #      Enqueue all neighbors of node

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node)  # Process the node
            visited.add(node)
            queue.extend(graph[node])  # Add all neighbors to queue

bfs(graph, 'A')

from collections import deque

#1. Initialize queue with (0,0) state (both jugs empty).
#2. Create a visited set to track seen states.
#3. While queue is not empty:
   #a. Dequeue the front state (a, b).
  # b. If (a, b) contains 2 liters, print the sequence and stop.
 #  c. Generate all possible next states by applying the allowed operations.
#   d. If a new state is not visited, add it to the queue and mark it visited.

def water_jug_bfs(capA, capB, target):
    queue = deque([(0, 0)])  # Start with empty jugs
    visited = set()
    parent = {}  # To track steps

    while queue:
        a, b = queue.popleft()

        if (a, b) in visited:
            continue
        visited.add((a, b))

        # If we reached the target
        if a == target or b == target:
            path = []
            while (a, b) in parent:
                path.append((a, b))
                a, b = parent[(a, b)]
            path.append((0, 0))
            path.reverse()
            return path

        # Generate next possible states
        next_states = set()
        next_states.add((capA, b))  # Fill Jug A
        next_states.add((a, capB))  # Fill Jug B
        next_states.add((0, b))  # Empty Jug A
        next_states.add((a, 0))  # Empty Jug B
        pourAtoB = (a - min(a, capB - b), b + min(a, capB - b))  # Pour A to B
        pourBtoA = (a + min(b, capA - a), b - min(b, capA - a))  # Pour B to A
        next_states.add(pourAtoB)
        next_states.add(pourBtoA)

        # Add valid states to queue
        for state in next_states:
            if state not in visited:
                queue.append(state)
                parent[state] = (a, b)

    return "No solution"


# Run BFS for 4L, 3L jugs to get 2L
solution_path = water_jug_bfs(4, 3, 2)
print(solution_path)




#1. Initialize a stack with (0,0) state.
#2. Use a set to track visited states.
#3. While the stack is not empty:
 #  a. Pop the top state (a, b).
 # b. If (a, b) contains 2 liters, print solution and stop.
 #c. Generate possible states and push them onto the stack if not visited.

def water_jug_dfs(capA, capB, target):
    stack = [(0, 0)]
    visited = set()
    parent = {}

    while stack:
        a, b = stack.pop()

        if (a, b) in visited:
            continue
        visited.add((a, b))

        if a == target or b == target:
            path = []
            while (a, b) in parent:
                path.append((a, b))
                a, b = parent[(a, b)]
            path.append((0, 0))
            path.reverse()
            return path

        next_states = set()
        next_states.add((capA, b))  # Fill Jug A
        next_states.add((a, capB))  # Fill Jug B
        next_states.add((0, b))     # Empty Jug A
        next_states.add((a, 0))     # Empty Jug B
        pourAtoB = (a - min(a, capB - b), b + min(a, capB - b))  # Pour A to B
        pourBtoA = (a + min(b, capA - a), b - min(b, capA - a))  # Pour B to A
        next_states.add(pourAtoB)
        next_states.add(pourBtoA)

        for state in next_states:
            if state not in visited:
                stack.append(state)
                parent[state] = (a, b)

    return "No solution"

# Run DFS
solution_path = water_jug_dfs(4, 3, 2)
print(solution_path)

### Conclusion ###
#BFS is better suited for this problem as it guarantees the shortest path.

#DFS may still find a valid solution but can be inefficient.

#Both methods involve a graph traversal approach where each state (jug state) is a node.
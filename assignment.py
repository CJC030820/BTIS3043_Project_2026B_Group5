import time
import heapq
from collections import deque

# Standard target configuration
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def get_moves(state):
    """Generates valid successor states by moving blank tile '0' UP, DOWN, LEFT, RIGHT."""
    zero_idx = state.index(0)
    row, col = zero_idx // 3, zero_idx % 3
    moves = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < 3 and 0 <= c < 3:
            new_idx = r * 3 + c
            state_list = list(state)
            state_list[zero_idx], state_list[new_idx] = state_list[new_idx], state_list[zero_idx]
            moves.append(tuple(state_list))
    return moves

def manhattan_distance(state):
    """Calculates admissible Manhattan Distance heuristic."""
    distance = 0
    for idx, val in enumerate(state):
        if val != 0:
            target_idx = val - 1
            target_row, target_col = target_idx // 3, target_idx % 3
            curr_row, curr_col = idx // 3, idx % 3
            distance += abs(curr_row - target_row) + abs(curr_col - target_col)
    return distance

# 1. Breadth-First Search (BFS)
def solve_bfs(start_state):
    start_time = time.perf_counter()
    queue = deque([(start_state, [])])
    visited = {start_state}
    nodes_expanded = 0
    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1
        if current == GOAL_STATE:
            runtime_ms = (time.perf_counter() - start_time) * 1000
            return len(path), nodes_expanded, len(visited), runtime_ms
        
        for neighbor in get_moves(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, nodes_expanded, len(visited), (time.perf_counter() - start_time) * 1000

# 2. Greedy Best-First Search
def solve_greedy(start_state):
    start_time = time.perf_counter()
    counter = 0
    pq = [(manhattan_distance(start_state), counter, start_state, [])]
    visited = {start_state}
    nodes_expanded = 0
    while pq:
        _, _, current, path = heapq.heappop(pq)
        nodes_expanded += 1
        if current == GOAL_STATE:
            runtime_ms = (time.perf_counter() - start_time) * 1000
            return len(path), nodes_expanded, len(visited), runtime_ms
        
        for neighbor in get_moves(current):
            if neighbor not in visited:
                visited.add(neighbor)
                counter += 1
                heapq.heappush(pq, (manhattan_distance(neighbor), counter, neighbor, path + [neighbor]))
    return None, nodes_expanded, len(visited), (time.perf_counter() - start_time) * 1000

# 3. A* Search
def solve_a_star(start_state):
    start_time = time.perf_counter()
    counter = 0
    pq = [(manhattan_distance(start_state), 0, counter, start_state, [])]
    visited = {start_state: 0}
    nodes_expanded = 0
    while pq:
        f, g, _, current, path = heapq.heappop(pq)
        nodes_expanded += 1
        if current == GOAL_STATE:
            runtime_ms = (time.perf_counter() - start_time) * 1000
            return len(path), nodes_expanded, len(visited), runtime_ms
        
        for neighbor in get_moves(current):
            new_g = g + 1
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                counter += 1
                f_val = new_g + manhattan_distance(neighbor)
                heapq.heappush(pq, (f_val, new_g, counter, neighbor, path + [neighbor]))
    return None, nodes_expanded, len(visited), (time.perf_counter() - start_time) * 1000
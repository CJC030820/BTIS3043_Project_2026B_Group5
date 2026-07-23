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
        
        # Skip stale nodes if a shorter path to 'current' was already processed
        if g > visited.get(current, g):
            continue
            
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

# --- EXPERIMENTAL TEST BENCHMARK ---
test_cases = [
    {"id": "TC01", "diff": "Easy",   "board": (1, 2, 3, 4, 5, 6, 7, 0, 8)},
    {"id": "TC02", "diff": "Easy",   "board": (1, 2, 3, 4, 5, 6, 0, 7, 8)},
    {"id": "TC03", "diff": "Easy",   "board": (1, 2, 3, 0, 4, 6, 7, 5, 8)},
    {"id": "TC04", "diff": "Medium", "board": (1, 2, 3, 4, 0, 5, 7, 8, 6)},
    {"id": "TC05", "diff": "Medium", "board": (1, 2, 3, 5, 0, 6, 4, 7, 8)},
    {"id": "TC06", "diff": "Medium", "board": (1, 3, 6, 4, 2, 0, 7, 5, 8)},
    {"id": "TC07", "diff": "Medium", "board": (4, 1, 3, 0, 2, 6, 7, 5, 8)},
    {"id": "TC08", "diff": "Hard",   "board": (1, 6, 2, 5, 3, 0, 4, 7, 8)},
    {"id": "TC09", "diff": "Hard",   "board": (2, 8, 3, 1, 6, 4, 7, 0, 5)},
    {"id": "TC10", "diff": "Hard",   "board": (5, 1, 3, 4, 0, 2, 7, 6, 8)}
]

if __name__ == "__main__":
    for tc in test_cases:
        b_len, b_nodes, b_vis, b_time = solve_bfs(tc["board"])
        g_len, g_nodes, g_vis, g_time = solve_greedy(tc["board"])
        a_len, a_nodes, a_vis, a_time = solve_a_star(tc["board"])
        print(f"{tc['id']} ({tc['diff']}): BFS={b_len}s/{b_nodes}n/{b_time:.2f}ms | "
              f"Greedy={g_len}s/{g_nodes}n/{g_time:.2f}ms | A*={a_len}s/{a_nodes}n/{a_time:.2f}ms")

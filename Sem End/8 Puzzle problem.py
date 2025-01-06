from queue import PriorityQueue


def h(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                x, y = divmod(goal.index(state[i][j]), 3)
                distance += abs(x - i) + abs(y - j)
    return distance


def a_star(start, goal):
    start = tuple(tuple(row) for row in start)
    goal_flat = [num for row in goal for num in row]

    open_set = PriorityQueue()
    open_set.put((0, start, []))
    visited = set()

    while not open_set.empty():
        _, current, path = open_set.get()
        if current in visited:
            continue
        visited.add(current)

        if current == tuple(tuple(row) for row in goal):
            return path

        zero_pos = [(i, row.index(0)) for i, row in enumerate(current) if 0 in row][0]
        x, y = zero_pos

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = [list(row) for row in current]
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                new_state = tuple(tuple(row) for row in new_state)
                new_path = path + [new_state]
                open_set.put((len(new_path) + h(new_state, goal_flat), new_state, new_path))

    return None


start_state = [[1, 2, 3], [4, 0, 5], [7, 8, 6]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

solution = a_star(start_state, goal_state)
if solution:
    print("Solution found!")
    for step in solution:
        for row in step:
            print(row)
        print()
else:
    print("No solution found.")

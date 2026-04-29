import pygame
import math
from queue import PriorityQueue
from collections import deque
import sys

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
GRID_COLS   = 40
GRID_ROWS   = 30
CELL_SIZE   = 22
PANEL_W     = 260

GRID_W  = GRID_COLS * CELL_SIZE
GRID_H  = GRID_ROWS * CELL_SIZE
WIN_W   = GRID_W + PANEL_W
WIN_H   = GRID_H

FPS = 60

# ─────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────
BG          = (18,  18,  30)
PANEL_BG    = (24,  24,  40)
PANEL_BORDER= (60,  60,  100)
GRID_LINE   = (35,  35,  55)

C_EMPTY     = (28,  28,  45)
C_BARRIER   = (0,  0,  0)
C_START     = (50,  220, 120)
C_END       = (230, 80,  80)
C_VISITED   = (60,  100, 200)
C_FRONTIER  = (100, 160, 240)
C_PATH      = (255, 210, 60)

TEXT_MAIN   = (220, 220, 240)
TEXT_DIM    = (130, 130, 160)
TEXT_GREEN  = (80,  220, 140)
TEXT_YELLOW = (255, 210, 60)
TEXT_RED    = (230, 100, 100)

BTN_COLORS  = {
    "BFS"  : ((50,  130, 220), (70,  160, 255)),
    "DFS"  : ((180, 80,  200), (210, 110, 235)),
    "A*"   : ((40,  180, 120), (60,  220, 150)),
    "CLEAR": ((90,  90,  120), (130, 130, 165)),
    "RAND" : ((180, 130, 50),  (220, 165, 70)),
}

# ─────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────
EMPTY, BARRIER, START, END, VISITED, FRONTIER, PATH = range(7)

STATE_COLOR = {
    EMPTY   : C_EMPTY,
    BARRIER : C_BARRIER,
    START   : C_START,
    END     : C_END,
    VISITED : C_VISITED,
    FRONTIER: C_FRONTIER,
    PATH    : C_PATH,
}

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.state = EMPTY
        self.neighbors = []

    def pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.state == BARRIER

    def reset(self, keep_walls=False):
        if keep_walls and self.state == BARRIER:
            return
        if self.state not in (START, END, BARRIER) or not keep_walls:
            if self.state not in (START, END):
                self.state = EMPTY

    def update_neighbors(self, grid):
        self.neighbors = []
        r, c = self.row, self.col
        if r > 0           and not grid[r-1][c].is_barrier(): self.neighbors.append(grid[r-1][c])
        if r < GRID_ROWS-1 and not grid[r+1][c].is_barrier(): self.neighbors.append(grid[r+1][c])
        if c > 0           and not grid[r][c-1].is_barrier(): self.neighbors.append(grid[r][c-1])
        if c < GRID_COLS-1 and not grid[r][c+1].is_barrier(): self.neighbors.append(grid[r][c+1])

    def __lt__(self, other):
        return False

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return (self.row, self.col) == (other.row, other.col)


# ─────────────────────────────────────────────
#  GRID HELPERS
# ─────────────────────────────────────────────
def make_grid():
    return [[Node(r, c) for c in range(GRID_COLS)] for r in range(GRID_ROWS)]

def clear_grid(grid, full=True):
    for row in grid:
        for node in row:
            if full:
                node.state = EMPTY
            else:
                if node.state not in (START, END, BARRIER):
                    node.state = EMPTY

def random_walls(grid, density=0.28):
    import random
    clear_grid(grid, full=True)
    for row in grid:
        for node in row:
            if random.random() < density:
                node.state = BARRIER

def update_all_neighbors(grid):
    for row in grid:
        for node in row:
            node.update_neighbors(grid)

def pixel_to_cell(px, py):
    c = px // CELL_SIZE
    r = py // CELL_SIZE
    c = max(0, min(GRID_COLS - 1, c))
    r = max(0, min(GRID_ROWS - 1, r))
    return r, c


# ─────────────────────────────────────────────
#  HEURISTIC
# ─────────────────────────────────────────────
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


# ─────────────────────────────────────────────
#  ALGORITHMS  (generators for step-by-step viz)
# ─────────────────────────────────────────────
def reconstruct(came_from, current):
    path = []
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path


def algo_bfs(grid, start, end):
    queue = deque([start])
    visited = {start}
    came_from = {}
    steps = 0

    while queue:
        current = queue.popleft()
        steps += 1

        if current == end:
            path = reconstruct(came_from, end)
            for n in path:
                if n != start:
                    n.state = PATH
            yield True, steps, len(path)
            return

        for nb in current.neighbors:
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                if nb != end:
                    nb.state = FRONTIER
                queue.append(nb)

        if current != start and current != end:
            current.state = VISITED
        yield False, steps, 0

    yield False, steps, 0


def algo_dfs(grid, start, end):
    stack = [start]
    visited = {start}
    came_from = {}
    steps = 0

    while stack:
        current = stack.pop()
        steps += 1

        if current == end:
            path = reconstruct(came_from, end)
            for n in path:
                if n != start:
                    n.state = PATH
            yield True, steps, len(path)
            return

        for nb in current.neighbors:
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                if nb != end:
                    nb.state = FRONTIER
                stack.append(nb)

        if current != start and current != end:
            current.state = VISITED
        yield False, steps, 0

    yield False, steps, 0


def algo_astar(grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g = {start: 0}
    f = {start: heuristic(start.pos(), end.pos())}
    open_hash = {start}
    steps = 0

    while not open_set.empty():
        current = open_set.get()[2]
        open_hash.discard(current)
        steps += 1

        if current == end:
            path = reconstruct(came_from, end)
            for n in path:
                if n != start:
                    n.state = PATH
            yield True, steps, len(path)
            return

        for nb in current.neighbors:
            tg = g[current] + 1
            if nb not in g or tg < g[nb]:
                came_from[nb] = current
                g[nb] = tg
                f[nb] = tg + heuristic(nb.pos(), end.pos())
                if nb not in open_hash:
                    count += 1
                    open_set.put((f[nb], count, nb))
                    open_hash.add(nb)
                    if nb != end:
                        nb.state = FRONTIER

        if current != start and current != end:
            current.state = VISITED
        yield False, steps, 0

    yield False, steps, 0


# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────
def draw_grid(surf, grid):
    for row in grid:
        for node in row:
            color = STATE_COLOR[node.state]
            x = node.col * CELL_SIZE
            y = node.row * CELL_SIZE
            rect = pygame.Rect(x + 1, y + 1, CELL_SIZE - 1, CELL_SIZE - 1)
            pygame.draw.rect(surf, color, rect, border_radius=2)

    # grid lines
    for r in range(GRID_ROWS + 1):
        pygame.draw.line(surf, GRID_LINE, (0, r*CELL_SIZE), (GRID_W, r*CELL_SIZE))
    for c in range(GRID_COLS + 1):
        pygame.draw.line(surf, GRID_LINE, (c*CELL_SIZE, 0), (c*CELL_SIZE, GRID_H))


def draw_panel(surf, fonts, selected_algo, stats, mode):
    px = GRID_W
    pygame.draw.rect(surf, PANEL_BG, (px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surf, PANEL_BORDER, (px, 0), (px, WIN_H), 2)

    F_TITLE, F_BODY, F_SMALL = fonts
    x0 = px + 18
    y = 18

    # Title
    t = F_TITLE.render("PATHFINDER", True, TEXT_MAIN)
    surf.blit(t, (px + PANEL_W//2 - t.get_width()//2, y))
    y += t.get_height() + 4
    t2 = F_SMALL.render("Visualizer  •  v2.0", True, TEXT_DIM)
    surf.blit(t2, (px + PANEL_W//2 - t2.get_width()//2, y))
    y += t2.get_height() + 18

    pygame.draw.line(surf, PANEL_BORDER, (px+10, y), (px+PANEL_W-10, y))
    y += 14

    # Algo buttons
    t = F_SMALL.render("SELECT ALGORITHM", True, TEXT_DIM)
    surf.blit(t, (x0, y)); y += t.get_height() + 10

    btn_rects = {}
    algos = ["BFS", "DFS", "A*"]
    bw = (PANEL_W - 40) // 3 - 4
    for i, algo in enumerate(algos):
        bx = px + 14 + i * (bw + 8)
        by = y
        base, hover = BTN_COLORS[algo]
        color = hover if selected_algo == algo else base
        br = pygame.Rect(bx, by, bw, 38)
        pygame.draw.rect(surf, color, br, border_radius=8)
        if selected_algo == algo:
            pygame.draw.rect(surf, (255,255,255,60), br, 2, border_radius=8)
        lbl = F_BODY.render(algo, True, (255,255,255))
        surf.blit(lbl, (bx + bw//2 - lbl.get_width()//2, by + 10))
        btn_rects[algo] = br
    y += 50

    # Utility buttons
    t = F_SMALL.render("ACTIONS", True, TEXT_DIM)
    surf.blit(t, (x0, y)); y += t.get_height() + 10

    util_btns = ["CLEAR", "RAND"]
    uw = (PANEL_W - 40) // 2 - 4
    for i, lbl in enumerate(util_btns):
        bx = px + 14 + i*(uw+8)
        by = y
        base, hover = BTN_COLORS[lbl]
        br = pygame.Rect(bx, by, uw, 34)
        pygame.draw.rect(surf, base, br, border_radius=7)
        t2 = F_SMALL.render(lbl, True, (255,255,255))
        surf.blit(t2, (bx + uw//2 - t2.get_width()//2, by + 8))
        btn_rects[lbl] = br
    y += 48

    pygame.draw.line(surf, PANEL_BORDER, (px+10, y), (px+PANEL_W-10, y))
    y += 14

    # Stats
    t = F_SMALL.render("STATISTICS", True, TEXT_DIM)
    surf.blit(t, (x0, y)); y += t.get_height() + 12

    def stat_row(label, value, color=TEXT_MAIN):
        lbl = F_SMALL.render(label, True, TEXT_DIM)
        val = F_BODY.render(str(value), True, color)
        surf.blit(lbl, (x0, y))
        surf.blit(val, (px + PANEL_W - 18 - val.get_width(), y))

    algo_name, steps, path_len, found, running = stats

    stat_row("Algorithm :", algo_name or "—")
    y += 24
    stat_row("Steps     :", steps if steps else "—", TEXT_YELLOW)
    y += 24
    stat_row("Path Len  :", path_len if path_len else "—", TEXT_GREEN)
    y += 24

    status_text = "Running…" if running else ("Found ✓" if found else ("No Path ✗" if steps else "Ready"))
    status_color = TEXT_YELLOW if running else (TEXT_GREEN if found else (TEXT_RED if steps else TEXT_DIM))
    stat_row("Status    :", status_text, status_color)
    y += 36

    pygame.draw.line(surf, PANEL_BORDER, (px+10, y), (px+PANEL_W-10, y))
    y += 14

    # Legend
    t = F_SMALL.render("LEGEND", True, TEXT_DIM)
    surf.blit(t, (x0, y)); y += t.get_height() + 10

    legend = [
        (C_START,    "Start Node"),
        (C_END,      "End Node"),
        (C_BARRIER,  "Wall / Barrier"),
        (C_VISITED,  "Visited"),
        (C_FRONTIER, "Frontier"),
        (C_PATH,     "Shortest Path"),
    ]
    for color, label in legend:
        pygame.draw.rect(surf, color, (x0, y+3, 14, 14), border_radius=3)
        lt = F_SMALL.render(label, True, TEXT_MAIN)
        surf.blit(lt, (x0+22, y))
        y += 22
    y += 8

    pygame.draw.line(surf, PANEL_BORDER, (px+10, y), (px+PANEL_W-10, y))
    y += 14

    # Controls
    t = F_SMALL.render("CONTROLS", True, TEXT_DIM)
    surf.blit(t, (x0, y)); y += t.get_height() + 8

    controls = [
        ("LMB",      "Place start / end / wall"),
        ("RMB",      "Erase wall"),
        ("SPACE",    "Run selected algo"),
        ("B/D/A",    "Run BFS / DFS / A*"),
        ("C",        "Clear everything"),
        ("R",        "Random maze"),
        ("ESC",      "Quit"),
    ]
    for key, desc in controls:
        k = F_SMALL.render(f"[{key}]", True, TEXT_YELLOW)
        d = F_SMALL.render(desc, True, TEXT_DIM)
        surf.blit(k, (x0, y))
        surf.blit(d, (x0+50, y))
        y += 20

    # Mode indicator
    if mode == "place_start":
        msg = "Click to place  START"
        col = C_START
    elif mode == "place_end":
        msg = "Click to place  END"
        col = C_END
    else:
        msg = "Draw walls  |  SPACE to run"
        col = TEXT_DIM

    pygame.draw.rect(surf, (30,30,50), (px+8, WIN_H-44, PANEL_W-16, 36), border_radius=8)
    mt = F_SMALL.render(msg, True, col)
    surf.blit(mt, (px + PANEL_W//2 - mt.get_width()//2, WIN_H - 30))

    return btn_rects


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    pygame.init()
    surf = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Pathfinding Visualizer — A* | BFS | DFS")
    clock = pygame.time.Clock()

    try:
        F_TITLE = pygame.font.SysFont("segoeui",   20, bold=True)
        F_BODY  = pygame.font.SysFont("segoeui",   16, bold=True)
        F_SMALL = pygame.font.SysFont("segoeui",   13)
    except Exception:
        F_TITLE = pygame.font.SysFont(None, 22, bold=True)
        F_BODY  = pygame.font.SysFont(None, 18, bold=True)
        F_SMALL = pygame.font.SysFont(None, 14)
    fonts = (F_TITLE, F_BODY, F_SMALL)

    grid = make_grid()
    start = None
    end   = None

    selected_algo = "A*"
    mode = "place_start"   # place_start → place_end → draw_walls

    # stats: (algo_name, steps, path_len, found, running)
    stats = (None, 0, 0, False, False)

    generator = None   # active algorithm generator
    running   = False
    ALGO_SPEED = pygame.USEREVENT + 1
    pygame.time.set_timer(ALGO_SPEED, 8)   # ms per step

    draw_mode_active = False  # LMB held
    erase_mode       = False  # RMB held

    while True:
        clock.tick(FPS)

        # ── events ──────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # ── algo tick ──
            if event.type == ALGO_SPEED and running and generator:
                try:
                    found, steps, path_len = next(generator)
                    algo_name = stats[0]
                    stats = (algo_name, steps, path_len, found, True)
                    if found or (not found and path_len == 0 and steps > 0):
                        pass  # keep going until StopIteration
                except StopIteration:
                    running = False
                    algo_name, steps, path_len, found, _ = stats
                    stats = (algo_name, steps, path_len, found, False)

            # ── keyboard ──
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                if not running:
                    if event.key == pygame.K_c:
                        grid  = make_grid()
                        start = None; end = None
                        mode  = "place_start"
                        stats = (None, 0, 0, False, False)

                    if event.key == pygame.K_r:
                        random_walls(grid)
                        start = None; end = None
                        mode  = "place_start"
                        stats = (None, 0, 0, False, False)

                    if event.key in (pygame.K_b, pygame.K_d, pygame.K_a, pygame.K_SPACE):
                        if event.key == pygame.K_b: selected_algo = "BFS"
                        elif event.key == pygame.K_d: selected_algo = "DFS"
                        elif event.key == pygame.K_a: selected_algo = "A*"
                        # SPACE keeps currently selected

                        if start and end:
                            clear_grid(grid, full=False)
                            update_all_neighbors(grid)
                            if selected_algo == "BFS":
                                generator = algo_bfs(grid, start, end)
                            elif selected_algo == "DFS":
                                generator = algo_dfs(grid, start, end)
                            else:
                                generator = algo_astar(grid, start, end)
                            stats   = (selected_algo, 0, 0, False, True)
                            running = True

            # ── mouse click (buttons) ──
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx >= GRID_W:  # panel click
                    # we handle panel buttons below via btn_rects
                    pass

            # ── mouse press / drag ──
            if pygame.mouse.get_pressed()[0]:
                mx, my = pygame.mouse.get_pos()
                if mx < GRID_W and not running:
                    r, c = pixel_to_cell(mx, my)
                    node = grid[r][c]
                    if mode == "place_start":
                        if node.state == EMPTY:
                            node.state = START
                            start = node
                            mode = "place_end"
                    elif mode == "place_end":
                        if node.state == EMPTY:
                            node.state = END
                            end = node
                            mode = "draw_walls"
                    elif mode == "draw_walls":
                        if node.state == EMPTY:
                            node.state = BARRIER

            if pygame.mouse.get_pressed()[2]:  # RMB erase
                mx, my = pygame.mouse.get_pos()
                if mx < GRID_W and not running:
                    r, c = pixel_to_cell(mx, my)
                    node = grid[r][c]
                    if node.state == BARRIER:
                        node.state = EMPTY
                    elif node == start:
                        node.state = EMPTY; start = None; mode = "place_start"
                    elif node == end:
                        node.state = EMPTY; end = None; mode = "place_end"

        # ── draw ────────────────────────────────
        surf.fill(BG)
        draw_grid(surf, grid)
        btn_rects = draw_panel(surf, fonts, selected_algo, stats, mode)

        # ── panel button clicks ──
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if mx >= GRID_W and not running:
                for label, rect in btn_rects.items():
                    if rect.collidepoint(mx, my):
                        if label in ("BFS", "DFS", "A*"):
                            selected_algo = label
                        elif label == "CLEAR":
                            grid  = make_grid()
                            start = None; end = None
                            mode  = "place_start"
                            stats = (None, 0, 0, False, False)
                        elif label == "RAND":
                            random_walls(grid)
                            start = None; end = None
                            mode  = "place_start"
                            stats = (None, 0, 0, False, False)

        pygame.display.flip()


if __name__ == "__main__":
    main()
import heapq
import random
import tkinter as tk

# SETTINGS

BOARD_SIZE = 9
CELL = 55
GAP = 10
STEP = CELL + GAP
BOARD_PX = BOARD_SIZE * CELL + (BOARD_SIZE - 1) * GAP
START_WALLS = 10

# True  = official rule: if the opponent is directly in front, you jump over
# False = the opponent is simply an obstacle, no jumping at all.
ENABLE_JUMP = True

# AI strength / personality
# Pick "easy", "medium" or "hard".
DIFFICULTY = "medium"

PRESETS = {
    #           search depth, wall value, wall candidates, mistake chance, AI walls
    "easy":   dict(depth=1, wall_value=6, wall_limit=3, noise=0.40, ai_walls=5),
    "medium": dict(depth=2, wall_value=4, wall_limit=5, noise=0.25, ai_walls=8),
    "hard":   dict(depth=2, wall_value=3, wall_limit=6, noise=0.05, ai_walls=10),
}
_p = PRESETS[DIFFICULTY]

AI_DEPTH = _p["depth"]            # plies searched (AI action + player reply)
WALL_VALUE = _p["wall_value"]     # higher = AI is stingier with walls
ROOT_WALL_LIMIT = _p["wall_limit"]  # wall candidates the AI considers on its turn
REPLY_WALL_LIMIT = 3              # wall candidates it assumes the player considers
HUMAN_NOISE = _p["noise"]         # chance of a "nearly as good" action instead of the best
HUMAN_MARGIN = 4                  # "nearly as good" = within this many points (< 1 square)
AI_START_WALLS = _p["ai_walls"]

GOAL_ROW = {"player": 0, "ai": 8}

# GAME STATE

def other(who):
    return "ai" if who == "player" else "player"


class State:
    """Everything the rules need. Cheap to copy, so minimax can try moves."""

    def __init__(self):
        self.pos = {"player": (8, 4), "ai": (0, 4)}
        self.walls_left = {"player": START_WALLS, "ai": AI_START_WALLS}
        self.walls = set()          # entries look like ("H", row, col) / ("V", row, col)

    def copy(self):
        new = State.__new__(State)
        new.pos = dict(self.pos)
        new.walls_left = dict(self.walls_left)
        new.walls = set(self.walls)
        return new

    def apply(self, who, action):
        kind, value = action
        if kind == "move":
            self.pos[who] = value
        else:
            self.walls.add(value)
            self.walls_left[who] -= 1

    def winner(self):
        if self.pos["player"][0] == GOAL_ROW["player"]:
            return "player"
        if self.pos["ai"][0] == GOAL_ROW["ai"]:
            return "ai"
        return None


# RULES: movement
# Wall ("H", r, c) lies under row r and covers columns c and c+1.
# Wall ("V", r, c) lies right of column c and covers rows r and r+1.

def on_board(cell):
    return 0 <= cell[0] < BOARD_SIZE and 0 <= cell[1] < BOARD_SIZE


def blocked(walls, a, b):
    """Is there a wall between two ADJACENT cells a and b?"""
    (r1, c1), (r2, c2) = a, b
    if r1 != r2:                                   # vertical step
        r = min(r1, r2)
        return ("H", r, c1) in walls or ("H", r, c1 - 1) in walls
    c = min(c1, c2)                                # horizontal step
    return ("V", r1, c) in walls or ("V", r1 - 1, c) in walls


def neighbors(walls, cell):
    """Adjacent cells reachable without crossing a wall (pawns ignored)."""
    r, c = cell
    result = []
    for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if on_board(nxt) and not blocked(walls, cell, nxt):
            result.append(nxt)
    return result


def pawn_moves(state, who):
    """All legal pawn moves for `who`, including jumps."""
    me = state.pos[who]
    opp = state.pos[other(who)]
    walls = state.walls
    moves = []

    for nxt in neighbors(walls, me):
        if nxt != opp:
            moves.append(nxt)
            continue

        if not ENABLE_JUMP:
            continue

        # Opponent is directly in front -> jump.
        dr, dc = nxt[0] - me[0], nxt[1] - me[1]
        far = (nxt[0] + dr, nxt[1] + dc)

        if on_board(far) and not blocked(walls, nxt, far):
            moves.append(far)                      # straight jump
        else:                                      # straight jump blocked -> diagonals
            for side in ((nxt[0] + dc, nxt[1] + dr), (nxt[0] - dc, nxt[1] - dr)):
                if on_board(side) and not blocked(walls, nxt, side):
                    moves.append(side)
    return moves

# A* PATHFINDING
# f(n) = g(n) + h(n)
#   g = steps taken so far
#   h = |row - goal_row|   (never overestimates, so A* is optimal)
# Pawns are ignored here: the rules say a wall may not cut off a goal,
# but a pawn standing in the way does not count as a block.

def a_star(walls, start, goal_row):
    def h(cell):
        return abs(cell[0] - goal_row)

    heap = [(h(start), 0, start)]
    cost = {start: 0}
    came_from = {start: None}

    while heap:
        _, g, cell = heapq.heappop(heap)
        if g > cost[cell]:
            continue                               

        if cell[0] == goal_row:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1]

        for nxt in neighbors(walls, cell):
            new_g = g + 1
            if nxt not in cost or new_g < cost[nxt]:
                cost[nxt] = new_g
                came_from[nxt] = cell
                heapq.heappush(heap, (new_g + h(nxt), new_g, nxt))
    return None


def distance(walls, cell, goal_row):
    path = a_star(walls, cell, goal_row)
    return 99 if path is None else len(path) - 1

# RULES: walls

def wall_fits(state, wall):
    """Board bounds, no duplicate, no overlap, no crossing."""
    d, r, c = wall
    if not (0 <= r < 8 and 0 <= c < 8):
        return False
    w = state.walls
    if wall in w:
        return False
    if d == "H":
        return not (("H", r, c - 1) in w or ("H", r, c + 1) in w or ("V", r, c) in w)
    return not (("V", r - 1, c) in w or ("V", r + 1, c) in w or ("H", r, c) in w)


def wall_is_legal(state, wall):
    """Fits on the board AND leaves both pawns a path to their goal."""
    if not wall_fits(state, wall):
        return False
    state.walls.add(wall)
    ok = all(a_star(state.walls, state.pos[w], GOAL_ROW[w]) for w in ("player", "ai"))
    state.walls.discard(wall)
    return ok

# AI: HEURISTIC WALL CANDIDATES
# Instead of trying all 128 walls, only try walls that cut the
# opponent's current shortest path, then keep the best few.

def walls_blocking_path(path):
    found = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 != r2:                                   # vertical step -> horizontal wall
            r = min(r1, r2)
            found += [("H", r, c1), ("H", r, c1 - 1)]
        else:                                          # horizontal step -> vertical wall
            c = min(c1, c2)
            found += [("V", r1, c), ("V", r1 - 1, c)]
    return list(dict.fromkeys(found))                  # remove duplicates, keep order


def good_walls(state, who, limit):
    """
    Heuristic score of a wall = (how much longer the opponent's route gets)
                              - (how much longer MY route gets)
    Only walls with score >= 1 are returned, best first.
    """
    foe = other(who)
    foe_path = a_star(state.walls, state.pos[foe], GOAL_ROW[foe])
    foe_before = len(foe_path) - 1
    me_before = distance(state.walls, state.pos[who], GOAL_ROW[who])

    scored = []
    for wall in walls_blocking_path(foe_path):
        if not wall_fits(state, wall):
            continue
        state.walls.add(wall)
        foe_after = a_star(state.walls, state.pos[foe], GOAL_ROW[foe])
        me_after = a_star(state.walls, state.pos[who], GOAL_ROW[who])
        state.walls.discard(wall)

        if foe_after is None or me_after is None:      # illegal: blocks a goal
            continue

        gain = (len(foe_after) - 1 - foe_before) - (len(me_after) - 1 - me_before)
        if gain >= 1:
            scored.append((gain, wall))

    scored.sort(reverse=True)
    return [wall for _, wall in scored[:limit]]


def candidate_actions(state, who, wall_limit):
    actions = [("move", cell) for cell in pawn_moves(state, who)]
    if state.walls_left[who] > 0:
        actions += [("wall", w) for w in good_walls(state, who, wall_limit)]
    return actions


# AI: MINIMAX WITH ALPHA-BETA

# evaluate() > 0  means the AI is ahead.
#
#   10 * (player distance - AI distance)  +  WALL_VALUE * (AI walls - player walls)
#
# Worked example: AI is 8 steps away, player 6 steps away, 10 walls each:
#   10 * (6 - 8) + 3 * 0 = -20      (AI is behind -> walls look attractive)
# A wall that makes the player's route +3 changes it to
#   10 * (9 - 8) + 3 * (9 - 10) = +7
# A plain move gives 10 * (6 - 7) = -10.  The wall wins by a lot.
# A wall that only gains +1 gives 10*(7-8) + 3*(-1) = -13, worse than
# moving (-10), so the AI keeps its wall.  That is the "when is a wall
# worth it" decision.

def evaluate(state):
    ai_d = distance(state.walls, state.pos["ai"], GOAL_ROW["ai"])
    pl_d = distance(state.walls, state.pos["player"], GOAL_ROW["player"])
    wall_diff = state.walls_left["ai"] - state.walls_left["player"]
    return 10 * (pl_d - ai_d) + WALL_VALUE * wall_diff


def minimax(state, depth, alpha, beta, who):
    win = state.winner()
    if win == "ai":
        return 1000 + depth            # faster wins score higher
    if win == "player":
        return -1000 - depth
    if depth == 0:
        return evaluate(state)

    actions = candidate_actions(state, who, REPLY_WALL_LIMIT)
    if not actions:
        return evaluate(state)

    if who == "ai":                    # AI maximises
        best = -10_000
        for action in actions:
            child = state.copy()
            child.apply(who, action)
            best = max(best, minimax(child, depth - 1, alpha, beta, "player"))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    best = 10_000                      # player minimises
    for action in actions:
        child = state.copy()
        child.apply(who, action)
        best = min(best, minimax(child, depth - 1, alpha, beta, "ai"))
        beta = min(beta, best)
        if alpha >= beta:
            break
    return best


def choose_ai_action(state, recent_cells):
    actions = candidate_actions(state, "ai", ROOT_WALL_LIMIT)
    if not actions:
        return None

    scored = []
    for action in actions:
        child = state.copy()
        child.apply("ai", action)
        score = minimax(child, AI_DEPTH - 1, -10_000, 10_000, "player")

        # discourage walking back to a square it just left
        if action[0] == "move" and action[1] in recent_cells:
            score -= 2
        scored.append((score, action))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_action = scored[0]

    if best_score >= 1000:             # a forced win: always take it
        return best_action

    # Human-like: sometimes pick another action that is almost as good.
    # It never picks a clearly worse one, so there are no silly blunders.
    if random.random() < HUMAN_NOISE:
        near = [a for s, a in scored if best_score - s <= HUMAN_MARGIN]
        return random.choice(near)
    return best_action


# MOUSE HELPERS

def cell_at(x, y):
    col, ox = divmod(x, STEP)
    row, oy = divmod(y, STEP)
    if ox < CELL and oy < CELL and row < BOARD_SIZE and col < BOARD_SIZE:
        return (row, col)
    return None


def groove_at(x, y):
    """Nearest groove to the mouse: the wall that starts at this cell."""
    col, ox = divmod(x, STEP)
    row, oy = divmod(y, STEP)
    if not (0 <= row < 8 and 0 <= col < 8):
        return None
    mid = CELL + GAP / 2
    if abs(oy - mid) <= abs(ox - mid):
        return ("H", row, col)
    return ("V", row, col)


# WINDOW / DRAWING

class QuoridorApp:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Quoridor AI")
        self.window.resizable(False, False)

        self.canvas = tk.Canvas(self.window, width=BOARD_PX, height=BOARD_PX,
                                bg="white", highlightthickness=0)
        self.canvas.pack(padx=20, pady=10)

        self.info = tk.Label(self.window, font=("Arial", 12))
        self.info.pack(pady=5)
        self.counts = tk.Label(self.window, font=("Arial", 12))
        self.counts.pack(pady=5)

        buttons = tk.Frame(self.window)
        buttons.pack(pady=10)
        tk.Button(buttons, text="Wall mode (W)", font=("Arial", 11), takefocus=0,
                  command=self.toggle_wall_mode).pack(side="left", padx=5)
        tk.Button(buttons, text="Restart Game", font=("Arial", 11), takefocus=0,
                  command=self.new_game).pack(side="left", padx=5)

        self.window.bind("<Key>", self.on_key)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        self.ai_job = None
        self.new_game()
        self.window.mainloop()

    # game flow

    def new_game(self):
        if self.ai_job is not None:                 # stop a pending AI move
            self.window.after_cancel(self.ai_job)
            self.ai_job = None
        self.state = State()
        self.turn = "player"
        self.wall_mode = False
        self.hover = None
        self.ai_recent = []
        self.redraw()

    def end_player_turn(self):
        self.hover = None
        if self.state.winner():
            self.redraw()
            return
        self.turn = "ai"
        self.redraw()
        self.ai_job = self.window.after(300, self.ai_turn)

    def ai_turn(self):
        self.ai_job = None
        if self.turn != "ai" or self.state.winner():
            return

        action = choose_ai_action(self.state, self.ai_recent)
        if action is not None:
            if action[0] == "move":
                self.ai_recent = (self.ai_recent + [self.state.pos["ai"]])[-3:]
            self.state.apply("ai", action)

        self.turn = "player"
        message = None

        # Extremely rare: player cannot move and has no walls -> skip the turn
        if (not self.state.winner()
                and not pawn_moves(self.state, "player")
                and self.state.walls_left["player"] == 0):
            self.turn = "ai"
            self.ai_job = self.window.after(800, self.ai_turn)
            message = "You have no legal move - turn skipped"

        self.redraw(message)

    # input 

    def show(self, text):
        self.info.config(text=text)

    def toggle_wall_mode(self):
        if self.turn != "player" or self.state.winner():
            return
        if self.state.walls_left["player"] == 0:
            self.show("You have no walls left. You must move.")
            return
        self.wall_mode = not self.wall_mode
        self.hover = None
        self.redraw()

    def try_move(self, cell):
        if cell not in pawn_moves(self.state, "player"):
            self.show("Illegal move! A wall, the edge, or the opponent is in the way.")
            return
        self.state.apply("player", ("move", cell))
        self.end_player_turn()

    def on_key(self, event):
        if self.turn != "player" or self.state.winner():
            return

        key = event.keysym
        if key.lower() == "w":
            self.toggle_wall_mode()
        elif key == "Escape" and self.wall_mode:
            self.wall_mode = False
            self.hover = None
            self.redraw()
        elif key in ("Up", "Down", "Left", "Right") and not self.wall_mode:
            dr, dc = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}[key]
            r, c = self.state.pos["player"]
            target = (r + dr, c + dc)
            if target == self.state.pos["ai"]:      # opponent in front -> jump square
                target = (r + 2 * dr, c + 2 * dc)
            self.try_move(target)

    def on_click(self, event):
        if self.turn != "player" or self.state.winner():
            return

        if self.wall_mode:
            wall = groove_at(event.x, event.y)
            if wall is None:
                return
            if not wall_is_legal(self.state, wall):
                self.show("Illegal wall (overlap, crossing, or it blocks every path).")
                return
            self.state.apply("player", ("wall", wall))
            self.wall_mode = False
            self.end_player_turn()
        else:
            cell = cell_at(event.x, event.y)
            if cell is not None:
                self.try_move(cell)

    def on_motion(self, event):
        if self.wall_mode and self.turn == "player":
            wall = groove_at(event.x, event.y)
            if wall != self.hover:
                self.hover = wall
                self.redraw()

    # drawing

    def draw_wall(self, wall, color):
        d, r, c = wall
        if d == "H":
            x0 = c * STEP
            y = r * STEP + CELL + GAP / 2
            self.canvas.create_rectangle(x0, y - 4, x0 + 2 * CELL + GAP, y + 4,
                                         fill=color, outline=color)
        else:
            x = c * STEP + CELL + GAP / 2
            y0 = r * STEP
            self.canvas.create_rectangle(x - 4, y0, x + 4, y0 + 2 * CELL + GAP,
                                         fill=color, outline=color)

    def draw_pawn(self, cell, color):
        x, y = cell[1] * STEP, cell[0] * STEP
        self.canvas.create_oval(x + 10, y + 10, x + CELL - 10, y + CELL - 10,
                                fill=color, outline="black")

    def redraw(self, message=None):
        state = self.state
        self.canvas.delete("all")

        highlight = []
        if self.turn == "player" and not self.wall_mode and not state.winner():
            highlight = pawn_moves(state, "player")

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = "#b7e4a7" if (r, c) in highlight else "lightgray"
                x, y = c * STEP, r * STEP
                self.canvas.create_rectangle(x, y, x + CELL, y + CELL,
                                             fill=color, outline="black")

        for wall in state.walls:
            self.draw_wall(wall, "black")

        if self.wall_mode and self.hover is not None:
            ok = wall_is_legal(state, self.hover)
            self.draw_wall(self.hover, "#4caf50" if ok else "#e53935")

        self.draw_pawn(state.pos["player"], "blue")
        self.draw_pawn(state.pos["ai"], "red")

        self.counts.config(text=f"You: {state.walls_left['player']} walls     "
                                f"AI: {state.walls_left['ai']} walls")

        if message:
            text = message
        elif state.winner() == "player":
            text = "YOU WIN!"
        elif state.winner() == "ai":
            text = "AI WINS!"
        elif self.turn == "ai":
            text = "AI is thinking..."
        elif self.wall_mode:
            text = "WALL MODE - click near a groove (Esc to cancel)"
        else:
            text = "Your turn - click a green square or use arrow keys | W = wall"
        self.info.config(text=text)


if __name__ == "__main__":
    QuoridorApp()
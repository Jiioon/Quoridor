import heapq
import random


# ============================================================
# SETTINGS
# ============================================================

BOARD_SIZE = 9
CELL = 55
GAP = 10
STEP = CELL + GAP
BOARD_PX = BOARD_SIZE * CELL + (BOARD_SIZE - 1) * GAP
START_WALLS = 10

# True  = official rule: if the opponent is directly in front, you jump over
#         them (this is the "moves 2 squares" behaviour, and it is correct).
# False = the opponent is simply an obstacle, no jumping at all.
ENABLE_JUMP = True

# ---- AI strength / personality --------------------------------
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


# ============================================================
# GAME STATE
# ============================================================

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


# ============================================================
# RULES: movement
# ============================================================
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


# ============================================================
# A* PATHFINDING
# ============================================================
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
            continue                               # outdated heap entry

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


# ============================================================
# RULES: walls
# ============================================================

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


# ============================================================
# AI: HEURISTIC WALL CANDIDATES
# ============================================================
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


# ============================================================
# AI: MINIMAX WITH ALPHA-BETA
# ============================================================
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




import streamlit as st

st.set_page_config(page_title="Quoridor vs AI", page_icon="♟️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1.5rem;}
div[data-testid="stButton"] > button { min-height: 42px; font-size: 18px; }
div[data-testid="stButton"] > button:disabled { opacity: 0.8; }
</style>
""", unsafe_allow_html=True)


def configure_difficulty(name):
    """Update AI settings to match the chosen web-game difficulty."""
    global DIFFICULTY, AI_DEPTH, WALL_VALUE, ROOT_WALL_LIMIT, HUMAN_NOISE, AI_START_WALLS
    DIFFICULTY = name
    preset = PRESETS[name]
    AI_DEPTH = preset["depth"]
    WALL_VALUE = preset["wall_value"]
    ROOT_WALL_LIMIT = preset["wall_limit"]
    HUMAN_NOISE = preset["noise"]
    AI_START_WALLS = preset["ai_walls"]


def reset_game():
    st.session_state.game_state = State()
    st.session_state.turn = "player"
    st.session_state.ai_recent = []
    st.session_state.notice = "New game started. Your turn!"


def do_ai_turn():
    state = st.session_state.game_state
    if state.winner():
        st.session_state.turn = "player"
        return
    st.session_state.turn = "ai"
    with st.spinner("AI is thinking..."):
        action = choose_ai_action(state, st.session_state.ai_recent)
        if action is not None:
            if action[0] == "move":
                st.session_state.ai_recent = (st.session_state.ai_recent + [state.pos["ai"]])[-3:]
            state.apply("ai", action)
    st.session_state.turn = "player"
    if state.winner() == "ai":
        st.session_state.notice = "The AI reached its goal. Try again!"
    else:
        st.session_state.notice = "Your turn. Choose a highlighted square or place a wall."


def finish_player_action(action):
    state = st.session_state.game_state
    state.apply("player", action)
    if state.winner() == "player":
        st.session_state.turn = "player"
        st.session_state.notice = "You win! Press Restart Game to play again."
    else:
        do_ai_turn()
    st.rerun()


# Sidebar: choose AI strength. Changing difficulty begins a fresh game.
st.sidebar.title("Game settings")
selected_difficulty = st.sidebar.selectbox(
    "Game mode", options=["easy", "medium", "hard"], index=1,
    format_func=lambda x: {"easy": "Easy", "medium": "Medium", "hard": "Hard"}[x],
    key="difficulty_selector",
)
configure_difficulty(selected_difficulty)

if "active_difficulty" not in st.session_state:
    st.session_state.active_difficulty = selected_difficulty
    reset_game()
elif st.session_state.active_difficulty != selected_difficulty:
    st.session_state.active_difficulty = selected_difficulty
    reset_game()

if "game_state" not in st.session_state:
    reset_game()

st.title("♟️ Quoridor: You vs AI")
st.caption("Browser version of your Python game — no desktop installation required.")

state = st.session_state.game_state
winner = state.winner()

col_board, col_controls = st.columns([3, 2], gap="large")
with col_board:
    st.subheader("Game board")
    st.markdown("🔵 **You** · 🔴 **AI** · 🟢 **Legal move** · ━ horizontal wall · ┃ vertical wall")
    legal = set(pawn_moves(state, "player")) if st.session_state.turn == "player" and winner is None else set()

    for r in range(BOARD_SIZE):
        row_cols = st.columns(BOARD_SIZE, gap="small")
        for c in range(BOARD_SIZE):
            cell = (r, c)
            markers = []
            # Show wall markers in the two cells covered by each wall.
            for d, wr, wc in state.walls:
                if d == "H" and r == wr and c in (wc, wc + 1):
                    markers.append("━")
                elif d == "V" and c == wc and r in (wr, wr + 1):
                    markers.append("┃")
            if cell == state.pos["player"]:
                label = "🔵" + (" " + "".join(markers) if markers else "")
            elif cell == state.pos["ai"]:
                label = "🔴" + (" " + "".join(markers) if markers else "")
            elif cell in legal:
                label = "🟢" + (" " + "".join(markers) if markers else "")
            elif markers:
                label = "".join(markers)
            else:
                label = "·"

            clicked = row_cols[c].button(
                label, key=f"cell_{r}_{c}",
                type="primary" if cell in legal else "secondary",
                disabled=(cell not in legal),
                use_container_width=True,
                help=f"Row {r + 1}, column {c + 1}" if cell in legal else None,
            )
            if clicked:
                finish_player_action(("move", cell))

with col_controls:
    st.subheader("Game status")
    st.markdown(f"**Mode:** {selected_difficulty.title()}")
    if winner == "player":
        st.success("YOU WIN! 🎉")
    elif winner == "ai":
        st.error("AI WINS! 🤖")
    elif st.session_state.turn == "ai":
        st.info("AI is thinking...")
    else:
        st.info(st.session_state.get("notice", "Your turn"))

    m1, m2 = st.columns(2)
    m1.metric("Your walls", state.walls_left["player"])
    m2.metric("AI walls", state.walls_left["ai"])

    st.divider()
    st.subheader("Place a wall")
    st.caption("Choose its direction and the starting row/column. A wall covers two spaces between cells.")
    direction_label = st.radio("Orientation", ["Horizontal", "Vertical"], horizontal=True, key="wall_orientation")
    wall_row = st.selectbox("Starting row (1–8)", list(range(1, 9)), index=3, key="wall_row")
    wall_col = st.selectbox("Starting column (1–8)", list(range(1, 9)), index=3, key="wall_col")
    wall = ("H" if direction_label == "Horizontal" else "V", wall_row - 1, wall_col - 1)

    if st.button("Place Wall", use_container_width=True, disabled=(winner is not None or st.session_state.turn != "player" or state.walls_left["player"] <= 0)):
        if wall_is_legal(state, wall):
            finish_player_action(("wall", wall))
        else:
            st.error("Illegal wall: it overlaps/crosses another wall or blocks every path to a goal.")

    if st.button("Restart Game", use_container_width=True):
        reset_game()
        st.rerun()

    with st.expander("Walls already placed"):
        if not state.walls:
            st.write("No walls placed yet.")
        else:
            readable = []
            for d, r, c in sorted(state.walls):
                if d == "H":
                    readable.append(f"Horizontal — between rows {r + 1}–{r + 2}, columns {c + 1}–{c + 2}")
                else:
                    readable.append(f"Vertical — between columns {c + 1}–{c + 2}, rows {r + 1}–{r + 2}")
            for item in readable:
                st.write("• " + item)

st.divider()
st.caption("Rules: click a green square to move. Jumps over an adjacent opponent are allowed when legal. Walls cannot overlap, cross, or block every route to a goal.")

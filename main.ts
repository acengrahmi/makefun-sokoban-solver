// Sokoban solver: push-optimal A* per room, stitched by the player's walk.
//
// Strategy. Every level this generator draws is a hallway with 1-4 rooms off
// it; a box pushed out of its room can never come back, so each room is an
// independent problem and the level's push optimum P* is the sum of the rooms'
// optima. For each room we run a push-optimal A* over push-states (boxes +
// the player's reachable region, canonicalised by the region's lowest square):
//   - heuristic: exact minimum-cost matching of boxes to goals on shortest
//     path distances that ignore other boxes (admissible: one push moves one
//     box one square), computed by DP over goal subsets;
//   - pruning: dead squares (pull-back from the goals), frozen 2x2 blocks of
//     walls/boxes, and boxes frozen flat against a wall;
//   - deterministic: fixed direction order, fixed tie-breaks, budgets counted
//     in expanded states, never in wall-clock time, so two runs write the
//     same bytes.
// If a room's search exhausts its budget we fall back to a weighted A* for
// that room (still a valid solution, just possibly over P*), and if room
// decomposition fails we fall back to the same search over the whole level.
// Every instance always gets a solution.txt written, even if empty.
//
// Node 22 runs this file as it is (plain JavaScript, no TypeScript-only
// syntax). Only node:fs / node:path are used. No network, no clock, no
// randomness.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: up, down, left, right. The order is the only
// tie-break the search uses, so it must never change.
const DIRS = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
];
const INF = 1e9;

/** Parse XSB text into a level: width, walls, goals, boxes, player. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Live squares: every square from which a box alone (other boxes ignored)
 * can be pushed onto some goal. Computed by pulling back from the goals: a
 * square y is live when some live square x = y + d has the square y - d open,
 * i.e. a box on y can be pushed in direction d onto x's pull-back chain.
 * A box pushed onto a non-live square is dead, so such pushes are pruned.
 */
function liveSquares(level) {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const d of DIRS) {
      const step = d.dy * w + d.dx;
      const y = x - step;
      if (!wall[y] && !wall[y - step] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Shortest walking distances from every goal to every square, ignoring boxes
 * (BFS per goal). dist[g][s] is the fewest steps from s to goal g; -1 means
 * unreachable. Used by the matching heuristic and by walkTo.
 */
function goalDistances(level) {
  const { w, wall, goals } = level;
  const dists = [];
  for (const g of goals) {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi];
      for (const d of DIRS) {
        const q = p + d.dy * w + d.dx;
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p] + 1;
          queue.push(q);
        }
      }
    }
    dists.push(dist);
  }
  return dists;
}

/**
 * The player's reachable region from `from` with `occupied` in the way.
 * Marks `seen` with `stamp`, returns the region's lowest square, which names
 * the region: any player square inside the same region reaches the same
 * pushes, so the lowest square is the state's canonical player.
 */
function regionLow(level, occupied, from, seen, stamp, stack) {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Shortest walk from `from` to `to` around walls and boxes (BFS, fixed order). */
function walkTo(level, occupied, from, to) {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * Whether the box just pushed to `t` is frozen in a 2x2 block of walls and
 * boxes holding a box off its goal: no push can ever move any box of the
 * block again, so the state is dead.
 */
function frozenBlock(level, occupied, goal, t) {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) ||
        !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) ||
        (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * Whether the box at `b` (off its goal) is frozen flat against a wall: a
 * wall on one side, and a wall or another box on both sides along that wall,
 * so it can move neither along the wall nor away from it.
 */
function frozenAgainstWall(level, occupied, goal, b) {
  const { w, wall } = level;
  if (goal[b]) return false;
  const x = b % w;
  // Wall above/below blocks vertical movement; check the row neighbours.
  if (wall[b - w] && (wall[b - 1] || occupied[b - 1]) && (wall[b + 1] || occupied[b + 1])) return true;
  if (wall[b + w] && (wall[b - 1] || occupied[b - 1]) && (wall[b + 1] || occupied[b + 1])) return true;
  // Wall left/right blocks horizontal movement; check the column neighbours.
  if (wall[b - 1] && (wall[b - w] || occupied[b - w]) && (wall[b + w] || occupied[b + w])) return true;
  if (wall[b + 1] && (wall[b - w] || occupied[b - w]) && (wall[b + w] || occupied[b + w])) return true;
  return false;
}

/**
 * Exact minimum-cost assignment of boxes to goals on the precomputed
 * distances (DP over goal subsets; at most 4 boxes here). Returns INF when
 * no assignment exists (then the state is unsolvable and pruned). Admissible:
 * one push moves exactly one box one square, so the matched distance sum is
 * a lower bound on the pushes still needed.
 */
function matchingHeuristic(boxes, goals, dists) {
  const n = boxes.length;
  const m = goals.length;
  if (n === 0) return 0;
  const size = 1 << m;
  const dp = new Float64Array(size).fill(INF);
  dp[0] = 0;
  for (let mask = 0; mask < size; mask++) {
    const k = popcount(mask); // boxes[0..k) already assigned
    if (k >= n || dp[mask] >= INF) continue;
    const b = boxes[k];
    for (let j = 0; j < m; j++) {
      if (mask & (1 << j)) continue;
      const d = dists[j][b];
      if (d < 0) continue;
      const nm = mask | (1 << j);
      const v = dp[mask] + d;
      if (v < dp[nm]) dp[nm] = v;
    }
  }
  let best = INF;
  for (let mask = 0; mask < size; mask++) {
    if (popcount(mask) === n && dp[mask] < best) best = dp[mask];
  }
  return best;
}

function popcount(x) {
  let c = 0;
  while (x) { x &= x - 1; c++; }
  return c;
}

/** A state's key: the region's lowest square, then the boxes' squares. */
function stateKey(boxes, low) {
  return low + ':' + boxes.join(',');
}

/**
 * Push-optimal A* for one (sub-)level: { pushes, moves } or null when the
 * search gives up (budget) or the level has no solution.
 *
 * A state is the boxes (sorted) plus the player's reachable region, named by
 * its lowest square. Expanding a state tries every push the player can make
 * from anywhere in its region. g is pushes made, h the matching heuristic,
 * f = g + weight * h (weight 1 is optimal; a larger weight trades optimality
 * for speed and is only used as a fallback). The heap orders by (f, -g, seq)
 * so ties break towards deeper states first, then by discovery order: fully
 * deterministic. `budget` counts expansions, never time.
 */
function solvePushAStar(level, opts) {
  const { maxExpansions, weight } = opts;
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dists = goalDistances(level);
  const solved = (boxes) => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // The states in discovery order: boxes, player square after the push that
  // made them, g (pushes), parent, and the push (box from -> to) that made
  // them, so the move string rebuilds by walking between pushes.
  const stBoxes = [level.boxes.slice()];
  const stPlayer = [level.player];
  const stG = [0];
  const stParent = [-1];
  const stPush = [null];
  const bestG = new Map();
  const h0 = matchingHeuristic(level.boxes, goals, dists);
  if (h0 >= INF) return null;
  const low0 = (() => {
    for (const b of level.boxes) occupied[b] = 1;
    const low = regionLow(level, occupied, level.player, seen, ++stamp, stack);
    for (const b of level.boxes) occupied[b] = 0;
    return low;
  })();
  bestG.set(stateKey(level.boxes, low0), 0);

  // Binary heap of [f, -g, seq, stateIndex].
  const heap = [];
  let seq = 0;
  const heapLess = (a, b) =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1], heap[c])) c++;
        if (!heapLess(heap[c], heap[i])) break;
        const t = heap[i]; heap[i] = heap[c]; heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, 0, seq++, 0]);

  let expanded = 0;
  let found = solved(level.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const [, negG, , s] = heapPop();
    const g = -negG;
    const boxes = stBoxes[s];
    for (const b of boxes) occupied[b] = 1;
    // The state's key: recompute the player's region from the square the
    // player stands on; a heap entry whose g is no longer the best recorded
    // for its key is stale (a cheaper path to the same state was found).
    const low = regionLow(level, occupied, stPlayer[s], seen, ++stamp, stack);
    if (bestG.get(stateKey(boxes, low)) < g) {
      for (const b of boxes) occupied[b] = 0;
      continue;
    }
    const here = stamp; // the region's marks, reused for the push checks
    expanded++;
    if (expanded > maxExpansions) {
      for (const b of boxes) occupied[b] = 0;
      return null;
    }
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const d of DIRS) {
        const step = d.dy * w + d.dx;
        const t = b + step;
        // The player must stand on b - step (inside its region), the target
        // square must be open, live, and the push must not freeze a box.
        if (seen[b - step] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t) ||
          frozenAgainstWall(level, occupied, goal, t);
        let key = null;
        let next = null;
        let h = INF;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          const low = regionLow(level, occupied, b, mark, ++stamp, stack);
          key = stateKey(next, low);
          const prevG = bestG.get(key);
          if (prevG === undefined || g + 1 < prevG) {
            h = matchingHeuristic(next, goals, dists);
          } else {
            key = null; // already reached at least as cheaply
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || key === null || h >= INF) continue;
        bestG.set(key, g + 1);
        const idx = stBoxes.length;
        stBoxes.push(next);
        stPlayer.push(b);
        stG.push(g + 1);
        stParent.push(s);
        stPush.push({ from: b, to: t, dir: d });
        if (solved(next)) {
          found = idx;
          break;
        }
        heapPush([g + 1 + weight * h, -(g + 1), seq++, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the move string: walk to each push's square, then push.
  const path = [];
  for (let s = found; s > 0; s = stParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const push = stPush[s];
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, push.from - (push.to - push.from));
    for (const b of boxes) occupied[b] = 0;
    moves += push.dir.push;
    player = push.from;
    const at = boxes.indexOf(push.from);
    boxes[at] = push.to;
    boxes.sort((a, b) => a - b);
  }
  return { pushes: path.length, moves };
}

/**
 * Split a level into rooms: cut the floor squares of the player's row that
 * no box can ever be pushed from onto a goal (the hallway: a box there
 * reaches no goal), then take the connected components of what is left.
 * Every component holding boxes must hold as many goals as boxes, and no
 * box may start in the cut. Returns null when the level is no set of rooms.
 */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const d of DIRS) {
        const q = squares[i] + d.dy * w + d.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The player square after playing `moves` (case aside) from `from`. */
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * Solve a level room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. A box
 * never leaves its room, so the rooms' optima add up to the level's.
 * Falls back to weighted A* per room when the optimal search gives up, and
 * to the whole-level search when the level is no set of rooms.
 * Returns { pushes, moves } or null.
 */
function solveLevel(level, budget) {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushAStar(level, { maxExpansions: budget.perSearch, weight: 1 });
    if (whole !== null) return whole;
    return solvePushAStar(level, { maxExpansions: budget.perSearch, weight: 2 });
  }
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub = {
      w: level.w, h: level.h, wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved = solvePushAStar(sub, { maxExpansions: budget.perSearch, weight: 1 });
    if (solved === null) {
      // Optimal search gave up: take a weighted (greedy) solution for this
      // room rather than no solution at all.
      solved = solvePushAStar(sub, { maxExpansions: budget.perSearch, weight: 3 });
    }
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/**
 * Solve one level's text, never throwing: an empty string is a valid answer
 * worth no points, and is what any failure writes.
 */
function solveText(text) {
  try {
    const level = parseLevel(text);
    if (level.player < 0 || level.boxes.length === 0) return '';
    if (level.boxes.length !== level.goals.length) return '';
    const solved = solveLevel(level, { perSearch: 400_000 });
    return solved === null ? '' : solved.moves;
  } catch {
    return '';
  }
}

// The set: every instance directory under /input, in sorted order; each
// answer written as /output/<id>/solution.txt, the directory created.
for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveText(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}



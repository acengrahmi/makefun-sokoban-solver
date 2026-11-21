# Sokoban Solver — make.fun entry

Push-optimal Sokoban solver for `sokoban-solver-01-run-20261121` ($100).

**Approach:** room decomposition (boxes can never leave their room, so the
level optimum is the sum of room optima) + push-optimal A* per room with an
admissible min-cost box→goal matching heuristic, dead-square / frozen-2x2 /
freeze-against-wall pruning, stitched together by the player's walk through
the hallway. Weighted-A* fallback if a room exhausts its budget; always
writes a valid `solution.txt`.

**Practice result (kit scorer): 1,000,000 / 1,000,000** — all 5 instances at
their push-optimum P* (13/16/29/47/60 pushes), in ~0.4 s for the whole set
(budget 60 s), deterministic. House baseline plain-astar 1.0.0 scores
200,000 on the same practice set.

Node 22, `node /bot/main.ts`, no dependencies, no randomness, no clock.

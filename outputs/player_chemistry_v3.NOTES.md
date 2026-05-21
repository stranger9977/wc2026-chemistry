# player_chemistry_v3 build notes

Generated: 2026-05-21

## Final row counts
- `player_chemistry_v3.parquet`: **2164** rows
- `career_familiarity.parquet`: **24213** pairs

## Contexts with usable data
- 43_3: 335 player-context rows
- 55_43: 266 player-context rows
- 43_106: 327 player-context rows
- 55_282: 251 player-context rows
- 223_282: 162 player-context rows
- 1267_107: 264 player-context rows
- 11_1: 94 player-context rows
- 11_4: 88 player-context rows
- 11_42: 79 player-context rows
- 11_90: 80 player-context rows
- 7_108: 60 player-context rows
- 7_235: 60 player-context rows
- 9_281: 81 player-context rows
- 44_107: 17 player-context rows

## Top-5 player-contexts by per90_vaep (sanity check)
- Lautaro Javier Martínez — Copa 2024 (Argentina): per90_vaep=1.9827, minutes=199, joi_top5_sum=-0.0162
- Lionel Andrés Messi Cuccittini — Barcelona La Liga 18/19 (Barcelona): per90_vaep=1.2029, minutes=2714, joi_top5_sum=1.3422
- Lionel Andrés Messi Cuccittini — Barcelona La Liga 17/18 (Barcelona): per90_vaep=1.0858, minutes=2998, joi_top5_sum=1.4004
- Denis Cheryshev — WC 2018 (Russia): per90_vaep=1.0508, minutes=281, joi_top5_sum=0.4545
- Yerry Fernando Mina González — WC 2018 (Colombia): per90_vaep=1.0206, minutes=270, joi_top5_sum=0.9527

## Methodological decisions
- VAEP v2 scored files have a single ``vaep_value`` column; we split
  per-action by SPADL action type. Defensive types (tackle, interception,
  clearance, keeper actions) contribute to ``per90_defensive``; everything
  else contributes to ``per90_offensive``. This is a coarser split than
  socceraction's native offensive/defensive value split (not in cache).

- JOI follows the existing repo convention
  (scripts/vaep_cross_context_pipeline.py): sum VAEP of the *second*
  action of every consecutive (different teammate, same team, same
  match) pair. JOI90 = JOI * 90 / shared_minutes, only valid for pairs
  with >= 180 shared minutes in the context.

- Shared minutes per pair computed from raw event Starting XI +
  Substitution events. Players who don't appear in events default to
  not being on the pitch. Red cards not modeled (rare).

- Pass graph for centrality uses SPADL pass-family types
  (pass/cross/freekick/corner) with result=success. Receiver inferred
  as next same-team-period action's actor. Eigenvector centrality
  computed via numpy power iteration on the symmetrized adjacency
  (networkx not installed in .venv and pip install blocked per spec).

- Per-context minimum: 180 minutes (matches site filter).

- ``embeddedness_score``: for each of player P's top-5 partners Q, sum
  Q's JOI90 with each other teammate R (R != P), then sum across all
  five partners. Higher = top partners are themselves well-connected.

- ``career_familiarity``: shared minutes summed across ALL contexts
  (club + intl), per player pair. Same player in different teams within
  the same context is collapsed via team-aware match aggregation, then
  pair shared minutes summed.

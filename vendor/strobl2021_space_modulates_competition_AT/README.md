# Strobl on-lattice controlled adapter

This directory contains a pristine pinned upstream snapshot and a separate,
headless controlled implementation based on the upstream Java model. See
`UPSTREAM.md` before redistributing any upstream material.

## Released-JAR baseline

The unmodified released JAR was first run from the temporary pinned checkout
with this exact deterministic short-smoke command:

```bash
java -Djava.awt.headless=true -jar vendor/strobl2021_space_modulates_competition_AT/upstream-repo/onLatticeModel.jar -initialSize 0.02 -rFrac 0.1 -turnover 0 -cost 0 -tEnd 2 -seed 7 -nReplicates 1 -compareToMTD false -profilingMode false -terminateAtProgression false -outDir vendor/strobl2021_space_modulates_competition_AT/baseline/
```

The same smoke can be rerun against the retained snapshot by replacing the JAR
path with `upstream/onLatticeModel.jar` and selecting a writable output
directory.

- Runtime: Java `17.0.1` (`Java HotSpot(TM) 64-Bit Server VM`,
  build `17.0.1+12-LTS-39`)
- JAR SHA-256:
  `42cb0b7cba654cfe2297c47d13285ffdc143a0554ed75b75ad40cc1a48ad3983`
- Golden output: `fixtures/released_jar_smoke.csv`
- Golden SHA-256:
  `902c85ace97418c193ffa8bb8033c24797ca3aa2133eac30329b95c347ef129a`

The released-JAR CSV schema is:

```text
TIdx,Time,NCells_S,NCells_R,NCells,DrugConcentration,rS,rR,mS,mR,dS,dR,dD,dt,NCycles,NAttemptedDivs,NFailedDivs,NDeaths,ReplicateId,InitSize,RFrac,TxName,Cost
```

The smoke deliberately preserves the released executable's stepping behavior:
with `-tEnd 2` it emits rows at times 0 through 3.

## Controlled model semantics

`ControlledModel.reset(...)` resets the model and both random streams.
`ControlledModel.step(double dose)` advances exactly one time step using one
homogeneous global dose in `[0,1]`.

- Grid values are exactly `0=empty`, `1=sensitive`, `2=resistant`.
- The lattice is unstackable, has no-flux boundaries, and has no movement.
- Drug killing applies only to sensitive cells and, matching the upstream
  source, is conditional on a division attempt having an empty neighbor.
- The externally controlled step never changes its own dose and never stops an
  episode for progression.
- Counts and per-phenotype attempted divisions, blocked divisions, natural
  deaths, and drug deaths are exposed after each step.
- The simulation and initial-condition random seeds are independent.

Initial-condition families preserve the requested sensitive/resistant counts:

- `random_mixed`: random occupied sites and phenotype assignment;
- `resistant_core`: compact tumor, resistant cells nearest its center;
- `resistant_edge`: compact tumor, resistant cells on its outer edge;
- `resistant_dispersed`: compact tumor, resistant sites greedily separated;
- `two_resistant_nests`: compact tumor with resistant cells split between two
  nests.

The optional batch policy `paper_adaptive` uses the released source condition
`N > N0`. The separately named `paper_text_ge` variant uses `N >= N0`.
Neither policy introduces progression-based episode termination.

## Build and test

```bash
vendor/strobl2021_space_modulates_competition_AT/build.sh
vendor/strobl2021_space_modulates_competition_AT/test.sh
```

The build targets Java 8 bytecode. It writes ZIP entries in sorted order with a
fixed timestamp so repeated builds produce the same `controlled-model.jar`.

## Fast batch CLI

```bash
java -jar vendor/strobl2021_space_modulates_competition_AT/controlled-model.jar \
  --mode batch --width 100 --height 100 \
  --family random_mixed --sensitive 6750 --resistant 750 \
  --simulation-seed 7 --ic-seed 7 --steps 100 --dose 0.5 \
  --policy external --out trajectory.csv --grid-out final_grid.csv
```

Batch trajectory schema:

```text
step,time,dose,sensitive,resistant,total,attempted_divisions_sensitive,attempted_divisions_resistant,blocked_divisions_sensitive,blocked_divisions_resistant,natural_deaths_sensitive,natural_deaths_resistant,drug_deaths_sensitive,drug_deaths_resistant
```

Use `--policy paper_adaptive` or `--policy paper_text_ge` and optionally
`--withdrawal-fraction 0.5` for internal dose policies. Use
`--division-sensitive`, `--division-resistant`, `--death-sensitive`,
`--death-resistant`, `--drug-kill`, and `--dt` to change model parameters.

## Persistent line protocol

Start a process:

```bash
java -jar vendor/strobl2021_space_modulates_competition_AT/controlled-model.jar \
  --mode serve --width 100 --height 100
```

Commands are whitespace-delimited, one per line:

```text
RESET family sensitive resistant simulationSeed icSeed [sharedOccupiedMask]
INIT family sensitive resistant simulationSeed icSeed [sharedOccupiedMask]
STEP dose
COUNTS
GRID
QUIT
```

Every successful reset, step, or count request returns one tab-delimited
`STATE` line. `GRID` returns row-major categorical values. Errors are returned
as one `ERROR` line without terminating the process. Set the optional final
reset argument to `true` for matched-state episodes that must reuse one compact
occupied-site mask across all five resistant-cell arrangements.
The initial `READY` line includes both protocol version and pinned upstream
commit; Python refuses runners that do not match either value.

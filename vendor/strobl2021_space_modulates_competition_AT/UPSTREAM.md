# Upstream provenance

- Repository: <https://github.com/MathOnco/strobl2021_space_modulates_competition_AT>
- Pinned commit: `aa3b3c2ad2e4acf9fd7cc6ac318f1bf79f9361e2`
- Released executable SHA-256:
  `42cb0b7cba654cfe2297c47d13285ffdc143a0554ed75b75ad40cc1a48ad3983`
- Snapshot date: 2026-08-24

## Citation

Strobl, M. A. R., Gallaher, J., West, J., Robertson-Tessi, M., Maini,
P. K., & Anderson, A. R. A. (2022). Spatial structure impacts adaptive
therapy by shaping intra-tumoral competition. *Communications Medicine*,
2, 46. <https://doi.org/10.1038/s43856-022-00110-x>

The associated archived release is:
<https://doi.org/10.5281/zenodo.5504425>.

## Licensing status and permission

No public software license was present in the pinned upstream repository.
GitHub API status at the pinned commit: `license: null`.
Consequently, this snapshot must not be assumed to grant redistribution,
modification, or sublicensing rights. The user explicitly authorized vendoring
this upstream code for this repository despite the absent public license. That
authorization is recorded here as project provenance; it is not a public
license and does not grant rights to third parties.

## Pristine snapshot

`upstream/` is an unmodified extraction from the pinned commit:

- upstream `README.md`;
- `abm/onLatticeCA/` model sources;
- `abm/Framework/` HAL source/subset and its checked-in support files;
- the upstream manifest; and
- the released `onLatticeModel.jar`.

`UPSTREAM_SHA256SUMS` records every retained snapshot file.

Do not edit files under `upstream/`. The controlled implementation is separate
under `src/`.

## Unmodified released-JAR regression

The release artifact was executed before local implementation work using Java
`17.0.1` (`Java HotSpot(TM) 64-Bit Server VM`, build
`17.0.1+12-LTS-39`):

```bash
java -Djava.awt.headless=true -jar upstream/onLatticeModel.jar \
  -initialSize 0.02 -rFrac 0.1 -turnover 0 -cost 0 -tEnd 2 \
  -seed 7 -nReplicates 1 -compareToMTD false \
  -profilingMode false -terminateAtProgression false -outDir baseline/
```

`profilingMode=false` is required because the released profiling path retains
only five observations. The emitted schema is:

```text
TIdx,Time,NCells_S,NCells_R,NCells,DrugConcentration,rS,rR,mS,mR,dS,dR,dD,dt,NCycles,NAttemptedDivs,NFailedDivs,NDeaths,ReplicateId,InitSize,RFrac,TxName,Cost
```

The retained golden trajectory is `fixtures/released_jar_smoke.csv`, SHA-256
`902c85ace97418c193ffa8bb8033c24797ca3aa2133eac30329b95c347ef129a`.
`test_released_jar.sh` verifies both this file and the released JAR checksum.

## Local changes

The local implementation does not patch the snapshot. It adds:

- a Java-8-compatible, headless scalar-dose model and API;
- deterministic initial-condition families with separate simulation and IC
  seeds;
- a shared compact occupied-site-mask option for matched-state architecture
  comparisons;
- exact categorical grid and count accessors;
- per-phenotype attempted/blocked division and natural/drug-death diagnostics;
- an external one-step control API with no movement and no internal
  progression termination;
- batch CSV and persistent line-protocol CLIs;
- explicit `paper_adaptive` (`N > N0`) and `paper_text_ge` (`N >= N0`)
  policies;
- a reproducible JAR build; and
- deterministic regression fixtures and smoke tests.

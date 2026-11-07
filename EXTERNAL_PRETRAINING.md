# Public external pretraining

Source: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data),
Sample Games 1 and 2, commit `e706dd506b360d69d9d123d5b8026e7294b13996`.
Credit: Metrica Sports. The upstream README provides public sample data, asks
users to act responsibly and acknowledge the source in public work. No separate
LICENSE was supplied. Raw external files are downloaded into the private workspace
and are not redistributed here. Reproduction must fetch the pinned public URLs.

The upstream event-definition document defines PASS as a transfer between players
of the same team. Tracking and events are synchronized; normalized coordinates
use a 105 x 68 m pitch. For experiments 17 and 20, only closed PASS intervals provide weak H/A supervision.
Unknown gaps, offside passes and opposite-team overlaps are excluded. Those experiments infer no dead-ball
labels. This is external weak supervision, not target-match ground
truth. The external tracking has missing ball observations on about 40% of frames.

Experiment 17's fixed training and holdout protocol is in [F1_TARGET.md](F1_TARGET.md).
No target-match reference labels enter external preparation or pretraining.

Experiment 23 additionally prepares full event-derived control context from Games
1/2. It uses the conservative state-machine semantics fixed in F1_TARGET.md:
known control starts, known stops, censored unresolved gaps and contradictory
pass spans. D/X frames supply context but no owner loss. A frozen spatial encoder
and instantaneous probe feed a shared temporal residual owner and invariant
turnover head. Team and half boundaries are preserved; synthetic missing-ball
blocks are generated independently of labels.

Sample Game 3 is reserved for a single conditional external transfer gate after
the fixed final epoch. Its XML metadata identifies DEMO_1002 dated 2020-05-08; its
two-goal sequence and half durations differ from the target fixture. Three pinned
files (metadata XML, FIFA tracking text and event JSON) are hash-checked. The
parser follows roster sections and goalkeeper identities rather than assuming a
fixed player column order. Neither Game 3 nor target-match observations or labels
fit temporal normalization, parameters, checkpoints or hyperparameters. External
conditional owner/turnover metrics do not measure target-match spell F1.

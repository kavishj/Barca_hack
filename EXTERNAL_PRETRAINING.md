# Public external spatial pretraining

Source: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data),
Sample Games 1 and 2, commit `e706dd506b360d69d9d123d5b8026e7294b13996`.
Credit: Metrica Sports. The upstream README provides public sample data, asks
users to act responsibly and acknowledge the source in public work. No separate
LICENSE was supplied. Raw external files are downloaded into the private workspace
and are not redistributed here. Reproduction must fetch the pinned public URLs.

The upstream event-definition document defines PASS as a transfer between players
of the same team. Tracking and events are synchronized; normalized coordinates
use a 105 x 68 m pitch. Only closed PASS intervals provide weak H/A supervision.
Unknown gaps, offside passes and opposite-team overlaps are excluded. No dead-ball
labels are inferred. This is external weak supervision, not target-match ground
truth. The external tracking has missing ball observations on about 40% of frames.

Experiment 17's fixed training and holdout protocol is in [F1_TARGET.md](F1_TARGET.md).
No target-match reference labels enter external preparation or pretraining.

# The M1.5 network

M1.5 is Mouse's own network, shipped as `M1.5.onnx` with 19,904,304 learned
parameters. It takes Lc0's 112-plane input and uses Lc0's 1858-move policy map. It runs on
TensorRT through the `mouse_moe_trt` plugin, whose source is in `engine/nn/trt_moe/`.

## Licence

The network file is licensed under the GNU General Public License, version 3 or later,
like the rest of Mouse.

## Training data notice

Contains information from the
[Leela Chess Zero training data](https://storage.lczero.org/files/training_data/), which is
made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/). Individual
contents are licensed under the
[Database Contents License (DbCL)](https://opendatacommons.org/licenses/dbcl/1-0/).

M1.5 was trained on the project's `test80` and `test90` runs:

- 22 `test80` files, about 182 million positions, using the changes described below.
- About 103 million `test90` positions, unchanged.

## Changes made to the data

Mouse used a further copy of the `test80` positions with new targets from an Lc0 network
acting as a teacher, [BT4-1024x15x32h-swa-6147500-policytune-332](https://lczero.org/).
This section is the method of making those changes, offered as section 4.6 of the ODbL
provides:

1. Encode each position as Lc0's 112 input planes and evaluate it once with the teacher,
   in half precision (FP16). No search is involved.
2. Replace the position's policy target with the teacher's 16 most likely legal moves and
   their probabilities. The probabilities come from a softmax over the legal moves only and
   are not renormalised over the 16; a position with fewer than 16 legal moves has
   probability 0 in the unused places.
3. Replace its win/draw/loss and moves-left targets with the teacher's outputs.

While training on either run, positions in which neither side could still castle were
mirrored left to right with probability ½, and positions from the first 10 plies of a game
were kept with probability 0.4, so that repeated openings counted for less. Neither step
changes the stored data.

## Credit

M1.5 exists because of the Leela Chess Zero project: it was trained on Lc0's open
training data and distilled from Lc0's BT4 network, which provided its training targets.
That data and those networks are the work of the Lc0 project and the many contributors who
generated its games. Mouse and M1.5 are not affiliated with or endorsed by the project.

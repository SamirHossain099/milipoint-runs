# milipoint-runs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23149604.svg)](https://doi.org/10.5281/zenodo.23149604)

Run-disjoint evaluation of the MiliPoint millimeter-wave radar dataset (Cui et al., NeurIPS 2023 Datasets and
Benchmarks): what its released splits reward, and protocols that measure what the tasks ask.

- The released identification and keypoint splits shuffle overlapping five-frame windows before dividing them, so
  about 96% of test windows share four frames with a training window. Both splits are rebuilt exactly, each in its
  loader's own frame order.
- Five participants recorded the workout in several runs. On these five, the dataset authors' PointMLP identifies
  about 94% of randomly split windows but 37-39% when one run per participant is held out (chance 20%); a
  PointNet-style network shows the same drop, also with a training set cut to the run-disjoint size.
- Under a shuffled split, a nearest-neighbor lookup with no learning reaches about 99% action accuracy on fifty-frame
  windows; under the released contiguous action split it reaches about 5%. The split of the MAK-GCN code is rebuilt
  too.
- Keypoint error of a simple network rises by at most about 19% with participants held out.
- The released action loader indexes the unfiltered label array with filtered positions, so about 3% of action
  windows (five frames, the released configs) or 30% (fifty frames, the paper) are stacked wrongly.

## Running it

Download `MiliPoint_data.zip` (link in https://github.com/yizzfz/MiliPoint), unzip it to `<root>/raw/`, clone the
official repository to `<root>/code/`, and set `MILIPOINT_ROOT=<root>`.

```
pip install -r requirements.txt
python scripts/gate.py && python scripts/pointnet_iden.py
python scripts/pointmlp_iden.py 0 && python scripts/pointmlp_iden.py 1
python scripts/action_oracle.py && python scripts/action_bug.py
python manuscript_numbers.py && python figures.py
python -m pytest tests -q
```

`pointmlp_iden.py` runs the authors' `pointmlp.py` unchanged; its two torch_geometric operations (farthest-point
sampling from a random start, and k-nearest neighbors) are supplied in plain PyTorch because the compiled library has
no build for recent PyTorch. It trains for 3 epochs at a learning rate of 1e-3, shorter than the released defaults.
`results/` holds aggregated results only; the per-frame caches are rebuilt from the dataset. The claim tests and the
packaging guard run in CI on Python 3.11 to 3.13.

## Licence

MIT for this code. MiliPoint belongs to its authors and is MIT licensed.

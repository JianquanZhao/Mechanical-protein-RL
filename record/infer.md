## 2026-09-02: DDQN batch protein-design entry point

### Goal and inference boundary

Added the project-level inference entry point `run_rl.py`. It applies a trained
per-residue DDQN online Q network to every supported protein file in a folder
and writes designed amino-acid sequences as FASTA files.

Inference deliberately does not instantiate `MechanicalProteinEnv`, PyRosetta,
the step reward, the terminal mechanical-property predictor, replay, or an
optimizer. Both structure and sequence inputs are reduced to a single-chain
sequence. Each design step then follows the observation/action contract used
during training:

```text
current sequence
-> ESM2 per-residue embedding
-> optional visited-position channel inferred from checkpoint shape
-> online Q head
-> no-op/visited action masking
-> epsilon=0 greedy mutation
-> updated sequence
```

ESM2 is recomputed after every mutation because changing the sequence changes
the contextual embedding at every residue. Positions are not revisited, so an
output named with `N` updates contains exactly `N` changed positions relative
to its source sequence.

### Command-line interface

Required arguments:

```text
--model-path
--input-type {structure,sequence}
--input-dir
--output-dir
```

`--model-path` accepts either one checkpoint file or a checkpoint directory.
For a directory, the most recently modified `agent*.pt` file is selected. The
script reads `state_shape`, `action_dim`, `config`, and `online_network` from
the checkpoint and automatically infers:

- ESM2 embedding dimension: 1280, 2560, or 5120;
- Q-head hidden dimensions;
- whether the checkpoint expects an additional visited-position feature;
- compatibility with the dynamic per-residue `L * 20` action layout.

Optional runtime arguments:

```text
--updates 24
--updates 24,48,72
--updates 24:72
--device cuda:0
--esm-device cuda:0
--esm-model-dir /path/to/esm/checkpoints
--seed 7
--overwrite
--fail-fast
--log-level INFO
```

When `--updates` is omitted, one greedy trajectory is run through sequence
length `L`, and snapshots are emitted for every update count in the inclusive
range `24..L`. Multiple output counts reuse this one trajectory rather than
restarting inference. An explicitly supplied update count may be below 24 but
must be positive. If either the default minimum or an explicitly requested
count exceeds `L`, the count is capped to `L` with a warning. For example,
`--updates 24` on an 11-residue sequence produces `<name>_11.fasta`. This keeps
folder-wide inference running while preserving the guarantee that every update
changes a distinct residue position.

### Input parsing and validation

Structure mode recursively discovers `.pdb` and `.ent` files. It parses
canonical residues from `ATOM` records in the first model, handles residue
insertion codes and common alternate locations, and requires exactly one
canonical protein chain. Ligands, water, nucleic acids, and noncanonical
residues are not converted into sequence positions. Multi-chain structures are
reported as invalid rather than silently selecting a chain.

Sequence mode recursively discovers `.fa`, `.faa`, `.fas`, `.fasta`, and
`.fsa`. Every file must contain exactly one FASTA record. Both modes accept only
the 20 canonical amino acids because the trained action/no-op mask is aligned
to that alphabet.

Input basenames must be unique across the recursive folder. This prevents two
files such as `folder_a/1ACF.pdb` and `folder_b/1ACF.pdb` from overwriting the
same requested output name.

By default, an invalid protein is logged with its traceback and the remaining
folder continues. The process returns a nonzero exit code when any input
failed. `--fail-fast` instead stops immediately.

### Output format

For source `1ACF.pdb` or `1ACF.fasta`, the 24-update result is:

```text
<output-dir>/1ACF_24.fasta
```

Its FASTA header is `>1ACF_24`, and sequence lines are wrapped at 80 residues.
Files are written through a temporary path and atomically renamed. Existing
files are protected unless `--overwrite` is supplied.

### Usage examples

Structure folder, one requested design depth:

```bash
python run_rl.py \
  --model-path /path/to/checkpoints/agent_00302400.pt \
  --input-type structure \
  --input-dir /path/to/pdbs \
  --output-dir /path/to/designed_sequences \
  --updates 24 \
  --device cuda:0 \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm
```

FASTA folder, several snapshots from each trajectory:

```bash
python run_rl.py \
  --model-path /path/to/checkpoints \
  --input-type sequence \
  --input-dir /path/to/fastas \
  --output-dir /path/to/designed_sequences \
  --updates 24,48,72 \
  --device cuda:0 \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm
```

Default `24..L` behavior:

```bash
python run_rl.py \
  --model-path /path/to/agent.pt \
  --input-type sequence \
  --input-dir /path/to/fastas \
  --output-dir /path/to/designed_sequences
```

### Verification

Added `tests/test_run_rl.py` with coverage for:

- single-record and multi-record FASTA behavior;
- canonical single-chain PDB extraction and multi-chain rejection;
- default, list, and inclusive-range update specifications;
- no-op and visited-position action masks;
- unique-position greedy mutations and trajectory snapshots;
- loading a dynamic per-residue checkpoint;
- required output filename and FASTA header through the CLI.

Targeted tests:

```text
python -m pytest -q tests/test_run_rl.py
9 passed in 2.21s
```

A real CPU smoke test used:

```text
checkpoint: agent_00302400.pt
checkpoint environment steps: 7,257,681
checkpoint optimization steps: 906,693
state feature dimension: 1281
ESM2: local esm2_t33_650M_UR50D checkpoint
input: model/reward_module/wild_type.pdb
parsed chain: G
parsed length: 173
updates: 1
output: wild_type_1.fasta
result: completed successfully
```

The smoke test took 12.27 seconds on CPU, of which approximately 11 seconds
were one-time ESM2 model loading. This validates the complete
checkpoint -> PDB sequence -> ESM2 -> Q action -> FASTA path; production folder
runs should use CUDA for ESM2 throughput.

# confgen

This project is based on ehoogeboom's e3_diffusion_for_molecules(https://github.com/ehoogeboom/e3_diffusion_for_molecules/) released under the MIT License.

For training:

python train.py --exp_name "$EXP_NAME" --data_path "$PATH_TO_DATASET_DIR" [--num_workers "$NUM_WORKERS_FOR_DATALOADER" --gpus "true/false"]

Notes:
1. num_workers is the num_workers argument to be passed on to the dataloaders.
2. You may wonder why gpus is an option and not something automatically recognized by the python code. I agree. But's let's just call this a "feature".
3. There's bunch of other options, but they're self-explantory or same as ehoogeboom.
4. lambda_dist is a parameter that I have because of "historical" reasons (horrible bond lengths in my ver0 model). I made a bunch of changes since then and it might not be required, but I never trained without it. Set to 0.002 by default.
5. There is --dp option for data parallelism. It doesn't really work as expected so don't use the option. Without DP, training on the entire qm9 dataset takes about 5 days for 3000 epochs

For conformer generation:

python generate.py --name "$EXP_NAME" --dir "$PATH_TO_MOL_GRAPH_DIR" [--chain "true/false"]

Notes:
1. --chain default is false. true to visualize diffusion trajectory.
2. Make sure all the python files are in the same directory when you run them.
3. train.py makes an outputs/ directory that stores all the params/hyperparams. generate.py must be in the same directory as train.py for the file to access outputs/

Molecular graph file must be a txt file containing a graph for one molecule (1 file for 1 graph)
format:

<NUM_ATOMS><br />
<ATOM_SYMBOL> <HYBRIDIZATION> <AROMATICITY> <DEGREE> <VALENCE><br />
...<br />
<ATOM_SYMBOL> <HYBRIDIZATION> <AROMATICITY> <DEGREE> <VALENCE><br />
<ATOM_IDX> <ATOM_IDX> <BOND_ORDER> <AROMATICITY><br />
...<br />
<ATOM_IDX> <ATOM_IDX> <BOND_ORDER> <AROMATICITY><br />

Notes:
1. Atom symbol must be uppercase
2. Hybridization is 1 for sp, 2 for sp2, 3 for sp3
3. In accordance with openbabel, terminal H or F is sp hybridized
4. aromaticity is 1 for aromatic and 0 for nonaromatic. (No anti-aromatics)
5. Atom indexing starts at 0
6. Bond order is an integer. For aromatic molecules, find a resonance form you like and go with it.
7. Yes hand-jamming sucks but this was never meant to be practical.

Example: Methane

5<br />
C 3 0 4 4<br />
H 1 0 1 1<br />
H 1 0 1 1<br />
H 1 0 1 1<br />
H 1 0 1 1<br />
0 1 1 0<br />
0 2 1 0<br />
0 3 1 0<br />
0 4 1 0<br />

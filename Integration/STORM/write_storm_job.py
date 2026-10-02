#!/usr/bin/env python
"""
Write the SLURM job script that runs STORM/storm_chain_run.py.

    python write_storm_job.py --sections S1 S2 ... --edges <edges.json> --tag <run id> \
        --graph-only --save-graph <dir>          # the graph build (CPU)
    python write_storm_job.py --sections S1 S2 ... --edges <edges.json> --tag <run id> \
        --load-graph <dir>                       # the fit on the saved graph (GPU)

Checks the inputs first, so a missing file fails here rather than after the
queue wait, then prints `wrote <script>`. run_all.py submits the script and
waits for the job.
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import REPO_ROOT as BASE, rpath
# BASE is the REPOSITORY (code, shared caches); dataset data is rpath()
STORM_DIR = os.path.dirname(os.path.abspath(__file__))   # Integration/STORM: wrappers + the STORM package
# storm_chain_run.py lives INSIDE the STORM package dir (PKG_DIR) and does
# `from STORM import STORM, ...`. `python /path/to/PKG_DIR/storm_chain_run.py`
# sets sys.path[0] to the SCRIPT'S OWN DIRECTORY (PKG_DIR) regardless of cwd/cd
# -- cd does not affect this. From inside PKG_DIR, 'STORM' resolves to the
# sibling STORM.py file, not the package, so STORM.py's own `from STORM.utils
# import ...` breaks with "'STORM' is not a package". The job therefore cds to
# STORM_DIR (PKG_DIR's parent) and runs the worker as `python -m
# STORM.storm_chain_run`: -m adds cwd (STORM_DIR) to sys.path[0] instead of the
# script's directory, so 'STORM' correctly resolves to the STORM/ package.
PKG_DIR   = os.path.join(STORM_DIR, 'STORM')
WORKER    = os.path.join(PKG_DIR, 'storm_chain_run.py')
PREPPED   = rpath('prepped')
OUTBASE   = rpath('storm_out')
JOBDIR    = os.path.join(BASE, 'jobs')
CONDA_ENV = 'st-graph-parafac2-gpu'

# 'string' fetches from the STRING API and caches under PPI_CACHE. The version
# is pinned and the resolved adjacency is hashed into the run record, because
# an unpinned fetch would make the run irreproducible -- STRING releases change
# and a rerun months later would silently get a different graph.
# Pass a file path instead if the node has no outbound HTTPS.
PPI       = 'string'
PPI_CACHE = os.path.join(BASE, 'cache', 'ppi_cache')
STRING_VERSION = '12.0' 

# HVG candidate pool: HGNC protein-coding genes only (default since 2026-09-11).
# '--gene-pool all --hvg-tiebreak order' reproduces the earlier runs.
GENE_ANNOTATION = os.path.join(BASE, 'reference', 'hgnc_complete_set.txt')

ap = argparse.ArgumentParser()
ap.add_argument('--sections', nargs='+', required=True,
                help='the sections to fit (the data card\'s)')
ap.add_argument('--rank', type=int, default=10,
                help='fixed rank (a stated choice); 0 = select by congruence')
ap.add_argument('--resolutions', nargs='*', type=float, default=None,
                help='pass ONE value to pin the clustering resolution by hand')
ap.add_argument('--gamma', type=float, default=1.0)
ap.add_argument('--iters', type=int, default=35)
ap.add_argument('--n-niches', type=int, default=800)
ap.add_argument('--n-top-genes', type=int, default=2000)
ap.add_argument('--gene-pool', choices=['coding', 'all'], default='coding',
                help='HVG candidates: HGNC protein-coding only, or every gene')
ap.add_argument('--gene-annotation', default=GENE_ANNOTATION)
ap.add_argument('--hvg-tiebreak', choices=['median_rank', 'order'],
                default='median_rank')
ap.add_argument('--k-cross', type=float, default=5.0,
                help='literal cross degree; only used with --lambda-cross < 0')
# within-sample graph + clustering representation (see storm_chain_run.py)
ap.add_argument('--within-graph', choices=['scc', 'radius'], default='scc',
                help="scc = Stereopy SCC graph (expression kNN U lattice), "
                     "the paper's layer-calling graph; radius = old graph")
ap.add_argument('--expr-k', type=int, default=10)
ap.add_argument('--spatial-neigh', choices=['rook', 'queen'], default='rook')
ap.add_argument('--lambda-cross', type=float, default=0.5,
                help='cross degree as a fraction of realised within degree')
ap.add_argument('--cluster-rep', choices=['QHD', 'QH'], default='QH',
                help='QHD = Q_k H D_k (default), QH = Q_k H')
ap.add_argument('--gene-chunk-cols', type=int, default=256)
ap.add_argument('--save-graph', default=None,
                help='dir to persist L_s for later --load-graph refits')
ap.add_argument('--load-graph', default=None)
ap.add_argument('--edges', default=None,
                help="edge JSON file (default prepped/edges.json) or 'derive' (lot-matched)")
ap.add_argument('--ppi', default=PPI,
                help="'string' / 'string:12.0' for the API, or a file path")
ap.add_argument('--ppi-cache', default=PPI_CACHE)
ap.add_argument('--string-version', default=STRING_VERSION)
ap.add_argument('--ppi-min-score', type=float, default=700.0,
                help='STRING combined_score cut; 700 = high confidence (default '
                     'with the coding pool), 600 = the cut of earlier runs')
ap.add_argument('--tag', default='')
ap.add_argument('--time', default='12:00:00')
ap.add_argument('--mem', default='250G')
ap.add_argument('--cpus', type=int, default=8)
ap.add_argument('--gpu', default='a100:1', help="'' for CPU only")
ap.add_argument('--graph-only', action='store_true',
                help='build and save the graph only (CPU); fit later with --load-graph')
ap.add_argument('--job-label', default='',
                help='suffix for the job name and its .sb/.out files, so a graph job '
                     'and a fit job of the same tag keep separate logs')
ap.add_argument('--jobdir', default=JOBDIR,
                help='where the job script and its log go (run_all.py: <run>/jobs)')
a = ap.parse_args()
JOBDIR = a.jobdir

os.makedirs(JOBDIR, exist_ok=True)

# ---- pre-flight: fail here, not after the queue wait ---------------------
problems, warns = [], []
for f in (WORKER,
          os.path.join(PKG_DIR, 'ChainGraph.py'),
          os.path.join(PKG_DIR, 'AnchorGraph.py'),
          os.path.join(PKG_DIR, 'SCCGraph.py'),
          os.path.join(PKG_DIR, 'STORM.py'),
          os.path.join(PKG_DIR, 'fit_STORM_chunked.py')):
    if not os.path.isfile(f):
        problems.append(f'missing {f}')

if not os.path.isdir(PREPPED):
    problems.append(f'missing prepped dir: {PREPPED}')
else:
    h5 = sorted(f.split('_prepped')[0] for f in os.listdir(PREPPED)
                if f.endswith('_prepped.h5ad'))
    if not h5:
        problems.append(f'no *_prepped.h5ad in {PREPPED}')
    else:
        print(f'prepped: {len(h5)} ({", ".join(h5)})')
        miss = [s for s in a.sections if s not in h5]
        if miss:
            problems.append(f'no prepped h5ad for {miss} in {PREPPED}')

if str(a.ppi).lower().startswith('string'):
    os.makedirs(a.ppi_cache, exist_ok=True)
    cached = [f for f in os.listdir(a.ppi_cache) if f.startswith('string_v')]
    print(f'PPI: STRING API v{a.string_version} '
          f'(cache {a.ppi_cache}, {len(cached)} entries)')
    if not cached:
        print('     first run will fetch over HTTPS; it hard-fails if the '
              'node has no outbound access')
elif not os.path.isfile(a.ppi):
    problems.append(f'PPI file not found: {a.ppi}')

if a.gene_pool == 'coding' and not os.path.isfile(a.gene_annotation):
    problems.append(f'gene annotation not found: {a.gene_annotation}')

for w in warns:
    print(f'\n[WARN] {w}\n')
if problems:
    sys.exit('pre-flight failed:\n  ' + '\n  '.join(problems))

# ---- assemble -----------------------------------------------------------
parts = [f'n{len(a.sections)}']
parts.append(f'r{a.rank}' if a.rank else 'rauto')
parts.append(f'g{a.gamma:g}')
parts.append(a.within_graph)
parts.append(a.cluster_rep)
if a.tag:
    parts.append(a.tag)
tag = '_'.join(parts)
outdir = os.path.join(OUTBASE, tag)
jname = tag + (f'_{a.job_label}' if a.job_label else '')
sb = os.path.join(JOBDIR, f'__storm_{jname}.sb')
if a.save_graph and a.load_graph:
    sys.exit('--save-graph and --load-graph are mutually exclusive')
if a.load_graph and not os.path.isfile(os.path.join(a.load_graph,
                                                    'graph_key.json')):
    sys.exit(f'--load-graph {a.load_graph}: no graph_key.json there')

cmd = ['python -m STORM.storm_chain_run',
       f'--prepped "{PREPPED}"',
       f'--out "{outdir}"',
       f'--ppi "{a.ppi}"',
       f'--ppi-cache "{a.ppi_cache}"',
       f'--string-version {a.string_version}',
       f'--ppi-min-score {a.ppi_min_score:g}',
       f'--gamma {a.gamma:g}',
       f'--iters {a.iters}',
       f'--n-niches {a.n_niches}',
       f'--n-top-genes {a.n_top_genes}',
       f'--gene-pool {a.gene_pool}',
       f'--gene-annotation "{a.gene_annotation}"',
       f'--hvg-tiebreak {a.hvg_tiebreak}',
       f'--k-cross {a.k_cross:g}',
       f'--within-graph {a.within_graph}',
       f'--expr-k {a.expr_k}',
       f'--spatial-neigh {a.spatial_neigh}',
       f'--lambda-cross {a.lambda_cross:g}',
       f'--cluster-rep {a.cluster_rep}',
       f'--gene-chunk-cols {a.gene_chunk_cols}',
       '--device cuda' if a.gpu else '--device cpu']
if a.edges:
    cmd.append(f'--edges "{a.edges}"')
if a.save_graph:
    cmd.append(f'--save-graph "{a.save_graph}"')
if a.load_graph:
    cmd.append(f'--load-graph "{a.load_graph}"')
if a.graph_only:
    cmd.append('--graph-only')
if a.rank:
    cmd.append(f'--rank {a.rank}')
if a.resolutions:
    cmd.append('--resolutions ' + ' '.join(f'{r:g}' for r in a.resolutions))
cmd.append('--sections ' + ' '.join(a.sections))

with open(sb, 'w') as f:
    f.write('#!/bin/bash --login\n#SBATCH --export=NONE\n')
    f.write(f'#SBATCH --time={a.time}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n')
    f.write(f'#SBATCH --cpus-per-task={a.cpus}\n#SBATCH --mem={a.mem}\n')
    if a.gpu:
        f.write(f'#SBATCH --gpus={a.gpu}\n')
    f.write(f'#SBATCH --job-name=storm_{jname}\n')
    f.write(f'#SBATCH --output={JOBDIR}/__storm_{jname}.out\n\n')
    f.write('set -eo pipefail\n')
    f.write('module purge\nmodule load Miniforge3\n')
    f.write('eval "$(conda shell.bash hook)"\n')
    f.write(f'conda activate {CONDA_ENV}\n\n')
    f.write(f'cd {STORM_DIR}\n')
    # fail fast on a missing dependency instead of burning the queue slot
    f.write('python -c "import torch, ot, scanpy, sklearn, igraph; '
            'print(\'deps ok, cuda:\', torch.cuda.is_available())" || '
            '{ echo FATAL: missing dependency; exit 1; }\n')
    if a.gpu:
        f.write('nvidia-smi || true\n')
    f.write('export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n\n')
    f.write(' \\\n    '.join(cmd) + '\n')

print(f'\nwrote {sb}')
print(f'  out  : {outdir}/')

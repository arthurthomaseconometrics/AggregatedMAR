#from subsampling_interactive import run_study

# Étude complète (M=200 suffit pour n=50000)
#run_study(sample_sizes=[1000, 10000, 50000],
#          M=200, b_exponent=2/3, n_cores=-1)

#from subsampling_reparam import run_study_reparam

# Avec b = n^{2/3} (recommandé)
#run_study_reparam(
#    sample_sizes=[250, 500, 1000, 10000,50000],
#    M=200, b_exponent=2/3, n_cores=-1
#)

#from subsampling_restricted_truepi import run_study_restricted

# Avec b = n^{2/3} (recommandé)
#run_study_restricted(
##    sample_sizes=[250, 500, 1000, 10000,50000,100000],
#    M=200, b_exponent=2/3, n_cores=-1
#)

from subsampling_reparam import run_study_reparam

run_study_reparam(
    sample_sizes=[250, 500, 1000, 10000, 50000, 100000],
    M=200, b_exponent=2/3, kappa=1.0, n_cores=-1, seed=42,
    ci_step=None,  # tous les blocs chevauchants, comme dans le papier
    output_dir='results_reparam'  # écrasera l'ancien dossier bugué — pensez à l'archiver avant si vous voulez garder une trace du "avant fix"
)
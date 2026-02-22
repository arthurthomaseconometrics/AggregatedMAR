#from subsampling_interactive import run_study

# Étude complète (M=200 suffit pour n=50000)
#run_study(sample_sizes=[250, 500, 1000, 5000],
#          M=200, b_exponent=0.75, n_cores=-1)

from subsampling_reparam import run_study_reparam

# Avec b = n^{2/3} (recommandé)
run_study_reparam(
    sample_sizes=[250, 500, 1000, 10000,50000,100000],
    M=200, b_exponent=2/3, n_cores=-1
)
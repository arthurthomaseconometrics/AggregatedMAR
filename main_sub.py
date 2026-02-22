from subsampling_interactive import run_sensitivity, run_study

# Sensibilité block size
run_sensitivity(sample_sizes=[500, 1000], M=100, n_cores=-1)

# Étude complète (M=200 suffit pour n=50000)
run_study(sample_sizes=[250, 500, 1000, 10000, 50000],
          M=200, b_exponent=0.75, n_cores=-1)
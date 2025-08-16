import cProfile
import pstats

p = pstats.Stats('profile_results.prof')
p.sort_stats('cumulative').print_stats(20)

import toh_mdp as tm
import solver_utils
import solvers

config = tm.TohMdpConfig(gamma=0.9, living_reward=0.0, noise=0.2, n_disks=2, n_goals=1)
mdp = tm.TohMdp.from_config(config)
solver = solvers.ValueIterationSolver(mdp)

v_table = solver.v_table
v_table, _, max_delta = solver_utils.value_iteration(mdp=mdp, v_table=v_table)
print("1st iteration max_delta was " + str(max_delta))

v_table, _, max_delta = solver_utils.value_iteration(mdp=mdp, v_table=v_table)
print("2nd iteration max_delta was " + str(max_delta))


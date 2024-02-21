import toh_mdp as tm
import solver_utils

config = tm.TohMdpConfig(0.9, 0.0, 0.2, 3, 1)
mdp = tm.TohMdp.from_config(config)

ans = solver_utils.value_iteration(mdp=mdp, v_table={})
print(ans[2])
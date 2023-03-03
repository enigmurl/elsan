from fluidsim.solvers.ns2d.solver import Simul
from fluidsim import load_sim_for_plot
from fluidsim import load_state_phys_file

params = Simul.create_default_params()

params.nu_2 = 1e-3
params.forcing.enable = False

params.init_fields.type = "noise"

params.output.periods_save.spatial_means = 1.0
params.output.periods_save.spectra = 1.0
params.output.periods_save.phys_fields = 2.0
sim = Simul(params)

sim.time_stepping.start()
sim.output.phys_fields.plot()
print(sim.output.phys_fields)
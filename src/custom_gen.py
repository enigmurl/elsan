from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()
sim = Simul(params)
sim.time_stepping.start()

if __name__ == '__main__':
    pass
import numpy
from matplotlib import pyplot
nx = 41
dx = 2 / (nx - 1)
nt = 25
dt = 0.025
c = 1

# initial conditions
u = numpy.ones(nx)
u[int(.5 / dx) : int(1 / dx + 1)] = 2
pyplot.plot(numpy.linspace(0, 2, nx), u)
pyplot.savefig("Step1-graph_1")

un = numpy.ones(nx)

for n in range(nt):
    un = u.copy()
#    for i in range(1, nx):
#        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

    u[1:] = un[1:] - c * dt / dx * (un[1:] - un[0:-1])

pyplot.plot(numpy.linspace(0, 2, nx), u)
pyplot.savefig("Step1-graph_2")

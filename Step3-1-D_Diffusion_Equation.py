import numpy
from matplotlib import pyplot

nx = 41
dx = 2 / (nx - 1)
nt = 20
nu = .3
sigma = .2
dt = sigma * dx**2 / nu

#initial conditions
u = numpy.ones(nx)
u[int(.5 / dx) : int(1 / dx + 1)] = 2
pyplot.plot(numpy.linspace(0, 2, nx), u)
pyplot.savefig("Step3-graph_1")

un = numpy.ones(nx)

for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx ** 2 * (un[i + 1] - 2 * un[i] + un[i - 1])

pyplot.plot(numpy.linspace(0, 2, nx), u)
pyplot.savefig("Step3-graph_2")
print(*u)

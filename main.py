# Description: Main file to run the optimization algorithm
# Importing the necessary packages
import matplotlib.pyplot as plt
from RUN import RUN
from BenchmarkFunctions import BenchmarkFunctions

# Number of Population
nP = 50

# Name of the test function, range from F1-F14
Func_name = 'F1'

# Maximum number of iterations
MaxIt = 500

# Load details of the selected benchmark function
lb, ub, dim, fobj = BenchmarkFunctions(Func_name)

Best_fitness, BestPositions, Convergence_curve = RUN(nP, MaxIt, lb, ub, dim, fobj)

# Draw the convergence curve
plt.figure()
plt.semilogy(Convergence_curve, color='r', linewidth=2)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.grid(False)
plt.legend(['RUN'])
plt.show()

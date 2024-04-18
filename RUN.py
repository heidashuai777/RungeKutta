import numpy as np
from scipy.optimize import minimize
from initialization import initialization
from RungeKutta import RungeKutta
# A function to determine a random number
# with uniform distribution
def Unifrnd(a, b, c, dim):
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    z = mu + sig * (2 * np.random.rand(c, dim) - 1)
    return z

# A function to determine thress random indices of solutions
def RndX(nP, i):
    Qi = np.random.permutation(nP)
    Qi = Qi[Qi != i]
    A = Qi[0]
    B = Qi[1]
    C = Qi[2]
    return A, B, C

def RUN(nP, MaxIt, lb, ub, dim, fobj):
    Cost = np.zeros(nP)  # Record the Fitness of all Solutions
    X = initialization(nP, dim, ub, lb)  # Initialize the set of random solutions
    Xnew2 = np.zeros(dim)

    Convergence_curve = np.zeros(MaxIt)

    for i in range(nP):
        Cost[i] = fobj(X[i, :])  # Calculate the Value of Objective Function

    ind = np.argmin(Cost)
    Best_Cost = Cost[ind]  # Determine the Best Solution
    Best_X = X[ind, :]

    Convergence_curve[0] = Best_Cost

    # Main Loop of RUN
    it = 1  # Number of iterations
    while it < MaxIt:
        it += 1
        f = 20 * np.exp(-(12 * (it / MaxIt)))  # (Eq.17.6)
        Xavg = np.mean(X, axis=0)  # Determine the Average of Solutions
        SF = 2 * (0.5 - np.random.rand(nP)) * f  # Determine the Adaptive Factor (Eq.17.5)

        for i in range(nP):
            ind_l = np.argmin(Cost)
            min_value = Cost[ind_l]
            lBest = X[ind_l, :]

            A, B, C = RndX(nP, i)  # Determine Three Random Indices of Solutions
            ind1 = np.argmin(Cost[[A, B, C]])

            # Determine Delta X (Eqs. 11.1 to 11.3)
            gama = np.random.rand() * (X[i, :] - np.random.rand(dim) * (ub - lb)) * np.exp(-4 * it / MaxIt)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gama)
            DelX = 2 * np.random.rand(dim) * np.abs(Stp)

            # Determine Xb and Xw for using in Runge Kutta method
            if Cost[i] < Cost[ind1]:
                Xb = X[i, :]
                Xw = X[ind1, :]
            else:
                Xb = X[ind1, :]
                Xw = X[i, :]

            SM = RungeKutta(Xb, Xw, DelX)  # Search Mechanism (SM) of RUN based on Runge Kutta Method

            L = np.random.rand(dim) < 0.5
            Xc = L * X[i, :] + (1 - L) * X[A, :]  # (Eq. 17.3)
            Xm = L * Best_X + (1 - L) * lBest  # (Eq. 17.4)

            vec = [1, -1]
            flag = np.floor(2 * np.random.rand(dim) + 1)
            r = vec[int(flag.item(0)) - 1]  # An Interger number

            g = 2 * np.random.rand()
            mu = 0.5 + 0.1 * np.random.randn(dim)

            # Determine New Solution Based on Runge Kutta Method (Eq.18)
            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * (SM) + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * (SM) + mu * (X[A, :] - X[B, :])

            # Check if solutions go outside the search space and bring them back
            FU = Xnew > ub
            FL = Xnew < lb
            Xnew = (Xnew * (~(FU + FL))) + ub * FU + lb * FL
            CostNew = fobj(Xnew)

            if CostNew < Cost[i]:
                X[i, :] = Xnew
                Cost[i] = CostNew

            # Enhanced solution quality (ESQ) (Eq. 19)
            if np.random.rand() < 0.5:
                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                r = np.floor(Unifrnd(-1, 2, 1, 1))

                u = 2 * np.random.rand(dim)
                w = Unifrnd(0, 2, 1, dim).flatten()  # (Eq.19-1)

                A, B, C = RndX(nP, i)
                Xavg = (X[A, :] + X[B, :] + X[C, :]) / 3  # (Eq.19-2)

                beta = np.random.rand(dim)
                Xnew1 = beta * (Best_X) + (1 - beta) * (Xavg)  # (Eq.19-3)

                for j in range(dim):
                    if w[j] < 1:
                        Xnew2[j] = Xnew1[j] + r * w[j] * np.abs((Xnew1[j] - Xavg[j]) + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[j] * np.abs((u[j] * Xnew1[j] - Xavg[j]) +
                                                                              np.random.randn())

                FU = Xnew2 > ub
                FL = Xnew2 < lb
                Xnew2 = (Xnew2 * (~(FU + FL))) + ub * FU + lb * FL
                CostNew = fobj(Xnew2)

                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                else:
                    if np.random.rand() < w[np.random.randint(dim)]:
                        SM = RungeKutta(X[i, :], Xnew2, DelX)
                        Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (
                                    SM + (2 * np.random.rand(dim) * Best_X - Xnew2))  # (Eq. 20)

                        FU = Xnew > ub
                        FL = Xnew < lb
                        Xnew = (Xnew * (~(FU + FL))) + ub * FU + lb * FL
                        CostNew = fobj(Xnew)

                        if CostNew < Cost[i]:
                            X[i, :] = Xnew
                            Cost[i] = CostNew

        # End of ESQ
        # Determine the Best Solution
        if Cost[i] < Best_Cost:
            Best_X = X[i, :]
            Best_Cost = Cost[i]

        # Save Best Solution at each iteration
        Convergence_curve[it - 1] = Best_Cost
        print('it : {}, Best Cost = {}'.format(it, Convergence_curve[it - 1]))

    return Best_Cost, Best_X, Convergence_curve







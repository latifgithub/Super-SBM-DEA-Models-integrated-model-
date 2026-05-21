import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import pandas as pd
import time
#from get_data_set import get_data_set
import warnings

import scipy
print('SciPy version= ',scipy.__version__)


#Data sets

 



def get_data_set(table_num, m, s, n):



    
    """Load table data; Table 13 is for random data."""


    def random_dea_data_set(m, s, n, seed=42):
        np.random.seed(seed)
        X = np.random.uniform(10, 100, size=(m, n))
        Y = np.random.uniform(10, 100, size=(s, n))
        return X, Y



    if table_num == 1:
        X = np.array([[2, 2, 3, 1],
                      [1, 3, 4, 2]], dtype=float)
        Y = np.array([[1, 1, 1, 1]], dtype=float)

    elif table_num == 3:
        X = np.array([[4, 7, 8, 4, 2, 10, 12],
                      [3, 3, 1, 2, 4,  1,  1]], dtype=float)
        Y = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=float)

    elif table_num == 5:
        X = np.array([[80, 65, 83, 40, 52, 94],
                      [600, 200, 400, 1000, 600, 700],
                      [54, 97, 72, 75, 20, 36],
                      [8,  1,  4,  7,  3,  5]], dtype=float)
        Y = np.array([[90, 58, 60, 80, 72, 96],
                      [5,  1,  7, 10,  8,  6]], dtype=float)

    elif table_num == 7:
        X = np.array([[4, 6, 8, 8, 2],
                      [3, 3, 1, 1, 4]], dtype=float)
        Y = np.array([[2, 2, 6, 6, 1],
                      [3, 3, 2, 1, 4]], dtype=float)

    elif table_num == 9:
        X = np.array([[4, 7, 8, 4, 2, 10, 12],
                      [3, 3, 1, 2, 4,  1,  1]], dtype=float)
        Y = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=float)

    elif table_num == 11:
        # Data from Table 11. Data set from Tran et al. (2019)
        # DMUs: C1 to C50

        # Input matrix X: rows correspond to x1, x2, x3, x4
        X = np.array([
            [1, 2.33, 2.83, 4.82, 5.4, 5.58, 6.1, 6.37, 6.52, 7.45, 7.8, 8.06, 8.94, 8.96, 9.97, 10.49, 10.87, 10.9, 11.21, 11.94, 11.98, 12.53, 13.49, 13.94, 14.02, 14.77, 14.81, 15.13, 15.32, 16.32, 16.96, 17.98, 19.62, 20.76, 21.23, 24.78, 25.18, 28.34, 31.23, 31.78, 33.22, 41.65, 43.21, 49.08, 49.43, 65.94, 80.69, 82.45, 85.16, 101],
            [1.45, 1.22, 1.22, 3.24, 11.44, 8.18, 6.84, 16.38, 6.84, 9.53, 7.82, 6.16, 20.19, 3.58, 20.19, 7.4, 20.19, 7.06, 7.84, 3.81, 8.52, 7.06, 6.16, 3.24, 10.32, 24.91, 10.32, 10.32, 12.78, 8.07, 17.05, 14.36, 3.69, 50.49, 1, 15.59, 8.63, 3.92, 37.7, 5.71, 15.59, 3.69, 43.65, 24.23, 12.56, 101, 8.18, 52.74, 52.74, 13.01],
            [15.98, 17.29, 17.63, 16.74, 20.13, 17.47, 21.22, 19.34, 21.22, 19.84, 23.46, 19.78, 21.39, 19.33, 21.39, 11.3, 21.39, 19.76, 19.55, 18.72, 17.96, 13.23, 23.97, 11.9, 6.02, 26.84, 5.98, 20.49, 22.62, 20.58, 12.53, 21.22, 18.72, 33.44, 39.65, 24.23, 30.17, 9.32, 37.89, 17.59, 33.36, 66.88, 5.46, 5.2, 1, 82.96, 56.25, 28.09, 28.09, 101],
            [1, 1.57, 2.34, 3.7, 3.87, 6.51, 7.37, 10.4, 7.37, 8.49, 5.84, 6.8, 1.54, 3.99, 1.54, 8.52, 1.54, 6.07, 6.62, 3.66, 2.95, 5.74, 5.44, 4, 7.5, 20.38, 7.5, 13.74, 5.71, 7.2, 12.52, 12.61, 2.9, 29.87, 2.36, 10.35, 6.21, 5.36, 21.72, 6.3, 10.2, 4.8, 29.71, 24.12, 14.24, 101, 7.88, 55.69, 55.69, 11.17],
        ])  # Transpose to get shape (50, 4)

        # Output matrix Y: rows correspond to y1, y2, y3, y4, y5
        Y = np.array([
            [1.43, 1, 5.79, 2.34, 5.15, 5.7, 2.37, 4.63, 3.31, 7.02, 4.8, 3.3, 21.32, 3.04, 20.98, 2.68, 19.46, 19.74, 19.97, 10.54, 7.47, 4.46, 13.64, 11.73, 13.09, 24.6, 14.32, 25.73, 9.4, 3.56, 5.88, 2.34, 14.99, 35.37, 2.55, 20.75, 6.63, 9.82, 11.18, 2.26, 55.45, 10.17, 22.24, 20.63, 28.33, 18.37, 37.8, 70.36, 70.36, 57.64],
            [10.36, 1, 70.03, 4.1, 22.99, 30.58, 7.32, 19.12, 11.42, 14.3, 28.26, 13.08, 53.91, 16.46, 53.3, 3.66, 58.03, 100.17, 101, 46.43, 39.45, 9.98, 38.84, 3.33, 62.55, 72.8, 68.04, 70.81, 11.53, 8.15, 12.97, 6.26, 63.99, 82.5, 1.89, 88.81, 20.5, 16.73, 14.96, 1.22, 98.89, 47.98, 12.91, 23.11, 29.7, 11.75, 55.08, 101, 101, 69.53],
            [1, 97.61, 85.75, 101, 101, 101, 101, 101, 94.22, 82.36, 101, 94.22, 89.14, 101, 89.14, 97.61, 94.22, 101, 101, 90.83, 101, 97.61, 101, 85.75, 89.14, 94.22, 89.14, 101, 101, 101, 84.05, 94.22, 101, 94.22, 77.27, 101, 101, 89.14, 77.27, 89.14, 89.14, 4.39, 101, 101, 101, 89.14, 94.22, 89.14, 94.22, 101],
            [2.35, 1.72, 3.75, 1, 12.92, 6.52, 13.28, 12.39, 13.29, 18.91, 8.18, 11.65, 25.41, 7.95, 25.41, 38.93, 25.41, 12.3, 12.28, 14.45, 10.89, 24.64, 22.14, 22.62, 12.89, 20.07, 12.8, 23.69, 47.47, 22.24, 16.96, 25.27, 14.46, 31.38, 45.62, 14.6, 18.64, 36.17, 32.76, 56.72, 38.29, 12.58, 86.52, 59.46, 64.46, 85.56, 46.08, 101, 101, 56.52],
            [37.32, 7.98, 57.25, 20.88, 44.77, 46.99, 26.76, 32.01, 29.27, 1, 27.82, 25.51, 101, 23.32, 93.23, 26.6, 75.83, 53.93, 53.25, 34.7, 34.64, 36.63, 37.5, 42.23, 61.3, 57.12, 56.53, 70.42, 53.22, 22.58, 27.7, 19.63, 28.96, 70.81, 19.75, 33.04, 25.09, 33.52, 43.85, 18.39, 51.49, 20.86, 39.85, 32.34, 45.07, 26.03, 42.42, 44.64, 43.33, 26.79],
        ])  # Transpose to get shape (50, 5), dtype=float)


    elif table_num >= 20:
        X, Y = random_dea_data_set(m, s, n, seed=42)
        X = X.astype(float)
        Y = Y.astype(float)

    else:
        raise ValueError(f"Invalid table number: {table_num}")
    
    return X, Y




# ==================== PARAMETERS ====================

unified_M_BIG= 1e+4
M_BIG = 1e+7
M_SMALL = 1e-6
EPSILON = 1e-6

NUM_INPUTS = 10
NUM_OUTPUTS = 15
NUM_DMUS = 400
RANDOM_SEED = 42

SHOW_DETAILS = True
OUTPUT_FILE = 'dea_results.xlsx'
# ====================================================

def random_dea_data_set(m, s, n, seed):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(1, 100, size=(m, n))
    Y = np.random.uniform(1, 100, size=(s, n))
    return X, Y


################################# SBM-SSBM and Enhanced SBM-SSBM  ########################
      # SBM-SSBM andmy modification on  SBM-SSBM
#####################################################################################

def solve_SBM_scipy(X, Y, p):
    m, n = X.shape
    s = Y.shape[0]
    xp = X[:, p]
    yp = Y[:, p]
    
    n_vars = 1 + m + s + n
    c = np.zeros(n_vars)
    c[0] = 1
    c[1:1+m] = -1 / (m * xp)
    
    A_eq = []
    b_eq = []
    
    eq_row = np.zeros(n_vars)
    eq_row[0] = 1
    eq_row[1+m:1+m+s] = 1 / (s * yp)
    A_eq.append(eq_row)
    b_eq.append(1)
    
    for i in range(m):
        eq_row = np.zeros(n_vars)
        eq_row[0] = -xp[i]
        eq_row[1+i] = 1
        eq_row[1+m+s:] = X[i, :]
        A_eq.append(eq_row)
        b_eq.append(0)
    
    for r in range(s):
        eq_row = np.zeros(n_vars)
        eq_row[0] = -yp[r]
        eq_row[1+m+r] = -1
        eq_row[1+m+s:] = Y[r, :]
        A_eq.append(eq_row)
        b_eq.append(0)
    
    bounds = [(0, None)] * n_vars
    result = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')
    
    if result.success:
        return result.fun, result.x[1+m+s:]
    return np.nan, np.zeros(n)

def solve_SSBM_scipy(X, Y, p):
    m, n = X.shape
    s = Y.shape[0]
    xp = X[:, p]
    yp = Y[:, p]
    
    peer_idx = [i for i in range(n) if i != p]
    X_peer = X[:, peer_idx]
    Y_peer = Y[:, peer_idx]
    n_peer = len(peer_idx)
    
    n_vars = 1 + m + s + n_peer
    c = np.zeros(n_vars)
    c[0] = 1
    c[1:1+m] = 1 / (m * xp)
    
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    
    eq_row = np.zeros(n_vars)
    eq_row[0] = 1
    eq_row[1+m:1+m+s] = -1 / (s * yp)
    A_eq.append(eq_row)
    b_eq.append(1)
    
    for i in range(m):
        ub_row = np.zeros(n_vars)
        ub_row[0] = -xp[i]
        ub_row[1+i] = -1
        ub_row[1+m+s:] = X_peer[i, :]
        A_ub.append(ub_row)
        b_ub.append(0)
    
    for r in range(s):
        ub_row = np.zeros(n_vars)
        ub_row[0] = yp[r]
        ub_row[1+m+r] = -1
        ub_row[1+m+s:] = -Y_peer[r, :]
        A_ub.append(ub_row)
        b_ub.append(0)
    
    bounds = [(0, None)] * n_vars
    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                     bounds=bounds, method='highs')
    
    return max(result.fun, 1.0) if result.success else 1.0


 
def SBM_SSBM_algorithm(X, Y, epsilon, use_enhanced=True, tol=1e-8):
    X_full = X.copy()
    Y_full = Y.copy()

    n = X.shape[1]
    eff_scores = np.ones(n)
    IE = []

    # DMUs still pending evaluation
    to_evaluate = list(range(n))

    # DMUs still permitted in the reference set
    reference_alive = np.ones(n, dtype=bool)

    start_total = time.time()

    while to_evaluate:
        p = to_evaluate.pop(0)

        # Current reference set
        ref_idx = np.where(reference_alive)[0].tolist()

        # If p is not in the reference set, add it temporarily for its own evaluation
        if p not in ref_idx:
            ref_idx = sorted(ref_idx + [p])

        X_ref = X_full[:, ref_idx]
        Y_ref = Y_full[:, ref_idx]
        p_local = ref_idx.index(p)

        delta_p, lambda_opt = solve_SBM_scipy(X_ref, Y_ref, p_local)
        eff_scores[p] = delta_p

        active_global = []
        if use_enhanced:
            active_local = np.where(lambda_opt > tol)[0].tolist()
            active_global = [ref_idx[j] for j in active_local]
            IE.extend(active_global)
        else:
            if delta_p >= 1 - epsilon:
                IE.append(p)

        # -----------------------------
        # 1) Remove active DMUs from the evaluation set
        # -----------------------------
        if use_enhanced and active_global:
            active_set = set(active_global)
            to_evaluate = [j for j in to_evaluate if j not in active_set]

        # -----------------------------
        # 2) Remove dominated DMUs from the reference set
        # -----------------------------
        if delta_p >= 1 - epsilon:
            x_o = X_full[:, p]
            y_o = Y_full[:, p]

            for j in range(n):
                if j == p:
                    continue
                if not reference_alive[j]:
                    continue

                dominated = (
                    np.all(X_full[:, j] >= x_o - tol) and
                    np.all(Y_full[:, j] <= y_o + tol)
                )

                if dominated:
                    reference_alive[j] = False

    IE = sorted(set(IE))
    super_eff = eff_scores.copy()

    for i in IE:
        super_eff[i] = solve_SSBM_scipy(X_full, Y_full, i)

    total_time = time.time() - start_total
    return eff_scores, IE, super_eff, total_time




def SBM_SSBM_algorithm1(X, Y, epsilon, use_enhanced=True):
    n = X.shape[1]
    IE = []
    J = list(range(n))
    eff_scores = np.ones(n)
    
    start_total = time.time()
    
    while J:
        p = J[0]
        delta_p, lambda_opt = solve_SBM_scipy(X, Y, p)
        eff_scores[p] = delta_p
        
        if use_enhanced:
            active_idx = np.where(lambda_opt > 1e-8)[0].tolist()
            IE.extend(active_idx)
        else:
            if delta_p >= 1 - epsilon:
                IE.append(p)
        
        J = [j for j in J if j not in IE and j != p]
    
    super_eff = eff_scores.copy()
    IE = list(set(IE))
    
    for i in IE:
        super_eff[i] = solve_SSBM_scipy(X, Y, i)
    
    total_time = time.time() - start_total
    return eff_scores, IE, super_eff, total_time

def New_unified_SBM_SuperSBM_scipy(X, Y, p):
    m, n = X.shape
    s = Y.shape[0]
    x_p = X[:, p]
    y_p = Y[:, p]
    
    num_vars = 1 + m + s + (n-1) + 1
    c = np.zeros(num_vars)
    c[0] = 1
    c[1:1+m] = -1/(m * x_p)
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    eq1 = np.zeros(num_vars)
    eq1[0] = 1
    eq1[1+m:1+m+s] = 1/(s * y_p)
    A_eq.append(eq1)
    b_eq.append(1)
    
    X_excl = np.delete(X, p, axis=1)
    Y_excl = np.delete(Y, p, axis=1)
    
    for i in range(m):
        row = np.zeros(num_vars)
        row[0] = -x_p[i]
        row[1+i] = 1
        row[1+m+s:1+m+s+(n-1)] = X_excl[i, :]
        A_ub.append(row)
        b_ub.append(0)
    
    for r in range(s):
        row = np.zeros(num_vars)
        row[0] = y_p[r]
        row[1+m+r] = 1
        row[1+m+s:1+m+s+(n-1)] = -Y_excl[r, :]
        A_ub.append(row)
        b_ub.append(0)
    
    for i in range(m):
        row = np.zeros(num_vars)
        row[1+i] = -1
        row[-1] = -unified_M_BIG
        A_ub.append(row)
        b_ub.append(0)
    
    for i in range(m):
        row = np.zeros(num_vars)
        row[1+i] = 1
        row[-1] = unified_M_BIG
        A_ub.append(row)
        b_ub.append(unified_M_BIG)
    
    for r in range(s):
        row = np.zeros(num_vars)
        row[1+m+r] = -1
        row[-1] = -unified_M_BIG
        A_ub.append(row)
        b_ub.append(0)
    
    for r in range(s):
        row = np.zeros(num_vars)
        row[1+m+r] = 1
        row[-1] = unified_M_BIG
        A_ub.append(row)
        b_ub.append(unified_M_BIG)
    
    bounds = Bounds(lb=[-np.inf]*num_vars, ub=[np.inf]*num_vars)
    bounds.lb[0] = 0
    bounds.lb[1+m+s:1+m+s+(n-1)] = 0
    bounds.lb[-1] = 0
    bounds.ub[-1] = 1
    
    constraints = [LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)),
                   LinearConstraint(np.array(A_eq), np.array(b_eq), np.array(b_eq))]
    
    integrality = np.zeros(num_vars)
    integrality[-1] = 1
    
    result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    return result.fun if result.success else np.nan


################################# small unified binary MILP  ########################
      # my modification on 
      #A modified slacks-based measure of efficiency in data envelopment analysis
      #Kaoru Tone a , Mehdi Toloo b , ∗, Mohammad Izadikhah c
      #2021
#####################################################################################


def NEw_Unified_SBM_SuperSBM_algorithm(X, Y, epsilon):
    n = X.shape[1]
    scores = np.zeros(n)
    start_time = time.time()
    
    for p in range(n):
        scores[p] = New_unified_SBM_SuperSBM_scipy(X, Y, p)
    
    elapsed_time = time.time() - start_time
    n_efficient = np.sum(scores >= 1 - epsilon)
    return scores, n_efficient, elapsed_time



def Tone_unified_SBM_SuperSBM_scipy(X, Y, p):
    m, n = X.shape
    s = Y.shape[0]
    x_p = X[:, p]
    y_p = Y[:, p]
    
    num_vars = 1 + m + s + (n-1) + 1
    c = np.zeros(num_vars)
    c[0] = 1
    c[1:1+m] = -1/(m * x_p)
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # Normalization constraint
    eq1 = np.zeros(num_vars)
    eq1[0] = 1
    eq1[1+m:1+m+s] = 1/(s * y_p)
    A_eq.append(eq1)
    b_eq.append(1)
    
    X_excl = np.delete(X, p, axis=1)
    Y_excl = np.delete(Y, p, axis=1)
    
    # Input constraints (converted to equality form)
    for i in range(m):
        row = np.zeros(num_vars)
        row[0] = -x_p[i]
        row[1+i] = 1
        row[1+m+s:1+m+s+(n-1)] = X_excl[i, :]
        A_eq.append(row)
        b_eq.append(0)
    
    # Output constraints (converted to equality form)
    for r in range(s):
        row = np.zeros(num_vars)
        row[0] = y_p[r]
        row[1+m+r] = 1
        row[1+m+s:1+m+s+(n-1)] = -Y_excl[r, :]
        A_eq.append(row)
        b_eq.append(0)
    
    # Big-M constraints for input slacks
    for i in range(m):
        row = np.zeros(num_vars)
        row[1+i] = -1
        row[-1] = -unified_M_BIG
        A_ub.append(row)
        b_ub.append(0)
    
    for i in range(m):
        row = np.zeros(num_vars)
        row[1+i] = 1
        row[-1] = unified_M_BIG
        A_ub.append(row)
        b_ub.append(unified_M_BIG)
    
    # Big-M constraints for output slacks
    for r in range(s):
        row = np.zeros(num_vars)
        row[1+m+r] = -1
        row[-1] = -unified_M_BIG
        A_ub.append(row)
        b_ub.append(0)
    
    for r in range(s):
        row = np.zeros(num_vars)
        row[1+m+r] = 1
        row[-1] = unified_M_BIG
        A_ub.append(row)
        b_ub.append(unified_M_BIG)
    
    # Variable bounds
    bounds = Bounds(lb=[-np.inf]*num_vars, ub=[np.inf]*num_vars)
    bounds.lb[0] = 0  # theta >= 0
    bounds.lb[1+m+s:1+m+s+(n-1)] = 0  # lambda >= 0
    bounds.lb[-1] = 0  # delta >= 0
    bounds.ub[-1] = 1  # delta <= 1
    
    constraints = [LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)),
                   LinearConstraint(np.array(A_eq), np.array(b_eq), np.array(b_eq))]
    
    # Binary variable for delta
    integrality = np.zeros(num_vars)
    integrality[-1] = 1
    
    result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    return result.fun if result.success else np.nan


#########################################################
      #Kaoru Tone a , Mehdi Toloo b , ∗, Mohammad Izadikhah c
      #2021
#####################################################################################


def Tone_Unified_SBM_SuperSBM_algorithm(X, Y, epsilon):
    n = X.shape[1]
    scores = np.zeros(n)
    start_time = time.time()
    
    for p in range(n):
        scores[p] = Tone_unified_SBM_SuperSBM_scipy(X, Y, p)
    
    elapsed_time = time.time() - start_time
    n_efficient = np.sum(scores >= 1 - epsilon)
    return scores, n_efficient, elapsed_time






#######################################

# Main

#######################################
TABLES = [20, 21, 22,23,24,25] # for generating random data set in different size

TABLES = [1, 3, 5, 7,9, 11]

NUM_INPUTS = 8 # for generating random data
NUM_OUTPUTS = 10 # for generating random data
NUM_DMUS = {20: 100, 21: 120, 22: 150, 23: 300, 24: 500, 25: 700} # for generating random data

#NUM_DMUS = {20: 150, 21: 200, 22: 300, 23: 500, 24: 700, 25: 1000} # for generating random data

METHODS = ['Simple-SupSBM', 'Tone-Unified', 'New-Unified']  


all_times = {method: [] for method in METHODS}
all_results = {}

for table_num in TABLES:
    print(f'\n{"="*90}')
    print(f'Processing Table {table_num}'.center(90))
    print("="*90)
    
    if table_num >= 20:
        X, Y = get_data_set(table_num, NUM_INPUTS, NUM_OUTPUTS, NUM_DMUS[table_num])
    else:
        X, Y = get_data_set(table_num, 2, 2, 2)
    
    # Store dimensions for this table
    all_results[table_num] = {
        'm': X.shape[0],
        's': Y.shape[0],
        'n': NUM_INPUTS if table_num >= 20 else X.shape[1]        
    }

    results = {}
    
    if 'Simple-SupSBM' in METHODS:
        eff_scores_sim, IE_sim, super_eff_sim, time_sim = SBM_SSBM_algorithm(X, Y, EPSILON, use_enhanced=False)
        results['Simple-SupSBM'] = {'scores': super_eff_sim, 'efficient': IE_sim, 'time': time_sim}
        all_times['Simple-SupSBM'].append(time_sim)

    if 'Tone-Unified' in METHODS:
        scores_unified, n_eff_unified, time_unified = Tone_Unified_SBM_SuperSBM_algorithm(X, Y, EPSILON)
        results['Tone-Unified'] = {'scores': scores_unified, 'efficient': np.where(scores_unified >= 1 - EPSILON)[0].tolist(), 'time': time_unified}
        all_times['Tone-Unified'].append(time_unified)

    
    if 'New-Unified' in METHODS:
        scores_unified, n_eff_unified, time_unified = NEw_Unified_SBM_SuperSBM_algorithm(X, Y, EPSILON)
        results['New-Unified'] = {'scores': scores_unified, 'efficient': np.where(scores_unified >= 1 - EPSILON)[0].tolist(), 'time': time_unified}
        all_times['New-Unified'].append(time_unified)
    
    
    all_results[table_num].update(results)
    
    # Find fastest method for this table
    times = [(m, results[m]['time']) for m in METHODS if m in results]
    fastest = min(times, key=lambda x: x[1])
    
    # Display results table
    print('\nRESULTS SUMMARY')
    print('-'*90)
    
    table_data = []
    for method in METHODS:
        if method not in results:
            continue
        eff_count = np.sum(results[method]['scores'] >= 1 - EPSILON)
        max_diff = 0 if method == 'Simple-SupSBM' else np.max(np.abs(results[method]['scores'] - results['Simple-SupSBM']['scores']))
        
        table_data.append({
            'Method': method,
            'Time(s)': f"{results[method]['time']:.4f}",
            'Speedup': f"{fastest[1]/results[method]['time']:.2f}X" if method != fastest[0] else "1.00X",
            'Eff#': eff_count,
            'Max|Δ|': f"{max_diff:.6f}"
        })
    
    df_table = pd.DataFrame(table_data)
    print(df_table.to_string(index=False))
    
    # Detailed scores table
    if SHOW_DETAILS:
        print('\nDETAILED SCORES (First 10 DMUs)')
        print('-'*90)
        
        n_show = min(10, len(results['Simple-SupSBM']['scores']))
        scores_table = {'DMU': [f'DMU{i+1}' for i in range(n_show)]}
        for method in METHODS:
            if method in results:
                scores_table[method] = [f"{s:.4f}" for s in results[method]['scores'][:n_show]]
        
        df_scores = pd.DataFrame(scores_table)
        print(df_scores.to_string(index=False))
    
    # Save to Excel
    OUTPUT_FILE = f'dea_results_table{table_num}.xlsx'
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        pd.DataFrame(table_data).to_excel(writer, sheet_name='Summary', index=False)
        
        for method, data in results.items():
            df = pd.DataFrame({
                'DMU': [f'DMU{i+1}' for i in range(len(data['scores']))],
                'Score': data['scores'],
                'Efficient': data['scores'] >= 1 - EPSILON
            })
            df.to_excel(writer, sheet_name=method, index=False)
    
    print(f'\n✓ Results saved to {OUTPUT_FILE}')

# Final comparison table with dimensions
print('\n' + '='*90)
print('TIMING COMPARISON ACROSS ALL TABLES'.center(90))
print('='*90)

time_comparison = {
    'Table': TABLES,
    'm': [all_results[t]['m'] for t in TABLES],
    's': [all_results[t]['s'] for t in TABLES],
    'n': [all_results[t]['n'] for t in TABLES]
    
}
for method in METHODS:
    time_comparison[method] = [f"{t:.4f}" for t in all_times[method]]

df_time = pd.DataFrame(time_comparison)
print(df_time.to_string(index=False))
print('='*90)

# Save timing comparison
with pd.ExcelWriter('dea_timing_comparison.xlsx', engine='openpyxl') as writer:
    df_time.to_excel(writer, sheet_name='Timing', index=False)

print('\n✓ Timing comparison saved to dea_timing_comparison.xlsx')


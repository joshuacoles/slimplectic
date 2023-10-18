from __future__ import division, print_function
import sympy
from sympy import *
import numpy
import scipy.optimize
from nptyping import NDArray, Float, Shape
from typing import Any, Callable


def flatten_table(table: list[list]) -> list:
    return [v for vec in table for v in vec]


def generate_collocation_points(r: int, precision: int = 20) -> tuple[list[float], list[float], list[list[float]]]:
    """
    Gives the Collocation points, weights and derivative matrix for the Galerkin-Gauss-Lobatto Variational Integrator as
    floats evaluated to precision.

    Args:
        r (int): the number of intermediate points.
        precision (int): (default 20), the precision of the resolved collocation points.

    Returns:
        - collocation_points[r+2], The numerical x - collocation points array,
        - point_weights[r+2], The numerical weights
        - derivative_matrix[r+2, r+2], The derivative matrix (as a function of x)

    All evaluated for a system with r intermediate points, i.e. (r+2) total collocation points, up to arbitrary
    precision, which is the number of sig. figs. in decimal representation
    """

    # Convenience lambda to evaluate a sympy expression to given precision
    nprec = lambda x: N(x, precision)
    list_nprec = lambda l: list(map(nprec, l))

    # Set polynomial order n for a given r intermediate points or r + 2 total collocation points
    n = r + 1

    # Find collocation points for the Gauss-Lobatto quadrature
    x = symbols('x')

    collocation_points = list_nprec(polys.polytools.real_roots(
        # Question: What is this function?
        (x ** 2 - 1) * diff(legendre(n, x), x),
        multiple=True
    ))

    # Determine the weight functions
    point_weights = list_nprec(
        [2 / (n * (n + 1) * (legendre(n, xj)) ** 2) for xj in collocation_points]
    )

    # Determine the derivative matrix using the grid points evaluated to the right position
    # Generate a 2D array of zeros, we will fill it in the below
    derivative_matrix = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    # Fills the derivative matrix. This formula is given by
    # \cite[Eq 5, p. 2]{tsangSLIMPLECTICINTEGRATORSVARIATIONAL2015}
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                if j == 0:
                    derivative_matrix[i][j] = nprec(-(1 / 4) * n * (n + 1))
                elif j == n:
                    derivative_matrix[i][j] = nprec((1 / 4) * n * (n + 1))
                else:
                    derivative_matrix[i][j] = S.Zero
            else:
                derivative_matrix[i][j] = nprec(legendre(n, collocation_points[i]) / (
                        legendre(n, collocation_points[j]) * (collocation_points[i] - collocation_points[j])))

    return collocation_points, point_weights, derivative_matrix


def q_Generate_pm(qlist: list):
    """
    Generates lists of sympy symbols for the q_+ and q_- terms. These are used in the non-conservative lagrangian.

    Args:
        qlist: The list of degrees of freedom

    Returns:
        1. qplist: The list of symbols of the form q_+
        2. qmlist: The list of symbols of the form q_-
    """

    qplist = [Symbol(repr(q) + '_+') for q in qlist]
    qmlist = [Symbol(repr(q) + '_-') for q in qlist]
    return qplist, qmlist


def Gen_pi_list(qlist: list):
    """
    Generate_pi generates the symbol list for the non-conservative discrete momenta pi

    Output: (pi_n_list, pi_np1_list)
    pi_n_list[dof] - list of symbols for the current pi_n
    pi_np1_list[dof] - list of symbols for the next pi_n+1
    Input:
    qlist[dof] - the 1-d list of symbols that you want make
                 momenta for
    """

    pi_n_list = [Symbol("\pi_" + repr(q) + "^{[n]}", real=True) for q in qlist]
    pi_np1_list = [Symbol("\pi_" + repr(q) + "^{[n+1]}", real=True) for q in qlist]
    return pi_n_list, pi_np1_list


def Physical_Limit(q_list: list, q_p_list: list, q_m_list: list, expression):
    """
    Generates a Sympy object derived from expression, under the Physical Limit. This involves taking,

    - q_- -> 0
    - q_+ -> q

    This is implemented through a call to Sympy `subs`.

    Note: We assume that the q_list, q_p_list, and q_m_list share the same ordering.

    If you are passing in q_tables please flatten them using something like:

        q_list = [qval for qvallist in qtable for qval in qvallist]

    Args:
        q_list: The original degrees of freedom of the system.
        q_p_list: The q_+ degrees of freedom.
        q_m_list: The q_0 degrees of freedom.
        expression: The expression to take into the Physical Limit.

    Returns:
        Expression substituted to be under the physical limit.
    """

    dof_count = len(q_list)

    sub_list = []
    for dof in range(dof_count):
        # q_+ -> q
        sub_list.append((q_p_list[dof], q_list[dof]))

        # q_- -> 0
        sub_list.append((q_m_list[dof], 0))

    # The expression substituted to be under the physical limit.
    return expression.subs(sub_list)


def GGL_q_Collocation_Table(qlist: list, collocation_point_count: int):
    """
    Generates the symbol table to evaluate the degrees of freedom (in qlist) across the collocation points (number given
    by `collocation_point_count`). This generates a table with 1 row per q in qlist, and columns given by:

        - q^[n], the initial point
        - q^(i) for each internal collocation point
        - q^[n + 1], the final point

    Note that the first and last collocation points are the the initial and the final point.

    Args:
        qlist: The list of degrees of freedom
        collocation_point_count: The number of collocation points between the final and initial point

    Returns:
        The symbol table as described above, (len(qlist), collocation_point_count)
    """

    form_row = lambda q: [
        # The initial point of the quadrature
        Symbol("{" + repr(q) + "^{[n]}}"),

        # The interior points (starting from index)
        *[Symbol("{" + repr(q) + "^{(" + repr(i) + ")}}") for i in range(1, collocation_point_count - 1)],

        # The final point
        Symbol("{" + repr(q) + "^{[n + 1]}}"),
    ]

    return [form_row(q) for q in qlist]


def DM_Sum(DMvec: list, qlist: list):
    """
    Helper function to matrix dot product the DM matrix with a qvector
    Assumes that DMVec is the same length as qlist
    """
    # TODO: To be replaced with dot product. I think the inputs are sympy expressions
    sum = 0
    for j in range(len(DMvec)):
        sum += DMvec[j] * qlist[j]
    return sum


def GGL_Gen_Ld(
        tsymbol: sympy.Symbol,
        q_list: list,
        qprime_list: list,
        L: sympy.Expr,
        ddt,
        r: int,
        paramlist: list[tuple[sympy.Expr, sympy.Expr]] = [],
        precision: int = 20
):
    """
    Generates the discrete Lagrangian for use in determining the GGL variational integrator.

    Args:
        tsymbol: the symbol used for the explicit time dependence of the Lagrangian
        q_list: list of sympy variables (not including time derivatives),
        qprime_list: list of sympy variables for the time derivatives, in corresponding order to q_list.
        L: the algebraic expression for the continuous conservative Lagrangian in terms of q_list and qprime_list
           variables L(t, q, dotq).
        ddt: symbol for the time step.
        r: number of intermediate quadrature steps.
        paramlist: the constant parameter substitution list for evaluations
        precision: precision for which evaluations occur. This should be higher than machine precision for for best
                   results.

    Returns:
        (Ld, q_Table), where Ld is the algebraic expression for the discrete Lagrangian, and q_Table is a 2D array of
        shape (len(q_list), r + 2) containing the sympy symbols for qs at each quadrature point.
    """

    # First compute collocation points, weights and derivative matrix for the quadrature.
    collocation_points, collocation_point_weights, derivative_matrix = generate_collocation_points(r, precision)

    # Next generate substitution tables for evaluating the Lagrangian at each collocation points.

    # Create q_Table for all the algebraic variables based on q_list. Will be of shape (len(q_list), r+2) having rows:
    # [[..],..,[qx_i0,...,qx_i(r+2)],..,[..]]
    q_Table = GGL_q_Collocation_Table(q_list, r + 2)

    # Create list of times for evaluating L
    t_list = [tsymbol + 0.5 * (1 + collocation_point) * ddt for collocation_point in collocation_points]

    # Create dphidt_Table for the algebraic form of dPhi/dt
    # where Phi is the polynomial
    # interpolation of q over the quadrature points
    # dphidt_Table[len(q_list)][r+2]
    # Make sure to multiply by dx/dt = 2/ddt TODO: Why
    dphidt_Table = [
        [DM_Sum(DMvec, qs) * 2 / ddt for DMvec in derivative_matrix]
        for qs in q_Table
    ]

    # Evaluate Ld which is the weighted sum over each point, using the global paramlist substitutions as well as local
    # substitutions for each quadrature location according to the q_Table, dphidt_Table, and t_list generated above.
    # TODO: This will become evaluation of functions and normal sums in JAX
    Ld = 0
    for i in range(r + 2):
        local_substitutions = []
        for j in range(len(q_list)):
            local_substitutions.append((q_list[j], q_Table[j][i]))
            local_substitutions.append((qprime_list[j], dphidt_Table[j][i]))
            local_substitutions.append((tsymbol, t_list[i]))

        Ld += 0.5 * ddt * collocation_point_weights[i] * L.subs(local_substitutions).subs(paramlist)

    return Ld, q_Table


def GGL_Gen_Kd(
        tsymbol: sympy.Symbol,
        q_p_list: list,
        q_m_list: list,
        qprime_p_list: list,
        qprime_m_list: list,
        K: sympy.Expr,
        ddt,
        r: int,
        paramlist: list[tuple[sympy.Expr, sympy.Expr]] = [],
        precision=20
):
    """
    Generates the discrete non-conserative Lagrangian $K$ for use in determining the GGL variational integrator.

    Args:
        tsymbol: the symbol used for the explicit time dependence of the Lagrangian
        q_p_list: list of sympy variables (not including time derivatives) for the q_+ doubled dof
        q_m_list: list of sympy variables (not including time derivatives) for the q_- doubled dof, in corresponding
                  order.
        qprime_p_list: list of sympy variables for the q_+ doubled dof time derivatives, in corresponding order.
        qprime_m_list: list of sympy variables for the q_- doubled dof time derivatives, in corresponding order.
        K: the algebraic expression for the continuous non-conservative Lagrangian in terms of q_list and qprime_list
           variables. L(t, q, dotq).
        ddt: symbol for the time step.
        r: number of intermediate quadrature steps.
        paramlist: the constant parameter substitution list for evaluations
        precision: precision for which evaluations occur. This should be higher than machine precision for for best
                   results.

    Returns:
        (Kd, q_p_Table, q_m_Table), where Kd is the algebraic expression for the discrete non-conservative Lagrangian,
        and q_p_Table and q_m_table is a 2D array of shape (len(q_list), r + 2) containing the sympy symbols for q_{p,m}
        at each quadrature point.
    """

    # First compute collocation points, weights and derivative matrix for the quadrature.
    collocation_points, collocation_point_weights, derivative_matrix = generate_collocation_points(r, precision)

    # Next generate substitution tables for evaluating the Lagrangian at each collocation points.

    # Create q_{p/m}_Table for all the algebraic variables based on q_list
    # q_{p/m}_Table[len(qlist)][r+2]
    # [[..],..,[qx_i0,...,qx_i(r+2)],..,[..]]
    q_p_Table = GGL_q_Collocation_Table(q_p_list, r + 2)
    q_m_Table = GGL_q_Collocation_Table(q_m_list, r + 2)

    # Create dphidt_{p/m}_Table for the algebraic form of dPhi/dt where
    # Phi is the polynomial interpolation of q over the quadrature points
    # dphidt_{p/m}_Table[len(qlist)][r+2]
    # Make sure to multiply by dx/dt = 2/ddt
    dphidt_p_Table = [
        [DM_Sum(DMvec, qs) * 2 / ddt for DMvec in derivative_matrix]
        for qs in q_p_Table
    ]

    dphidt_m_Table = [
        [DM_Sum(DMvec, qs) * 2 / ddt for DMvec in derivative_matrix]
        for qs in q_m_Table
    ]

    # Evaluate Kd which is the weighted sum over each point, using the global paramlist substitutions as well as local
    # substitutions for each quadrature location according to the q_p_Table, q_m_Table, dphidt_p_Table, dphidt_m_Table.
    # NOTE: That we do not allow explict t dependence in the non-conservative Lagrangian.
    # TODO: This will become evalution of functions and normal sums in JAX

    # Create list of substitution pairs used at each quadrature location
    # Evaluate Ld which is the weighted sum over each point
    Kd = 0
    for i in range(r + 2):
        local_substitutions = []
        for j in range(len(q_p_list)):
            local_substitutions.append((q_p_list[j], q_p_Table[j][i]))
            local_substitutions.append((qprime_p_list[j], dphidt_p_Table[j][i]))

        for j in range(len(q_m_list)):
            local_substitutions.append((q_m_list[j], q_m_Table[j][i]))
            local_substitutions.append((qprime_m_list[j], dphidt_m_Table[j][i]))

        Kd += 0.5 * ddt * collocation_point_weights[i] * (K.subs(local_substitutions)).subs(paramlist)

    return Kd, q_p_Table, q_m_Table


def Gen_iter_EOM_List(q_Table: list[list], q_p_Table: list[list], q_m_Table: list[list], pi_n_list: list,
                      pi_np1_list: list, Ld: sympy.Expr, Kd: sympy.Expr, ddt: sympy.Symbol):
    """
    Generate the symbolic Equation of Motion Tables to be used for iteration.

    Inputs:
        q_Table: Table of sympy variables for dofs, shape (dof, r + 2).
        q_p_table: Table of sympy variables for + dofs, shape (dof, r + 2).
        q_m_Table: Table of sympy variables for - dofs, shape (dof, r + 2).

        pi_n_list: List of sympy variables for the non-conservative discrete momemta at current time, shape (dof,).
        pi_np1_list: List of sympy variables for the non-conservative discrete momemta at the next step, shape (dof,).

        Ld: Sympy expression for Ld
        Kd: Sympy expression for Kd
        ddt: Sympy symbol for the time step

    Returns:
        EOM_List[dof*(r+1)] - List of sympy equations of motion
        Here the equations of motion are assumed to be:

        (for i in [1..r])
              [ddt*Ld_Table[0][i] + Kd_Table[0][i],
               pi_1^[n] + ddt*(Ld_Table[0][0] + Kd_Table[0][0]),
               ...
               ddt*Ld_Table[dof][i] + Kd_Table[dof][i],
               pi_1^[n] + ddt*(Ld_Table[dof][0] + Kd_Table[dof][0])]

        The equation for pi_n+1 will be computed separately as it does
        not need to be iterated.
    """

    # Define symbol long lists for use with the Physical Limit function
    q_longlist = flatten_table(q_Table)
    q_p_longlist = flatten_table(q_p_Table)
    q_m_longlist = flatten_table(q_m_Table)

    # Create the Ld and Kd parts of the EOM
    # By taking the derivative wrt the appropriate dof and
    # taking the physical limit for Kd
    # TODO: JAX, DIFF operation

    Ld_EOM_Table = [
        [diff(Ld, q) for q in qvec]
        for qvec in q_Table
    ]

    # We differentiate Kd wrt q_- and evaluate it in the physical limit (q_+ -> q, q_- -> 0)
    Kd_EOM_Table = [
        [Physical_Limit(q_longlist, q_p_longlist, q_m_longlist, diff(Kd, q_m)) for q_m in qvec]
        for qvec in q_m_Table
    ]

    # Create Symbolic Equation of Motion Tables
    # We don't have the EOM for pi_n+1 since it will not need to be solved implicitly later
    EOM_List = []
    for i in range(len(Ld_EOM_Table)):
        for j in range(1, len(Ld_EOM_Table[0]) - 1):
            EOM_List.append(Ld_EOM_Table[i][j] + Kd_EOM_Table[i][j])

        EOM_List.append(pi_n_list[i] + Ld_EOM_Table[i][0] + Kd_EOM_Table[i][0])

    # print EOM_List
    return EOM_List


def compute_jacobian(expr_vec, var_vec):
    """
    Generate a table representing the Jacobian of expr_vec with respect to var_vec.

    Args:
        expr_vec: The expressions to be differentiated
        var_vec: The expressions with which to differentiate against.

    Returns:
        J - Jacobian table of sympy expressions where

            J[i][j] = d(expr_vec[i])/d(var_vec[j])
    """
    J = [
        [
            diff(expr_vec[i], var_vec[j])
            for j in range(len(var_vec))
        ]
        for i in range(len(expr_vec))
    ]

    return J


# TODO: currently unused, what does this do?
def pi_ic_from_qdot(
        qdot_vec: NDArray[Any, Float],
        q_vec: NDArray[Any, Float],
        tval: float,
        ddt: float,
        pi_guess_vec: NDArray[Any, Float],
        qi_sol_func: Callable[[NDArray[Any, Float], NDArray[Any, Float], float, float], NDArray[Any, Float]],
        qdot_n_func: Callable[
            [NDArray[Any, Float], NDArray[Any, Float], NDArray[Any, Float], float, float],
            NDArray[Any, Float]
        ]
) -> NDArray[Any, Float]:
    """
    This finds the initial condition for the pi vector for a given qdot_vec and q_vec initial condition, since pi
    depends on the choice of discretization.

    Args:
        qdot_vec: Initial qdot to be matched, shape (dof,).
        q_vec: Initial q, shape (dof,).
        tval: Initial time
        ddt: Time step size
        pi_guess_vec: Initial guess for pi, shape (dof,).
        qi_sol_func: 1st function returned by Gen_GGL_NC_VI_Map that generates an ndarray for the qi_sol
        qdot_n_func: 4th function returned by Gen_GGL_NC_VI_Map that calculates the value of qdot for a given pi_n.

    Returns:
        pi_init_sol, an ndarray for the solution of that matches the initial condition in terms of q and qdot given.
    """

    def fun(pi_vec: NDArray[Any, Float]):
        qi_sol = qi_sol_func(q_vec, pi_vec, tval, ddt)
        qdot_guess = qdot_n_func(qi_sol, q_vec, pi_vec, tval, ddt)
        return qdot_vec - qdot_guess

    return scipy.optimize.root(fun, pi_guess_vec)


# TODO: currently unused, what does this do?
def pi_ic_from_qnext(
        q_next_vec: NDArray[Any, Float],
        q_vec: NDArray[Any, Float],
        tval: float,
        ddt: float,
        pi_guess_vec: NDArray[Any, Float],
        qi_sol_func: Callable[[NDArray[Any, Float], NDArray[Any, Float], float, float], NDArray[Any, Float]],
        q_np1_func: Callable[
            [NDArray[Any, Float], NDArray[Any, Float], NDArray[Any, Float], float, float], NDArray[Any, Float]]
) -> NDArray[Any, Float]:
    """
    This finds the initial condition for the pi vector for a given qdot_vec and q_vec initial condition, since pi
    depends on the choice of discretization. Unless stated otherwise all vectors are assumed to be of shape (dof,).

    Inputs:
        qdot_vec: Initial qdot to be matched vector
        q_vec: Initial q vector
        tval: Initial time
        ddt: Time step size
        pi_guess_vec: Initial guess for pi vector
        qi_sol_func: 1st function returned by Gen_GGL_NC_VI_Map that generates an ndarray for the qi_sol
        qdot_n_func: 4th function returned by Gen_GGL_NC_VI_Map that calculates the value of qdot for a given pi_n

    Returns:
        pi_init_sol, ndarray for the solution of that matches the initial condition in terms of q and qdot given.
    """

    def fun(pi_vec: NDArray[Any, Float]):
        qi_sol = qi_sol_func(q_vec, pi_vec, tval, ddt)
        q_next_guess = q_np1_func(qi_sol, q_vec, pi_vec, tval, ddt)
        return q_next_vec - q_next_guess

    return scipy.optimize.root(fun, pi_guess_vec)


def Convert_EOM_Args(qi_vec, qn_vec, pi_nvec, tval, ddt, r):
    # Convert_EOM_Args returns an argument list for the lambdified EOM functions.
    # this should be [q_1^[n], q_1^(i), q_1^[n+1], pi_1^n,...]
    # these should all be numerical values
    EOM_arg_list = []
    dof_count = len(qn_vec)

    for dof in range(dof_count):
        # q^[n]
        EOM_arg_list.append(qn_vec[dof])

        # q^(i)
        for i in range(r + 1):
            EOM_arg_list.append(qi_vec[dof * (r + 1) + i])

        # pi^n
        EOM_arg_list.append(pi_nvec[dof])

    EOM_arg_list.append(tval)
    EOM_arg_list.append(ddt)

    return EOM_arg_list


def Gen_GGL_NC_VI_Map(
        t_symbol,
        q_list, q_p_list, q_m_list,
        v_list, v_p_list, v_m_list,
        Lexpr,
        Kexpr,
        r,
        sym_paramlist=[],
        sym_precision=20,
        eval_modules="numpy",
        method='implicit',
        verbose=True,
        verbose_rational=True
):
    """Gen_GGL_NC_VI_Map generates the mapping functions for the
    Galerkin-Gauss-Lobatto Nonconservative Variational Integrator
    described in Tsang, Galley, Stein & Turner (2015), for a generic
    System specified by the user using sympy symbols.

    Output:
    (qi_table_func, q_np1_func, pi_np1_func, v_n_func)
    A tuple of functions:
    *********************
     qi_sol_func(q_n_vec, pi_n_vec, tval, ddt)
       description: Main function of the mapping, iterates
                    to find the r intermediate values for each
                    degree of freedom q.
       outputs: qi_sol
                   - scipy.OptimizeResult containing values for
                     the intermediate and next values of q used
                     for this (r+2)th order method. This is used
                     by the other functions to calculate the
                     mappings and other values
       inputs: q_n_vec[dof]
                   - list of q_n degree of freedom values at the
                     current step
               pi_n_vec[dof]
                   - list of pi_n nonconservative discrete
                     momentum at the current step
               tval - current step initial time value
               ddt - step size in time.

     q_np1_func(qi_table, ddt)
       description: Computes the next position in the mapping
       outputs: q_np1_vec[dof]
                   - list of next value for each dof
       inputs: qi_table[dof][r+2]
                   - qi_table of iterated starting, intermediate
                     and final values for the dof, created by
                     qi_table_func
               ddt - step size in time

     pi_np1_func(qi_table, ddt)
       description: Computes the next n.c. momenta in the mapping
       outputs: pi_np1_vec[dof]
                   - list of next n.c. momenta value for each dof
       inputs: qi_table[dof][r+2]
                   - qi_table of iterated starting, intermediate
                     and final values for the dof, created by
                     qi_table_func
               ddt - step size in time

    qdot_n_func(qi_table, ddt)
       description: Computes the velocity at the current time
       outputs: v_n_vec[dof]
                   - list of velocity values for each dof
       inputs: qi_table[dof][r+2]
                   - qi_table of iterated starting, intermediate
                     and final values for the dof, created by
                     qi_table_func
               ddt - step size in time

    **********************


    Inputs:
    t_symbol - Symbol used for the time variable in expressions
    q_list[dof] - list of symbols representing the degrees
                  of freedom of the problem
    q_p_list[dof] - list of symbols representing the + doubled
                    degrees of freedom
    q_m_list[dof] - list of symbols representing the - doubled
                    degrees of freedom
    v_list[dof] - list of symbols representing the time
                  derivatives of the dof
    v_p_list[dof] - list of symbols representing the
                    time derivatives of the + doubled dof
    v_m_list[dof] - list of symbols representing the
                    time derivatives of the - doubled dof
    Lexpr - sympy expression for the Lagrangian, L, in terms of
            q_list and v_list variables.
    Kexpr - sympy expression for the nonconservative potential,
            K, in terms of q_p/m_list and v_p/m_list variables.
    r - number of intermediate points to be evaluated for (r+2)
        total collocation points. The order of the GGL NC VI
        method is given by (2r + 2).
    sym_paramlist - list of symbolic parameters to be substituted
                    for evaluations (ie physical constants). All
                    such non-dof symbols must be substituted for
                    numerical values here. If this is not the case
                    an error during the lambdification will occur.
    sym_precision - precision which the GGL constants are
                    evaluated to, prior to lambidification to
                    machine precision. This is to prevent
                    roundoff error from building up
    eval_modules - modules that need to be used to numerically
                   evaluate Lexpr and Kexpr
    verbose - boolean value that turns on and off verbose output
              defaults to True
    verbose_rational - boolean value that sets the verbose output
                       if True numerical values will be rationalized
                       if False values will be left as floats.
    """

    if not (method in ['implicit', 'explicit']):
        print(f"GGL_NC_VI ERROR: method = {method} unknown.")
        return

    # Define the symbol for h, that we will use in the algebraic
    # expressions
    ddt_symbol = Symbol('h_{GGL}')

    # Determine the Ld and Kd symbolic expressions for this system
    # As well as the q symbols for each dof and collocation point

    Ld, q_Table = GGL_Gen_Ld(
        t_symbol,
        q_list, v_list,
        Lexpr,
        ddt_symbol,
        r,
        paramlist=sym_paramlist,
        precision=sym_precision
    )

    Kd, q_p_Table, q_m_Table = GGL_Gen_Kd(
        t_symbol,
        q_p_list,
        q_m_list,
        v_p_list,
        v_m_list,
        Kexpr,
        ddt_symbol,
        r,
        paramlist=sym_paramlist,
        precision=sym_precision
    )

    # Generate momenta symbol lists
    pi_n_list, pi_np1_list = Gen_pi_list(q_list)

    # Generate the Equation of Motion Table for
    # q^[n] and q^(i)'s, but not q^[n+1]
    # (since pi_n+1 will be evaluated directly later)
    EOM_List = Gen_iter_EOM_List(
        q_Table, q_p_Table, q_m_Table,
        pi_n_list, pi_np1_list,
        Ld, Kd,
        ddt_symbol
    )

    # return EOM_List
    # qi symbol list: q^i and q^n+1 for each dof
    # the variables to be solved for by the implicit
    # method
    qi_symbol_list = []
    for dof in range(len(q_Table)):
        for i in range(1, r + 2):
            qi_symbol_list.append(q_Table[dof][i])

    # Generate flat list of the symbols for lambdification
    full_variable_list = []
    for i in range(len(q_list)):
        for j in range(r + 2):
            full_variable_list.append(q_Table[i][j])
        full_variable_list.append(pi_n_list[i])
    full_variable_list.append(t_symbol)
    full_variable_list.append(ddt_symbol)

    # print full_variable_list

    # Generate the list of functions for evaulating the EOM
    EOM_Func_List = [lambdify(tuple(full_variable_list),
                              EOM,
                              modules=eval_modules)
                     for EOM in EOM_List]

    # Generate the J_qi_vec sympy variables to take
    # the Jacobian with respect to
    J_qi_vec = []
    for i in range(len(q_Table)):
        for j in range(1, r + 2):
            J_qi_vec.append(q_Table[i][j])

    # Generate the Jacobian function table
    J_Expr_Table = compute_jacobian(EOM_List, J_qi_vec)
    J_Func_Table = [[lambdify(tuple(full_variable_list),
                              J_Expr, modules=eval_modules)
                     for J_Expr in J_Expr_Vec]
                    for J_Expr_Vec in J_Expr_Table]

    # EOM_Val_Vec is the function to be passed to
    # scipy.optimize.root() that returns the Equation
    # of Motion evaulations that should be zero for
    # the correct values extra arguments qn_vec, pi_nvec,
    # t, and ddt should be passed as well

    def EOM_Val_Vec(qi_vec, qn_vec, pi_nvec, tval, ddt):
        # First convert the argument list for
        # the lambdified functions
        EOM_arg_list = Convert_EOM_Args(qi_vec,
                                        qn_vec,
                                        pi_nvec,
                                        tval,
                                        ddt, r=r)
        # print EOM_arg_list
        # Next we evaulate the EOM functions
        # in EOM_List
        out = numpy.array([EOM_Func(*tuple(EOM_arg_list))
                           for EOM_Func in EOM_Func_List])
        # print out
        return out

    # EOM_J_Matrix is the function to be passed
    # to scipy.optimize.root() that returns the
    # Jacobian matrix for

    def EOM_J_Matrix(qi_vec, q_n_vec, pi_n_vec, tval, ddt):
        # First convert the argument list for
        # the lambdified functions
        EOM_arg_list = Convert_EOM_Args(qi_vec,
                                        q_n_vec,
                                        pi_n_vec,
                                        tval,
                                        ddt, r=r)
        # Next Evaluate the J_Matrix
        J_Matrix = [[J_Func(*tuple(EOM_arg_list))
                     for J_Func in J_Func_Vec]
                    for J_Func_Vec in J_Func_Table]
        # print "J_matrix:"
        # print J_Matrix

        return numpy.array(J_Matrix)

    if method == 'explicit':
        # print 'EXPLICIT METHOD'
        qi_func_args = []
        for dof in range(len(q_Table)):
            qi_func_args.append(q_Table[dof][0])
        for dof in range(len(pi_n_list)):
            qi_func_args.append(pi_n_list[dof])
        qi_func_args.append(t_symbol)
        qi_func_args.append(ddt_symbol)

        qi_sol_dict = solve(EOM_List, qi_symbol_list, dict=True)
        #        print qi_symbol_list[0]
        #        print qi_sol_dict
        #        print qi_sol_dict[0]
        if not qi_sol_dict:
            print("ERROR: explicit solve failed, try implicit solution")
            return
        qi_sol_list = [qi_sol_dict[0][qi_symbol]
                       for qi_symbol in qi_symbol_list]
        #        print qi_sol_list
        qi_func_list = [lambdify(qi_func_args,
                                 qi_sol,
                                 modules=eval_modules)
                        for qi_sol in qi_sol_list]

        # print qi_sol_list
        # print qi_sol_list

    # These are the output functions:
    #############################

    if method == 'explicit':
        def qi_sol_func_explicit(q_n_vec, pi_n_vec, tval, ddt, root_args={}):
            """This function evaluates the explicit equations
            for the intemediate points [{q_1^(i)_0}, q_1^[n+1]_0, ...]
            to generate the iterated intermediate results
            for the explicit GGL-NC-VI method.

            Output:
            qi_sol - the ndarray that contains the value of qi
            Input:
            q_n_vec[dof] - ndarray of current q_n values
            pi_n_vec[dof] - ndarray of current pi_n values
            tval - float for the current value of time
            ddt - float for the size of the time step
            """
            # Populate qi_0 array for nsolve()

            qi_arg_vals = []
            for dof in range(len(q_n_vec)):
                qi_arg_vals.append(q_n_vec[dof])
            for dof in range(len(pi_n_vec)):
                qi_arg_vals.append(pi_n_vec[dof])
            qi_arg_vals.append(tval)
            qi_arg_vals.append(ddt)

            qi_sol = [qi_func(*tuple(qi_arg_vals))
                      for qi_func in qi_func_list]
            # print qi_arg_vals
            # print qi_sol

            return numpy.array(qi_sol)

    def qi_sol_func_implicit(q_n_vec, pi_n_vec, tval, ddt,
                             root_args={'tol': 1e-10}):
        """This function uses q_n_vec as a guess for each of the
        intermediate points [{q_1^(i)_0}, q_1^[n+1]_0, ...]
        to generate the iterated intermediate results
        for the implicit GGL-NC-VI method.

        Output:
        qi_sol - nd.array qi_sol for the
                 results of that give the roots of the appropriate
                 discrete equations of motion.
        Input:
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step
        root_args - dictionary of arguments to be included in the
                    scipy.optimize.root() method e.g.
                    {'method':'hybr', 'tol': 1e-8}

        """
        # Populate qi_0 array for nsolve()
        qi_0 = []
        for i in range(len(q_n_vec)):
            for j in range(1, r + 2):
                qi_0.append(q_n_vec[i])
        qi_0 = numpy.array(qi_0)

        qi_sol = scipy.optimize.root(**dict(list({'fun': EOM_Val_Vec,
                                                  'x0': qi_0,
                                                  'args': (q_n_vec,
                                                           pi_n_vec,
                                                           tval,
                                                           ddt),
                                                  'jac': EOM_J_Matrix
                                                  }.items())
                                            + list(root_args.items())
                                            )
                                     )
        return qi_sol.x

    def q_np1_func(qi_sol, q_n_vec, pi_n_vec, tval, ddt):
        """This function uses the qi_sol from the first
        Gen_GGL_NC_VI_Map returned function to calculate
        the q's for the next step. In this case it will be just a
        simple lookup.

        Outputs:
        q_np1_vec[dof] - ndarray of next q values

        Inputs:
        qi_sol[dof*(r+1)] - ndarray of qi_values from
                            qi_sol_func
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step

        """
        q_np1_vec = [qi_sol[dof * (r + 1) + r] for dof in range(len(q_n_vec))]
        return numpy.array(q_np1_vec)

    # Here are some variables and functions to help
    # with creating the output functions

    q_longlist = [q for qvec in q_Table for q in qvec]
    q_p_longlist = [q for qvec in q_p_Table for q in qvec]
    q_m_longlist = [q for qvec in q_m_Table for q in qvec]
    pi_n_expr = [diff(Ld, q_Table[dof][-1])
                 + Physical_Limit(q_longlist,
                                  q_p_longlist,
                                  q_m_longlist,
                                  diff(Kd, q_m_Table[dof][-1]))
                 for dof in range(len(q_Table))]
    # return pi_n_expr

    pi_Func_Vec = [lambdify(full_variable_list,
                            expr, modules=eval_modules)
                   for expr in pi_n_expr]

    def pi_np1_func(qi_sol, q_n_vec, pi_n_vec, tval, ddt):
        """This function uses the qi_sol from the first
        Gen_GGL_NC_VI_Map returned function to calculate
        the pi's for the next step. This involves evaluating the last
        equation of motion dL_d/d(q^[n+1]) + ...

        Outputs:
        pi_np1_vec[dof] - ndarray of next pi values

        Inputs:
        qi_sol[dof*(r+1)] - ndarray of qi_values from
                            qi_sol_func
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step
        """
        EOM_Arg_list = Convert_EOM_Args(qi_sol,
                                        q_n_vec,
                                        pi_n_vec,
                                        tval,
                                        ddt, r=r)
        pi_np1_vec = [pi_func(*tuple(EOM_Arg_list))
                      for pi_func in pi_Func_Vec]

        # print qi_sol, q_n_vec, pi_n_vec, ddt
        # print pi_np1_vec

        return numpy.array(pi_np1_vec)

    # We need DM for the dotq function
    xs, ws, DM = generate_collocation_points(r)

    def qdot_n_func(qi_sol, q_n_vec, pi_n_vec, tval, ddt):
        """This function uses the qi_sol from the first
        Gen_GGL_NC_VI_Map returned function to calculate
        the qdot velocities for current step. This involves
        evaluating qdot using the derivative matrix defined by
        the GGL_defs function, this will be evaluated when this
        function is generated, rather than each time it is called.

        Outputs:
        pi_np1_vec[dof] - ndarray of next pi values

        Inputs:
        qi_sol[dof*(r+1)] - ndarray of qi_values from
                            qi_sol_func
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step
        """
        qi_table = []
        for dof in range(len(q_n_vec)):
            qi_vec = [q_n_vec[dof]]
            for i in range(r + 1):
                qi_vec.append(qi_sol[dof * (r + 1) + i])
            qi_table.append(qi_vec)

        qdot_vec = [numpy.dot(numpy.array(DM), qi_vec)[0] * 2 / ddt
                    for qi_vec in qi_table]
        return numpy.array(qdot_vec, dtype=float)

    # Verbose output:

    if verbose:
        print('===================================')
        print('For Lagrangian:')
        print('\t L = ' + latex(Lexpr))
        print('and K-potential:')
        print('\t K = ' + latex(Kexpr))
        print('********************')
        print('The Order ' + repr(2 * r + 2) + ' discretized Lagrangian is:')

        if verbose_rational:
            print('\t L_d^n = '
                  + latex(nsimplify(Ld,
                                    tolerance=1e-15,
                                    rational=True)))
        else:
            print('\t L_d^n = ' + latex(simplify(Ld)))

        print('The Order ' + repr(2 * r + 2) + ' discretized K-potential is:')
        if verbose_rational:
            print('\t K_d^n = '
                  + latex(nsimplify(Kd,
                                    tolerance=1e-12,
                                    rational=verbose_rational)))
        else:
            print('\t K_d^n = ' + latex(simplify(Kd)))

        print('********************')
        print('The Order ' + repr(2 * r + 2) + ' Discretized Equations of motion:')

        if verbose_rational:
            for dof in range(len(q_Table)):
                for i in range(r + 1):
                    print('\t0 = ' + latex(nsimplify(expand(EOM_List[dof * (r + 1) + i]),
                                                     tolerance=1e-15,
                                                     rational=verbose_rational)))
                print('\t0 = ' + latex(nsimplify(-pi_np1_list[dof] + expand(pi_n_expr[dof]),
                                                 tolerance=1e-15,
                                                 rational=verbose_rational)))

        else:
            for dof in range(len(q_Table)):
                for i in range(r + 1):
                    print('\t0 = ' + latex(simplify(expand(EOM_List[dof * (r + 1) + i]))))
                print('\t0 = ' + latex(-pi_np1_list[dof] + simplify(expand(pi_n_expr[dof]))))
        print('===================================')

    #########################
    if method == 'implicit':
        return qi_sol_func_implicit, q_np1_func, pi_np1_func, qdot_n_func
    else:
        return qi_sol_func_explicit, q_np1_func, pi_np1_func, qdot_n_func

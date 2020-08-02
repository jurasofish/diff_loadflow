import pandapower as pp
import pandapower.networks as ppnw
import autograd.numpy as np
from autograd import grad
import time
from copy import deepcopy

np.set_printoptions(formatter={'complexfloat': lambda x: "{0:.3f}".format(x),
                               'float_kind': lambda x: "{0:.3f}".format(x)})


def fin_diff(f, x, eps=1e-6):
    """ Finite difference approximation of grad of function f. From JAX docs. """
    return np.array([(f(x + eps*v) - f(x - eps*v)) / (2*eps) for v in np.eye(len(x))])


def init_v(net, n, pd2ppc):
    """ Initial bus voltage vector using generator voltage setpoints or 1j+0pu. """
    v = [0j + 1 for _ in range(n)]
    for r in net.gen.itertuples():
        v[pd2ppc[r.bus]] = r.vm_pu
    for r in net.ext_grid.itertuples():
        v[pd2ppc[r.bus]] = r.vm_pu * np.exp(1j * r.va_degree * np.pi / 180)
    return np.array(v, dtype=np.complex64)


def scheduled_p_q(net, n, pd2ppc):
    """ Return known per unit real and reactive power injected at each bus.
    Does not include slack real/reactive powers nor PV gen reactive power.
    """
    psch, qsch = {b: 0 for b in range(n)}, {b: 0 for b in range(n)}
    for r in net.gen.itertuples():
        psch[pd2ppc[r.bus]] += r.p_mw / net.sn_mva
    for r in net.sgen.itertuples():
        psch[pd2ppc[r.bus]] += r.p_mw / net.sn_mva
        qsch[pd2ppc[r.bus]] += r.q_mvar / net.sn_mva
    for r in net.load.itertuples():
        psch[pd2ppc[r.bus]] -= r.p_mw / net.sn_mva
        qsch[pd2ppc[r.bus]] -= r.q_mvar / net.sn_mva
    return psch, qsch


def run_lf(load_p, net, tol=1e-9, comp_tol=1e-3, max_iter=10000):
    """ Perform Gauss-Seidel power flow on the given pandapower network.

    The ``load_p`` array is an iterable of additional real power load to add
    to each bus. By providing this as an input, this python function becomes
    the function ``slack_power = f(load_power)`` and thus the derivative of
    slack power with respect to load power can be calculated.

    By restricting the values of `load_p` to be very small we can ensure that
    this function is solving the load flow correctly by comparing it to the
    results in the pandapower network. Restricting to very small does not interfere
    with calculation of the derivative - that is, the derivative of
    x (the small value given as input) plus a constant (the load power specified
    in the pandapower network object) is equal to the derivative of x alone.

    Args:
        load_p (iterable): Iterable of additional loads to add to network.
            Index i will add load to bus i.
            These should be very small values so that the consistency
            assertion with pandapower succeeeds.
        net (pp.Network): Solved Pandapower network object that defines the
            elements of the network and contains the ybus matrix.
        tol (float): Convergence tolerance (voltage).
        comp_tol (float): Tolerance for comparison check against pandapower.
        max_iter(int): Max iterations to solve load flow.

    Returns:
        float: Sum of real power injected by slack buses in the network.
    """
    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    pd2ppc = net._pd2ppc_lookups["bus"]  # Pandas bus num --> internal bus num.
    n = ybus.shape[0]  # Number of buses.
    slack_buses = set(pd2ppc[net.ext_grid['bus']])
    gen_buses = set([pd2ppc[b] for b in net.gen['bus']])
    ybus_hollow = ybus * (1 - np.eye(n))  # ybus with diagonal elements zeroed.
    v = init_v(net, n, pd2ppc)
    psch, qsch = scheduled_p_q(net, n, pd2ppc)
    # Incorporate the variables we are differentiating with respect to:
    psch = {b: p - load_p[b] for b, p in psch.items()}

    it = 0
    while it < max_iter:
        old_v, v = v, [x for x in v]
        for b in [b for b in range(n) if b not in slack_buses]:
            qsch_b = (-1*np.imag(np.conj(old_v[b]) * np.sum(ybus[b, :] * old_v))
                      if b in gen_buses else qsch[b])
            v[b] = (1/ybus[b, b]) * ((psch[b]-1j*qsch_b)/np.conj(old_v[b])
                                     - np.sum(ybus_hollow[b, :] * old_v))
            if b in gen_buses:
                v[b] = np.abs(old_v[b]) * v[b] / np.abs(v[b])  # Only use angle.
        it += 1
        v = np.array(v)
        if np.allclose(v, old_v, rtol=tol, atol=0):
            break
    p_slack = sum((np.real(np.conj(v[b]) * np.sum(ybus[b, :] * v)) - psch[b])
                  for b in slack_buses)
    # Assert convergence and consistency with pandapower.
    assert it < max_iter, f'Load flow not converged in {it} iterations.'
    assert np.allclose(v, net._ppc["internal"]["V"], atol=comp_tol, rtol=0),\
           f'Voltage\npp:\t\t{net._ppc["internal"]["V"]}\nsolved:\t{v}'
    assert np.allclose(p_slack, net.res_ext_grid['p_mw'].sum(), atol=comp_tol, rtol=0),\
           f'Slack Power\npp:\t\t{net.res_ext_grid["p_mw"].sum()}\nsolved:\t{p_slack}'
    return p_slack


def run_lf_pp(load_p, net, algorithm='nr'):
    """ Calculate total slack generation with given load power changes.

    Args:
        load_p (iterable): Iterable of additional loads to add to network.
            Index i will add load to bus i.
        net (pp.Network):
        algorithm (str): Algorithm to pass to pandapower solver.

    Returns:
        float: Sum of real power injected by slack buses in the network.
    """
    net = deepcopy(net)
    pd2ppc = net._pd2ppc_lookups["bus"]  # Pandas bus num --> internal bus num.
    for b, extra_p in enumerate(load_p):
        pp.create_load(net, np.where(pd2ppc == b)[0][0], extra_p)
    pp.runpp(net, algorithm=algorithm)
    return net.res_ext_grid['p_mw'].sum()


def main():
    net = ppnw.case9()
    pp.runpp(net)  # Make sure network contains ybus and the solution values.
    load_p = np.zeros((net.bus.shape[0], ), np.float32)

    print(net)
    p_slack = run_lf(load_p, net)
    print(f'Slack generator power output: {p_slack}')

    print(f'\nGradient of slack power with respect to load at each bus:')

    t1 = time.perf_counter()
    f_grad_p_slack = grad(lambda x: run_lf(x, net))
    grad_p_slack = f_grad_p_slack(load_p)
    print(f'\tUsing this program with autograd:\n{grad_p_slack}')
    print(f'Took {time.perf_counter() - t1:.2f} Seconds')

    t1 = time.perf_counter()
    grad_p_slack_fin_diff = fin_diff(lambda x: run_lf(x, net), load_p)
    print(f'\tUsing this program finite differences:\n{grad_p_slack_fin_diff}')
    print(f'Took {time.perf_counter() - t1:.2f} Seconds')

    t1 = time.perf_counter()
    grad_p_slack_fin_diff_pp = fin_diff(lambda x: run_lf_pp(x, net, 'nr'), load_p)
    print(f'\nCalculated using finite differences and pandapower newton raphson'
          f':\n{grad_p_slack_fin_diff_pp}')
    print(f'Took {time.perf_counter() - t1:.2f} Seconds')

    t1 = time.perf_counter()
    grad_p_slack_fin_diff_pp = fin_diff(lambda x: run_lf_pp(x, net, 'gs'), load_p)
    print(f'\nCalculated using finite differences and pandapower gauss siedel'
          f':\n{grad_p_slack_fin_diff_pp}')
    print(f'Took {time.perf_counter() - t1:.2f} Seconds')


if __name__ == '__main__':
    main()

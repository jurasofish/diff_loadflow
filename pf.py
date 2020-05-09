import pandapower as pp
import pandapower.networks as ppnw
import autograd.numpy as np
from autograd import jacobian
import pandas as pd


pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100
np.set_printoptions(formatter={'complexfloat': lambda x: "{0:.3f}".format(x)})


class DiffLFException(Exception):
    pass


class ConvergenceError(DiffLFException):
    pass


class ConsistencyError(DiffLFException):
    pass


def init_v(net, n, pd2ppc):
    """ Initial voltage vector using generator voltage setpoints or 1j+0pu. """
    v = np.ones((n, ), dtype=np.complex64)
    for r in net.gen.itertuples():
        v[pd2ppc[r.bus]] = r.vm_pu
    for r in net.ext_grid.itertuples():
        v[pd2ppc[r.bus]] = r.vm_pu * np.exp(1j * r.va_degree * 180 / np.pi)
    return v


def scheduled_p_q(net, load_p, n, pd2ppc):
    """ Return known per unit absolute real and reactive power injected at each bus.
    That is, power injected from generators minus power absorbed by loads.
    Neither real nor reactive power injected by the slack gen is not included.
    Reactive power injected by PV gens is not included.
    """
    sb = net.sn_mva
    psch = {b: -1 * load_p[b]/sb for b in range(n)}
    qsch = {b: 0 for b in range(n)}
    for r in net.gen.itertuples():
        psch[pd2ppc[r.bus]] += r.p_mw / sb
    for r in net.sgen.itertuples():
        psch[pd2ppc[r.bus]] += r.p_mw / sb
        qsch[pd2ppc[r.bus]] += r.q_mvar / sb
    for r in net.load.itertuples():
        psch[pd2ppc[r.bus]] -= r.p_mw / sb
        qsch[pd2ppc[r.bus]] -= r.q_mvar / sb

    return psch, qsch


def run_lf(load_p, net, tol=1e-9, comparison_tol=1e-3, max_iter=10000):

    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    pd2ppc = net._pd2ppc_lookups["bus"]  # Pandas bus num --> internal bus num.
    n = ybus.shape[0]  # Number of buses.
    slack_bus = pd2ppc[net.ext_grid.iloc[0]['bus']]
    gen_buses = set([pd2ppc[b] for b in net.gen['bus']])
    ybus_hollow = ybus * (1 - np.eye(n))  # ybus with diagonal elements zeroed.

    v = init_v(net, n, pd2ppc)
    psch, qsch = scheduled_p_q(net, load_p, n, pd2ppc)

    it = 0
    while it < max_iter:
        old_v, v = v, [x for x in v]
        for b in [b for b in range(n) if b != slack_bus]:
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

    v = np.array(v)
    p_slack = (np.real(np.conj(v[slack_bus]) * np.sum(ybus[slack_bus, :] * v))
               - psch[slack_bus])

    if it >= max_iter:
        raise ConvergenceError(f'Load flow not converged in {it} iterations.')
    if not np.allclose(v, net._ppc["internal"]["V"], atol=comparison_tol, rtol=0):
        raise ConsistencyError(f'Voltages not consistent with pandapower\n'
                               f'pandapower\t\t{net._ppc["internal"]["V"]}'
                               f'\nthis program\t{v}')
    if not np.allclose(p_slack, net.res_ext_grid.iloc[0]['p_mw'],
                       atol=comparison_tol, rtol=0):
        raise ConsistencyError(f'Slack bus powers inconsistent\n'
                               f'pandapower\t\t{net.res_ext_grid.iloc[0]["p_mw"]}'
                               f'\nthis program\t{p_slack}')

    return p_slack


def main():
    net = ppnw.case9()
    net.ext_grid.at[0, 'vm_pu'] = 1.05
    net.gen.at[0, 'vm_pu'] = 0.99
    pp.create_bus(net, 345, index=1000)
    pp.create_line_from_parameters(net, 1000, 0, 1, 10, 100, 0, 1e10)
    pp.create_load(net, 1000, 100)
    pp.runpp(net)
    print(net)

    def run_lf_wrapper(__load_p):
        return run_lf(__load_p, net)

    load_p = np.zeros((net._ppc["internal"]["Ybus"].shape[0], ), np.float64)
    p_slack = run_lf_wrapper(load_p)
    print(f'Slack generator power output: {p_slack}')
    f_grad_p_slack = jacobian(run_lf_wrapper)
    grad_p_slack = f_grad_p_slack(load_p)
    print(f'Gradient of slack power with respect to load at each bus: {grad_p_slack}')


if __name__ == '__main__':
    main()

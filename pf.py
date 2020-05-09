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


def init_v(net, n):
    """ Initial voltage vector using generator voltage setpoints or 1j+0pu. """
    v = np.ones((n, ), dtype=np.complex64)
    for r in net.gen.itertuples():
        v[r.bus] = r.vm_pu
    for r in net.ext_grid.itertuples():
        v[r.bus] = r.vm_pu * np.exp(1j * r.va_degree * 180 / np.pi)
    return v


def scheduled_p_q(net, load_p, n):
    """ Return known per unit absolute real and reactive power injected at each bus.
    That is, power injected from generators minus power absorbed by loads.
    Neither real nor reactive power injected by the slack gen is not included.
    Reactive power injected by PV gens is not included.
    """
    sb = net.sn_mva
    psch, qsch = {b: 0 for b in range(n)}, {b: 0 for b in range(n)}
    for b in range(n):

        psch[b] -= load_p[b] / sb  # The magic.
        if b in net.gen['bus'].tolist():
            psch[b] += net.gen.loc[net.gen.bus == b, 'p_mw'].values.sum() / sb
        if b in net.sgen['bus'].tolist():
            psch[b] += net.sgen.loc[net.gen.bus == b, 'p_mw'].values.sum() / sb
            qsch[b] += net.sgen.loc[net.gen.bus == b, 'q_mvar'].values.sum() / sb
        if b in net.load['bus'].tolist():
            psch[b] -= net.load.loc[net.load['bus'] == b, 'p_mw'].values.sum() / sb
            qsch[b] -= net.load.loc[net.load['bus'] == b, 'q_mvar'].values.sum() / sb
    return psch, qsch


def run_lf(load_p, net, tol=1e-6, max_iter=100):

    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    n = ybus.shape[0]  # Number of buses.
    slack_bus = net.ext_grid.iloc[0]['bus']
    gen_buses = set(net.gen['bus'])
    ybus_hollow = ybus * (1 - np.eye(n))  # ybus with diagonal elements zeroed.

    v = init_v(net, n)
    psch, qsch = scheduled_p_q(net, load_p, n)

    it = 0
    while it < max_iter:
        old_v, v = v, [x for x in v]
        for b in [b for b in range(n) if b != slack_bus]:
            qsch_b = (-1*np.imag(np.conj(v[b]) * np.sum(ybus[b, :] * v))
                      if b in gen_buses else qsch[b])
            v[b] = (1/ybus[b, b]) * ((psch[b]-1j*qsch_b)/np.conj(old_v[b])
                                     - np.sum(ybus_hollow[b, :] * old_v))
            if b in gen_buses:
                v[b] = np.abs(old_v[b]) * v[b] / np.abs(v[b])  # Only use angle.
        it += 1

        errs = [np.abs(v[i] - old_v[i]) for i in range(n)]
        if all(np.real(x) < tol and np.imag(x) < tol for x in errs):
            break

    if it >= max_iter:
        raise ConvergenceError(f'Load flow not converged in {it} iterations.')

    v = np.array(v)
    if not np.allclose(v, net._ppc["internal"]["V"], atol=0, rtol=tol):
        raise ConsistencyError(f'Voltages not consistent with pandapower\n'
                               f'pandapower\t\t{net._ppc["internal"]["V"]}'
                               f'\nthis program\t{v}')

    p_slack = (np.real(np.conj(v[slack_bus]) * np.sum(ybus[slack_bus, :] * v))
               - psch[slack_bus])
    return p_slack


def main():
    net = ppnw.case4gs()
    net.ext_grid.at[0, 'vm_pu'] = 1.05
    net.gen.at[0, 'vm_pu'] = 0.99
    pp.runpp(net)
    print(net)

    def run_lf_wrapper(__load_p):
        return run_lf(__load_p, net)

    load_p = np.zeros_like(net.load['p_mw'].values)
    p_slack = run_lf_wrapper(load_p)
    f_grad_p_slack = jacobian(run_lf_wrapper)
    grad_p_slack = f_grad_p_slack(load_p)
    print(f'Gradient of slack power with respect to load at each bus: {grad_p_slack}')


if __name__ == '__main__':
    main()

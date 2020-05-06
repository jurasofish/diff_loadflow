import pandapower as pp
import pandapower.networks as ppnw
import autograd.numpy as np
from autograd import jacobian
import pandas as pd


pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100
np.set_printoptions(formatter={'complexfloat': lambda x: "{0:.3f}".format(x)})


def init_v(net, n):
    """ Return initial voltage vector based on generator voltage setpoints.
    Buses without a generator are 0j+1
    """
    v = np.ones((n, ), dtype=np.complex64)
    for r in net.gen.itertuples():
        v[r.bus] = r.vm_pu
    for r in net.ext_grid.itertuples():
        v[r.bus] = r.vm_pu * np.exp(1j * r.va_degree * 180 / np.pi)
    return v


def run_lf(load_p, net, tol=1e-6, max_iter=500):

    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    n = ybus.shape[0]  # Number of buses.
    slack_bus = net.ext_grid.iloc[0]['bus']
    gen_buses = set(net.gen['bus'])
    sb = net.sn_mva
    ybus_hollow = ybus - np.eye(n)*ybus  # ybus with diagonal elements zeroed.
    v = init_v(net, n)

    # Per unit absolute real and reactive power injected at each bus.
    psch, qsch = {i: 0 for i in range(n)}, {i: 0 for i in range(n)}
    for b in range(n):
        if b in net.gen['bus'].tolist():
            psch[b] += net.gen.loc[net.gen.bus == b, 'p_mw'].values[0] / sb
        if b in net.load['bus'].tolist():
            psch[b] -= load_p[b] / sb  # The magic.
            qsch[b] -= net.load.loc[net.load['bus'] == b, 'q_mvar'].values[0] / sb

    iter = 0
    while iter < max_iter:
        old_v, new_v = v, [None for _ in range(n)]
        for b in range(n):
            if b == slack_bus:
                bv = v[b]
            else:
                bv = ((psch[b] - 1j*qsch[b])/np.conj(v[b])
                      - np.sum(ybus_hollow[b, :] * v)) / ybus[b, b]
                if b in gen_buses:
                    bv = np.abs(v[b]) * bv / np.abs(bv)  # Correct magnitude.
            new_v[b] = bv
        v = np.array(new_v)
        iter += 1
        if np.sum(np.abs(v - old_v)) < n * tol:  # TODO: correct this.
            break
    print(f'{iter} iterations')

    print(v)
    print(net.res_bus.vm_pu * np.exp(1j * net.res_bus.va_degree * np.pi/180))

    '''
    loss = np.array(range(1, 1+len(load_p))) * load_p
    loss = loss + np.array(range(10, 10+len(load_p)))  # Should be ignored
    loss = loss * v
    loss = np.abs(np.sum(loss))
    '''
    loss = np.sum(np.abs(v)) / v.shape[0]
    return loss


def main():
    net = ppnw.case4gs()
    net.ext_grid.at[0, 'vm_pu'] = 1.05
    net.gen.at[0, 'vm_pu'] = 0.99
    net.load.at[2, 'p_mw'] = 1000
    pp.runpp(net)
    print(net)

    def run_lf_wrapper(__load_p):
        return run_lf(__load_p, net, tol=1e-4, max_iter=50)

    load_p = net.load['p_mw'].values
    loss = run_lf_wrapper(load_p)
    print(f'loss is {loss:.2f} MW')
    f_grad_loss = jacobian(run_lf_wrapper)
    grad_loss = f_grad_loss(load_p)
    print(f'Gradient of loss with respect to inputs {grad_loss}')


if __name__ == '__main__':
    main()

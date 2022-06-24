import numpy as np

def maxminnorm(d,attr=None):
    if attr is None:
        attr = {'max':np.max(d),'min':np.min(d)}
    else:
        attr['max'],attr['min'] = np.max(d), np.min(d)
    return (d-np.min(d))/(attr['max']-attr['min']), attr
def lognorm(d,attr=None,offset = 0):
    return np.log(d+offset),attr
def log1norm(d,attr=None):
    return np.log(1+d),attr
def sqrt1norm(d,attr=None):
    return np.sqrt(1+d),attr
def cubrt1norm(d,attr=None):
    return np.power(1+d,1/3),attr
def cubrtnorm(d,attr=None):
    return np.power(d,1/3),attr
def znorm(d,attr=None):
    m,s = np.mean(d),np.std(d)
    if attr is None:
        attr = {'mean':m,'std' :s}
    else:
        attr['mean'],attr['std'] = m,s
    return (d-m)/s,attr

transforms = {
    'ta'  :[
        lambda d: znorm(*log1norm(*maxminnorm(-d))),
        lambda o,a: -((a['max']-a['min'])*(np.exp((a['std']*o+a['mean']))-1)+a['min'])
    ],
    'ua'  :[
        lambda d: znorm(*cubrtnorm(*maxminnorm(d))),
        lambda o,a: (a['max']-a['min'])*(np.power(a['std']*o+a['mean'],3))+a['min']
    ],
    'va'  :[
        lambda d: znorm(d),
        lambda o,a: a['std']*o+a['mean']
    ],
    'wap' :[
        lambda d: znorm(d),
        lambda o,a: a['std']*o+a['mean']
    ],
    'hus' :[
        lambda d: znorm(*cubrtnorm(*maxminnorm(d))),
        lambda o,a: (a['max']-a['min'])*(np.power(a['std']*o+a['mean'],3))+a['min']
    ],
    'zeta':[
        lambda d: znorm(d),
        lambda o,a: a['std']*o+a['mean']
    ],
    'zg'  :[
        lambda d: znorm(*lognorm(*maxminnorm(-d),offset = 0.02)),
        lambda o,a:  -((a['max']-a['min'])*(np.exp((a['std']*o+a['mean']))-0.02)+a['min'])
    ],
    'd'   :[
        lambda d: znorm(d),
        lambda o,a: a['std']*o+a['mean']
    ],
    'ps'  :[
        lambda d: znorm(d), #znorm(*lognorm(*maxminnorm(-d),offset = 1)),
        lambda o,a: a['std']*o+a['mean'],#-((a['max']-a['min'])*(np.exp((a['std']*o+a['mean']))-1)+a['min'])
    ],
}

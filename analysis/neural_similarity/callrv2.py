"""Get matrix correlation."""
import datetime
import hoggorm as ho

def rv2(x, y, layer2):
    '''Calculate the modified RV coefficient between two matrices.'''
    print('-start {} at {}'.format(layer2, datetime.datetime.now()))
    res = ho.RV2coeff([x, y])[0, 1]
    # res = ho.RV2coeff([x[:1000,:64], y[:1000,:64]])[0, 1]
    print('-done {} at {}'.format(layer2, datetime.datetime.now()))
    return res

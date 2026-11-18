import numpy as np
import pacrunner

out = pacrunner.Output("processes/zgg")
a = out.subprocess("A1_uux_zgg")

momenta = np.array([
    [500., 0., 0., 500.],
    [ 500., 0., 0., -500.],
    [142.22002372, -53.10090083, 59.0409804, -74.87149502],
    [380.350097, -268.20692462, 256.15852034, -84.34485361],
    [477.42987928,  321.30782545, -315.19950074, 159.21634863]])

batch = momenta[None, :, :]
provider_cpu = a.compile(device="cpu") 
rt_cpu = provider_cpu.runtime() # the stock Madspace runtime
print(rt_cpu(batch, 0).matrix_element)    # should print [2.43239313e-05]

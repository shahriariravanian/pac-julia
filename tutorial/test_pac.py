import os
import numpy as np
import madspace as ms

artifact = os.path.join(os.path.dirname(__file__), "artifacts/zgg")
umami = os.path.join(os.path.dirname(__file__), "artifacts/.pyamplicol-api-build/zgg/umami/libumami.so")

momenta = np.array([
    [500., 0., 0., 500.],
    [ 500., 0., 0., -500.],
    [142.22002372, -53.10090083, 59.0409804, -74.87149502],
    [380.350097, -268.20692462, 256.15852034, -84.34485361],
    [477.42987928,  321.30782545, -315.19950074, 159.21634863]])

ctx = ms.Context(ms.cpu_device(), 1)
api = ctx.load_matrix_element(umami, artifact)
ME = ms.MatrixElement
me = ME(api.index(), len(momenta),
    [ME.momenta_in, ME.flavor_in, ME.channel_in],
    [ME.matrix_element_out],
    diagram_count=1, sample_random_inputs=False, me_frame=[], incoming_count=2, channel=0)
runtime = ms.FunctionRuntime(me.function(), ctx)
flavours = np.full(1, 0, np.int32)
(res,) = runtime.call([momenta[None, :, :], flavours])
print(res.numpy()[0])

import pacrunner

a = pacrunner.Artifact("artifacts/zgg/")    # a pyAmpliCol artifact
process, momenta = a.validation_points()[0]
batch = momenta[None, :, :]                 # a batch of one event, in host memory
provider_cuda = a.compile(device="cuda")    # PacRunner's provider on the GPU, called from the CPU
rt_cuda = provider_cuda.runtime()           # a pacrunner.Runtime of its channel 0
print(rt_cuda(batch, 0).matrix_element)     # should print [2.43239313e-05]

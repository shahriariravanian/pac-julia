import pacrunner

a = pacrunner.Artifact("artifacts/zgg/")   # a is a pyAmpliCol Artifact
process, momenta = a.validation_points()[0] 
batch = momenta[None, :, :]
provider_cuda = a.compile(device="cuda") 
rt_cuda = provider_cuda.runtime() # the stock Madspace runtime
print(rt_cuda(batch, 0).matrix_element)    # should print [2.43239313e-05]

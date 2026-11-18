import pacrunner
import cupy as cp

a = pacrunner.Artifact("artifacts/zgg/")   # a is a pyAmpliCol Artifact
process, momenta = a.validation_points()[0] 
batch = cp.array(momenta[None, :, :])
provider_cuda_provider = a.compile(device="cuda_provider") 
rt_cuda_provider = provider_cuda_provider.runtime() # the stock Madspace runtime
print(rt_cuda_provider(batch, 0).matrix_element)    # should print [2.43239313e-05]

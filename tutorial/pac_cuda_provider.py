import pacrunner
import cupy as cp

a = pacrunner.Artifact("artifacts/zgg/")    # a pyAmpliCol artifact
process, momenta = a.validation_points()[0]
batch = cp.array(momenta[None, :, :])       # a batch of one event, in GPU memory
provider_gpu = a.compile(device="cuda_full")   # PacRunner's CUDA provider (alias: "cuda_provider")
rt_gpu = provider_gpu.runtime()             # a pacrunner.Runtime of its channel 0, on a CUDA context
print(rt_gpu(batch, 0).matrix_element)      # should print [2.43239313e-05]

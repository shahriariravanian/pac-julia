import pacrunner

a = pacrunner.Artifact("artifacts/zgg/")   # a is a pyAmpliCol Artifact
process, momenta = a.validation_points()[0] 
batch = momenta[None, :, :]
provider_cpu = a.compile(device="cpu") 
rt_cpu = provider_cpu.runtime() # the stock Madspace runtime
print(rt_cpu(batch, 0).matrix_element)    # should print [2.43239313e-05]

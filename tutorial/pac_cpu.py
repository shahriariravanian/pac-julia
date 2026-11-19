import pacrunner

a = pacrunner.Artifact("artifacts/zgg/")    # a pyAmpliCol artifact
process, momenta = a.validation_points()[0]
batch = momenta[None, :, :]                 # a batch of one event
provider_cpu = a.compile(device="cpu")      # PacRunner's provider on the CPU
rt_cpu = provider_cpu.runtime()             # a pacrunner.Runtime of its channel 0
print(rt_cpu(batch, 0).matrix_element)      # should print [2.43239313e-05]

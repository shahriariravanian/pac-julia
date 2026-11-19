import pacrunner

out = pacrunner.Output("processes/zgg")
a = out.subprocess("A1_uux_zgg")

# the matrix elements on the GPU, MadSpace on the CPU: the output's default run card works
provider_cuda = a.compile(device="cuda")
print(provider_cuda.run().cross_section)    # should print about 75.3 (pb, ± 0.2: the default card draws a new seed each run)

import pacrunner

out = pacrunner.Output("processes/zgg")
a = out.subprocess("A1_uux_zgg")

provider_cuda_provider = a.compile(device="cuda_provider") 
print(provider_cuda_provider.run().cross_section)    # should print 75.4519586763261


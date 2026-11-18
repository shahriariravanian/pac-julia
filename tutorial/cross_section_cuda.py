import pacrunner

out = pacrunner.Output("processes/zgg")
a = out.subprocess("A1_uux_zgg")

provider_cuda = a.compile(device="cuda") 
print(provider_cuda.run().cross_section)    # should print 75.4519586763261


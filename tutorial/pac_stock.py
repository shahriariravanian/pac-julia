import pacrunner

a = pacrunner.Artifact("artifacts/zgg/")   # a is a pyAmpliCol Artifact
process, momenta = a.validation_points()[0] 
batch = momenta[None, :, :]
provider_stock = a.stock()  # the stock provider, i.e., `libumami.so` 
rt_stock = provider_stock.runtime() # the stock Madspace runtime
print(rt_stock(batch, 0).matrix_element)    # should print [2.43239313e-05]


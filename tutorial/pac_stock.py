import pacrunner

a = pacrunner.Artifact("artifacts/zgg/")    # a pyAmpliCol artifact
process, momenta = a.validation_points()[0]
batch = momenta[None, :, :]                 # a batch of one event
provider_stock = a.stock()                  # pyAmpliCol's own provider (libumami.so beside the artifact)
rt_stock = provider_stock.runtime()         # a pacrunner.Runtime of its channel 0
print(rt_stock(batch, 0).matrix_element)    # should print [2.43239313e-05]

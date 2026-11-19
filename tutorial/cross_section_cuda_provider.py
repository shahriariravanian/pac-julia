import re
from pathlib import Path
import pacrunner

# everything on the GPU (MadSpace's CUDA build): the run card needs run.device = ["cuda"] and leading
# colour (the full-colour correction is a CPU library), so a second output, processes/zgg_gpu
card = Path("processes/zgg_gpu/Cards/run_card.toml")
text = re.sub(r'(?m)^device = .*$', 'device = ["cuda"]', card.read_text())
text = re.sub(r'(?m)^color_accuracy = .*$', 'color_accuracy = "lc"', text)
card.write_text(text)

out = pacrunner.Output("processes/zgg_gpu")
a = out.subprocess("A1_uux_zgg")
provider_gpu = a.compile(device="cuda_full")    # alias of "cuda_provider"
print(provider_gpu.run().cross_section)    # should print about 94.4 (pb, ± 0.2: leading colour, so above the full-colour 75.3)

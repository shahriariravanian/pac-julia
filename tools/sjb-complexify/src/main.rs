// sjb-complexify IN OUT [IN OUT ...]: each saved symjit application (.sjb, .plane.symjit) loaded,
// its MIR complexified by symjit (Application::complexify: a complex model's complex MIR becomes
// the real MIR its kernels run), and saved as OUT. A real application is saved as it was loaded.
// symjit refuses to load a complexified file again; readers of the MIR (PacRunner's read_sjb) read it.
use std::fs;

use anyhow::{ensure, Context, Result};
use symjit::{Application, Config, Defuns, Storage};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!args.is_empty() && args.len() % 2 == 0, "usage: sjb-complexify IN OUT [IN OUT ...]");
    for pair in args.chunks(2) {
        let (input, output) = (&pair[0], &pair[1]);
        let bytes = fs::read(input).with_context(|| input.clone())?;
        let mut config = Config::default();
        config.set_defuns(Defuns::new());
        let mut app = Application::load(&mut &bytes[..], &config).with_context(|| input.clone())?;
        app.complexify()?;
        let mut out = Vec::new();
        app.save(&mut out)?;
        fs::write(output, &out).with_context(|| output.clone())?;
    }
    Ok(())
}

// The header and the MIR of a .sjb as symjit reads it, in the text of sjb.jl's test.
use std::fs::File;
use symjit::backend::{Loc, MirInstruction};
use symjit::{Application, CompilerType, Config, MirWriter};

fn code(l: &Loc) -> u64 {
    match l {
        Loc::Mem(i) => 0x100000000 | *i as u64,
        Loc::Stack(i) => 0x200000000 | *i as u64,
        Loc::Param(i) => 0x300000000 | *i as u64,
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut path = args.next().unwrap();
    if path == "--code" {
        // raw MIR code bytes: the instructions as symjit decodes them
        path = args.next().unwrap();
        let w = MirWriter { buf: std::fs::read(&path)?, ip: 0 };
        for (i, ins) in w.iter().enumerate() {
            match &ins {
                MirInstruction::Call { label, num_args } => println!("{:05}\tcall {} / {}", i, label, num_args),
                _ => println!("{:05}\t{:?}", i, ins),
            }
        }
        return Ok(());
    }
    let config = Config::new(CompilerType::ByteCode, 0)?;
    let (prog, reals, mir) = Application::read_sjb(&mut File::open(&path)?, &config)?;
    let c = prog.config();
    println!("config {:#x} {} {}", c.opt, c.stack_limit(), c.num_args());
    println!("counts {} {} {} {} {}", prog.count_states, prog.count_params, prog.count_obs, prog.count_diffs, prog.count_loops);
    println!("stack {:?} loops {}", prog.builder.count_stack, prog.builder.count_loops);
    for x in &prog.builder.consts { println!("const {:016x}", x.to_bits()); }
    let mut ft: Vec<&String> = prog.builder.ft.iter().collect();
    ft.sort();
    for f in ft { println!("function {}", f); }
    for (i, ins) in mir.code.iter().enumerate() {
        match &ins {
            MirInstruction::Call { label, num_args } => println!("{:05}\tcall {} / {}", i, label, num_args),
            _ => println!("{:05}\t{:?}", i, ins),
        }
    }
    for x in &mir.consts { println!("mir const {:016x}", x.to_bits()); }
    let mut r: Vec<u64> = reals.iter().map(code).collect();
    r.sort();
    for x in r { println!("real {:x}", x); }
    Ok(())
}

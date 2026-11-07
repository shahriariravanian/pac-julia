// The header and the MIR of a .sjb as symjit reads it, in the text of sjb.jl's test.
use std::fs::File;
use symjit::backend::{Complexifier, Loc, Mir, MirInstruction};
use symjit::{Application, CompilerType, Config, MirWriter};

fn code(l: &Loc) -> u64 {
    match l {
        Loc::Mem(i) => 0x100000000 | *i as u64,
        Loc::Stack(i) => 0x200000000 | *i as u64,
        Loc::Param(i) => 0x300000000 | *i as u64,
    }
}

fn print_code(mir: &Mir) {
    for (i, ins) in mir.code.iter().enumerate() {
        match &ins {
            MirInstruction::Call { label, num_args } => println!("{:05}\tcall {} / {}", i, label, num_args),
            _ => println!("{:05}\t{:?}", i, ins),
        }
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
    if path == "--run" {
        // the lowered MIR run by symjit's bytecode interpreter (Mir::exec_instruction): `--run FILE
        // INPUT`, INPUT = mem, stack, params (each a u64 count and f64 values); prints mem and stack
        path = args.next().unwrap();
        let input = std::fs::read(args.next().unwrap())?;
        let mut pos = 0;
        let mut next = || {
            let x = u64::from_le_bytes(input[pos..pos + 8].try_into().unwrap());
            pos += 8;
            x
        };
        let mut section = || {
            let n = next() as usize;
            (0..n).map(|_| f64::from_bits(next())).collect::<Vec<f64>>()
        };
        let (mut mem, mut stack, params) = (section(), section(), section());
        let config = Config::new(CompilerType::ByteCode, 0)?;
        let (prog, reals, saved) = Application::read_sjb(&mut File::open(&path)?, &config)?;
        let mir = Complexifier::new(&reals, prog.config().clone()).complexify(&saved)?;
        let mut regs = vec![0.0f64; 66];
        mir.exec_instruction(&mut mem, &mut stack, &mut regs, &params);
        for x in &mem { println!("mem {:016x}", x.to_bits()); }
        for x in &stack { println!("stack {:016x}", x.to_bits()); }
        return Ok(());
    }
    if path == "--lowered" {
        // the MIR as symjit lowers it (symjit-utils' SavedModel::read_with): `--lowered [--stable
        // true|false] FILE`
        path = args.next().unwrap();
        let mut stable = None;
        if path == "--stable" {
            stable = Some(args.next().unwrap());
            path = args.next().unwrap();
        }
        let config = Config::new(CompilerType::ByteCode, 0)?;
        let (mut prog, reals, saved) = Application::read_sjb(&mut File::open(&path)?, &config)?;
        if let Some(v) = stable {
            prog.builder.config.set_option("stable_math", &v).map_err(|e| anyhow::anyhow!("{}", e))?;
        }
        let mir = if prog.config().is_complex() {
            Complexifier::new(&reals, prog.config().clone()).complexify(&saved)?
        } else {
            saved
        };
        print_code(&mir);
        for x in &mir.consts { println!("mir const {:016x}", x.to_bits()); }
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

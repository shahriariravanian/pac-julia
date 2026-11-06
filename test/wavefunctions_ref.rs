// SPDX-License-Identifier: 0BSD
// The reference for test_wavefunctions.jl: the f64 source wavefunctions of RustiCol's
// rusticol-core/src/wavefunctions.rs (copied unchanged below main), dispatched as pacuda's runner
// does (`source_values`) and multiplied by the source factor as the runner does (`run_sources`).
// stdin: one case per line, `family orientation helicity chirality` then the bits (hex u64) of
// mass E px py pz factor.re factor.im; stdout: per case the bits of the 4 components (re im).
// Build: rustc -O wavefunctions_ref.rs
use std::io::{self, BufRead, Write};
use std::ops::Mul;

#[derive(Clone, Copy, Debug)]
struct Complex<T> {
    re: T,
    im: T,
}
impl Complex<f64> {
    fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
}
impl Mul<f64> for Complex<f64> {
    type Output = Complex<f64>;
    fn mul(self, r: f64) -> Self {
        Complex::new(self.re * r, self.im * r)
    }
}

fn source_values(family: u32, anti: bool, h: i32, ch: i32, mass: f64, p: [f64; 4]) -> Vec<Complex<f64>> {
    match family {
        0 => vec![c64(1.0, 0.0)],
        1 if !anti => ext_quark_weyl_array(p, h, ch).to_vec(),
        1 => ext_antiquark_weyl_array(p, h, ch).to_vec(),
        2 if !anti => ext_quark_dirac_massive(p, h, mass).to_vec(),
        2 => ext_antiquark_dirac_massive(p, h, mass).to_vec(),
        _ if mass == 0.0 => ext_gluon(p, h).to_vec(),
        _ => ext_massive_vector(p, h, mass).to_vec(),
    }
}

fn main() {
    let f = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
    let mut out = io::BufWriter::new(io::stdout());
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let t: Vec<&str> = line.split_whitespace().collect();
        let (family, anti) = (t[0].parse().unwrap(), t[1] == "1");
        let (h, ch): (i32, i32) = (t[2].parse().unwrap(), t[3].parse().unwrap());
        let mass = f(t[4]);
        let p = [f(t[5]), f(t[6]), f(t[7]), f(t[8])];
        let (fr, fi) = (f(t[9]), f(t[10]));
        let v = source_values(family, anti, h, ch, mass, p);
        let mut words = Vec::new();
        for k in 0..4 {
            let (re, im) = match v.get(k) {
                Some(c) => (fr * c.re - fi * c.im, fr * c.im + fi * c.re),
                None => (0.0, 0.0),
            };
            words.push(format!("{:016x} {:016x}", re.to_bits(), im.to_bits()));
        }
        writeln!(out, "{} {}", v.len(), words.join(" ")).unwrap();
    }
}

// ---- copied from rusticol-core/src/wavefunctions.rs (`pub(super)` dropped) ----

fn c64(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn negate(momentum: [f64; 4]) -> [f64; 4] {
    [-momentum[0], -momentum[1], -momentum[2], -momentum[3]]
}

fn fortran_sign(value: f64, sign_source: f64) -> f64 {
    value.abs().copysign(sign_source)
}

fn ext_quark_dirac(momentum: [f64; 4], helicity: i32) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            (energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(helicity as f64 * px / sqp0p3, -py / sqp0p3)
        };
        if helicity == 1 {
            return [chi1, chi2, c64(0.0, 0.0), c64(0.0, 0.0)];
        }
        return [c64(0.0, 0.0), c64(0.0, 0.0), chi2, chi1];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        -(-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(-(helicity as f64) * (-px) / sqp0p3, py / sqp0p3)
    };
    if -helicity == 1 {
        [chi1, chi2, c64(0.0, 0.0), c64(0.0, 0.0)]
    } else {
        [c64(0.0, 0.0), c64(0.0, 0.0), chi2, chi1]
    }
}

fn ext_antiquark_dirac(momentum: [f64; 4], helicity: i32) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            -(energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(-(helicity as f64) * px / sqp0p3, py / sqp0p3)
        };
        if -helicity == 1 {
            return [c64(0.0, 0.0), c64(0.0, 0.0), chi1, chi2];
        }
        return [chi2, chi1, c64(0.0, 0.0), c64(0.0, 0.0)];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        (-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(helicity as f64 * (-px) / sqp0p3, -py / sqp0p3)
    };
    if helicity == 1 {
        [c64(0.0, 0.0), c64(0.0, 0.0), chi1, chi2]
    } else {
        [chi2, chi1, c64(0.0, 0.0), c64(0.0, 0.0)]
    }
}

fn ext_quark_dirac_massive(
    momentum: [f64; 4],
    helicity: i32,
    mass: f64,
) -> [Complex<f64>; 4] {
    if mass.abs() < 1.0e-8 {
        return ext_quark_dirac(momentum, helicity);
    }
    let [energy, px, py, pz] = momentum;
    let nsf = if energy > 0.0 { 1 } else { -1 };
    let nh = nsf * helicity;
    let pp = (px * px + py * py + pz * pz).sqrt().abs();
    let omega1 = (energy.abs() + pp).sqrt();
    let omega2 = mass / omega1;
    let omega = [omega1, omega2];
    let sf1 = (1 + nsf + (1 - nsf) * nh) as f64 * 0.5;
    let sf2 = (1 + nsf - (1 - nsf) * nh) as f64 * 0.5;
    let ip = ((3 + nh) / 2 - 1) as usize;
    let im = ((3 - nh) / 2 - 1) as usize;
    let sfomeg = [sf1 * omega[ip], sf2 * omega[im]];
    let (signed_px, signed_py, signed_pz) = if energy > 0.0 {
        (px, py, pz)
    } else {
        (-px, -py, -pz)
    };
    let pp3 = (pp + signed_pz).max(0.0);
    let chi1 = if pp == 0.0 {
        c64(1.0, 0.0)
    } else {
        c64((pp3 * 0.5 / pp).sqrt(), 0.0)
    };
    let chi2 = if pp3 == 0.0 || pp == 0.0 {
        c64(-(nh as f64), 0.0)
    } else {
        let denom = (2.0 * pp * pp3).sqrt();
        c64((nh as f64) * signed_px / denom, -signed_py / denom)
    };
    let chi = [chi1, chi2];
    [
        chi[im] * sfomeg[1],
        chi[ip] * sfomeg[1],
        chi[im] * sfomeg[0],
        chi[ip] * sfomeg[0],
    ]
}

fn ext_antiquark_dirac_massive(
    momentum: [f64; 4],
    helicity: i32,
    mass: f64,
) -> [Complex<f64>; 4] {
    if mass.abs() < 1.0e-8 {
        return ext_antiquark_dirac(momentum, helicity);
    }
    let [energy, px, py, pz] = momentum;
    let nsf = if energy > 0.0 { -1 } else { 1 };
    let nh = nsf * helicity;
    let pp = (px * px + py * py + pz * pz).sqrt().abs();
    let omega1 = (energy.abs() + pp).sqrt();
    let omega2 = mass / omega1;
    let omega = [omega1, omega2];
    let sf1 = (1 + nsf + (1 - nsf) * nh) as f64 * 0.5;
    let sf2 = (1 + nsf - (1 - nsf) * nh) as f64 * 0.5;
    let ip = ((3 + nh) / 2 - 1) as usize;
    let im = ((3 - nh) / 2 - 1) as usize;
    let sfomeg = [sf1 * omega[ip], sf2 * omega[im]];
    let (signed_px, signed_py, signed_pz) = if energy > 0.0 {
        (px, py, pz)
    } else {
        (-px, -py, -pz)
    };
    let pp3 = (pp + signed_pz).max(0.0);
    let chi1 = if pp == 0.0 {
        c64(1.0, 0.0)
    } else {
        c64((pp3 * 0.5 / pp).sqrt(), 0.0)
    };
    let chi2 = if pp3 == 0.0 || pp == 0.0 {
        c64(-(nh as f64), 0.0)
    } else {
        let denom = (2.0 * pp * pp3).sqrt();
        c64((nh as f64) * signed_px / denom, signed_py / denom)
    };
    let chi = [chi1, chi2];
    [
        chi[im] * sfomeg[0],
        chi[ip] * sfomeg[0],
        chi[im] * sfomeg[1],
        chi[ip] * sfomeg[1],
    ]
}

fn ext_massive_vector(
    momentum: [f64; 4],
    helicity: i32,
    mass: f64,
) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    if energy < 0.0 {
        return ext_massive_vector(negate(momentum), -helicity, mass);
    }
    let sqh = 0.5f64.sqrt();
    let hel = helicity as f64;
    let nsvahl = helicity.abs() as f64;
    let pt2 = px * px + py * py;
    let pp = energy.min((pt2 + pz * pz).sqrt());
    let pt = pp.min(pt2.sqrt());
    let hel0 = 1.0 - hel.abs();
    if pp == 0.0 {
        return [
            c64(0.0, 0.0),
            c64(-hel * sqh, 0.0),
            c64(0.0, nsvahl * sqh),
            c64(hel0, 0.0),
        ];
    }
    let emp = energy / (mass * pp);
    let wf0 = c64(hel0 * pp / mass, 0.0);
    let wf3 = c64(hel0 * pz * emp + hel * pt / pp * sqh, 0.0);
    let (wf1, wf2) = if pt != 0.0 {
        let pzpt = pz / (pp * pt) * sqh * hel;
        (
            c64(hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh),
            c64(hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh),
        )
    } else {
        (
            c64(-hel * sqh, 0.0),
            c64(0.0, nsvahl * fortran_sign(sqh, pz)),
        )
    };
    [wf0, wf1, wf2, wf3]
}

fn ext_gluon(momentum: [f64; 4], helicity: i32) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    let sqh = 0.5f64.sqrt();
    if energy > 0.0 {
        let hel = helicity as f64;
        let pp = energy;
        let pt = (px * px + py * py).sqrt();
        let wf3 = c64(hel * pt / pp * sqh, 0.0);
        let (wf1, wf2) = if pt != 0.0 {
            let pzpt = pz / (pp * pt) * sqh * hel;
            (
                c64(-px * pzpt, -py / pt * sqh),
                c64(-py * pzpt, px / pt * sqh),
            )
        } else {
            (c64(-hel * sqh, 0.0), c64(0.0, fortran_sign(sqh, pz)))
        };
        return [c64(0.0, 0.0), wf1, wf2, wf3];
    }
    let hel = -helicity as f64;
    let pp = -energy;
    let pt = (px * px + py * py).sqrt();
    let wf3 = c64(hel * pt / pp * sqh, 0.0);
    let (wf1, wf2) = if pt != 0.0 {
        let pzpt = -pz / (pp * pt) * sqh * hel;
        (
            c64(px * pzpt, py / pt * sqh),
            c64(py * pzpt, -px / pt * sqh),
        )
    } else {
        (c64(-hel * sqh, 0.0), c64(0.0, -fortran_sign(sqh, pz)))
    };
    [c64(0.0, 0.0), wf1, wf2, wf3]
}

fn ext_quark_weyl_array(
    momentum: [f64; 4],
    helicity: i32,
    chirality: i32,
) -> [Complex<f64>; 2] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            (energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(helicity as f64 * px / sqp0p3, -py / sqp0p3)
        };
        if helicity == 1 && chirality == 1 {
            return [chi1, chi2];
        }
        if helicity == -1 && chirality == -1 {
            return [chi2, chi1];
        }
        return [c64(0.0, 0.0), c64(0.0, 0.0)];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        -(-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(-(helicity as f64) * (-px) / sqp0p3, py / sqp0p3)
    };
    if helicity == -1 && chirality == 1 {
        [chi1, chi2]
    } else if helicity == 1 && chirality == -1 {
        [chi2, chi1]
    } else {
        [c64(0.0, 0.0), c64(0.0, 0.0)]
    }
}

fn ext_antiquark_weyl_array(
    momentum: [f64; 4],
    helicity: i32,
    chirality: i32,
) -> [Complex<f64>; 2] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            -(energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(-(helicity as f64) * px / sqp0p3, py / sqp0p3)
        };
        if helicity == 1 && chirality == 1 {
            return [chi2, chi1];
        }
        if helicity == -1 && chirality == -1 {
            return [chi1, chi2];
        }
        return [c64(0.0, 0.0), c64(0.0, 0.0)];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        (-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(helicity as f64 * (-px) / sqp0p3, -py / sqp0p3)
    };
    if helicity == -1 && chirality == 1 {
        [chi2, chi1]
    } else if helicity == 1 && chirality == -1 {
        [chi1, chi2]
    } else {
        [c64(0.0, 0.0), c64(0.0, 0.0)]
    }
}

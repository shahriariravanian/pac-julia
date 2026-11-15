// metal_shim_run.cpp: a program's generated runner.metal (compile_metal) on the CPU, through
// msl_shim.h: uir_points called for every point in turn, as the GPU's threads would run them. Built
// per program by test/test_metal_shim.jl: c++ -std=c++17 -O1 -ffp-contract=off -fPIC -shared
// -Wno-attributes -I c/metal -I <the program's folder>.
//
// shim_run(program.bin, momenta[point][4 legs], points, out[point][outputs]): the momenta staged as
// df64 in the launch layout the Metal host uses (row 4 leg + j, points per row), the outputs read
// back as hi + lo. Returns the status (0, or a failing kernel's).

#include "msl_shim.h"
#include "runner.metal"
#include <vector>

extern "C" int shim_run(const unsigned char *program, const double *momenta, size_t points, double *out)
{
    msl_shim_flush_denormals();
    const uir_header *h = (const uir_header *)program;
    size_t legs = h->legs, forms = h->forms, planes = h->planes, amplitudes = h->amplitudes, outputs = h->outputs;
    std::vector<float2> m(4 * legs * points + 1), o(outputs * points + 1), f(4 * forms * points + 1),
        a(2 * amplitudes * points + 1);
    std::vector<float> pl(2 * planes * points + 1);
    for (size_t p = 0; p < points; p++)
        for (size_t k = 0; k < 4 * legs; k++) {
            double x = momenta[p * 4 * legs + k];
            float hi = (float)x;
            m[k * points + p] = float2(hi, (float)(x - (double)hi));
        }
    uir_launch launch = {(uint)points, 0u, (uint)(4 * points), (uint)points, (uint)points};
    atomic_int status(0);
    for (uint i = 0; i < points; i++)
        uir_points(program, launch, m.data(), o.data(), f.data(), a.data(), pl.data(), &status, i);
    for (size_t p = 0; p < points; p++)
        for (size_t k = 0; k < outputs; k++) out[p * outputs + k] = (double)o[k * points + p].x + (double)o[k * points + p].y;
    return status.load();
}

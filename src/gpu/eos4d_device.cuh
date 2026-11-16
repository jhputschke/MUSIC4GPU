// NEOS-4D equation of state (EOS 20) on the device.
//
// A port of EOS_4D::get_tilde_variables and EOS_4D::FourDLInterp
// (src/eos_4D.cpp): the tables are given on a regular grid in the
// temperature and chemical potentials a massless three-flavour parton gas
// would have at the same e, rho_B, rho_Q, rho_S ("tilde" variables), which
// follow from the densities in closed form, so a lookup is that transform and
// one 16-corner interpolation -- no root finding.
//
// Table layout (GPUGrid::eos4d_tab): the six tables of EOS_4D interleaved per
// node, node index as EOS_4D::index(), values in this order:
//   P [1/fm^4], T, mu_B, mu_Q, mu_S [1/fm], c_s [1]
// so one corner fetch brings all six values of a node into the same cache
// lines. Single precision throughout (the tables are float on disk and the
// CPU interpolates in float as well); the CPU forms the fractional indices in
// double, so the two agree to float rounding, not bit for bit.
//
// Only included from .cu translation units.

#pragma once
#include "gpu_types.h"

#ifndef DFI
#define DFI __device__ __forceinline__
#endif

struct GPUEos4DState {
    float P;        // pressure [1/fm^4], >= 1e-16 as on the CPU
    float T;        // temperature [1/fm], >= 1e-16
    float muB, muQ, muS;   // [1/fm]
    float cs2;      // squared speed of sound, clamped to [0.01, 1/3]
    float dPde;     // partial derivatives of P at fixed other densities
    float dPdrhob;
    float dPdrhoq;
    float dPdrhos;
};

// what a lookup has to provide (to save the loads and the arithmetic)
#define GPU_EOS4D_P       1   // P only
#define GPU_EOS4D_DERIV   2   // + dP/de, dP/drho_i
#define GPU_EOS4D_THERMO  4   // + T, mu_B, mu_Q, mu_S, cs2

DFI void gpu_eos4d_tilde(float e, float rhob, float rhoq, float rhos,
                         const GPUEos4DParams& p, float tv[4]) {
    // T~^4 = e / (3 alpha_Nf): e = 3 alpha_Nf T~^4 for the parton gas
    const float Tt_sq = sqrtf(fmaxf(e, 0.f) * (1.f / 3.f) * p.one_over_alphaNf);
    tv[0] = sqrtf(Tt_sq);
    // a vacuum cell (e = 0) has no defined mu~; the CPU divides by zero there.
    // Put it at mu~ = 0, which the clamping below would do with any finite
    // value outside the table as well.
    const float inv = (Tt_sq > 1.e-30f) ? 1.f / Tt_sq : 0.f;
    tv[1] = ( 5.f*rhob -     rhoq + 2.f*rhos) * inv;
    tv[2] = (-    rhob + 2.f*rhoq -     rhos) * inv;
    tv[3] = ( 2.f*rhob -     rhoq + 2.f*rhos) * inv;
}

DFI void gpu_eos4d_lookup(float e, float rhob, float rhoq, float rhos,
                          const float* __restrict__ tab,
                          const GPUEos4DParams& p, int what,
                          GPUEos4DState& s) {
    float tv[4];
    gpu_eos4d_tilde(e, rhob, rhoq, rhos, p, tv);

    const float frac[4] = {(tv[0] - p.T0)   / p.dT,
                           (tv[1] - p.mub0) / p.dmub,
                           (tv[2] - p.muq0) / p.dmuq,
                           (tv[3] - p.mus0) / p.dmus};
    const int nmax[4] = {p.N_T, p.N_mub, p.N_muq, p.N_mus};
    int   i0[4];
    float w1[4];   // weight of the upper node in each direction
    for (int d = 0; d < 4; d++) {
        // static_cast<int> truncates towards zero, then the clamp, as on the CPU
        int i = (isfinite(frac[d]) ? (int)fmaxf(fminf(frac[d], 1.e8f), -1.e8f) : 0);
        i = min(max(i, 0), nmax[d] - 2);
        i0[d] = i;
        w1[d] = fminf(1.f, fmaxf(0.f, isfinite(frac[d]) ? frac[d] - (float)i : 0.f));
    }

    const bool thermo = (what & GPU_EOS4D_THERMO) != 0;
    const bool deriv  = (what & GPU_EOS4D_DERIV)  != 0;

    float val[GPU_EOS4D_NVAL] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float Pc[16];   // P at the corners, for the derivatives
    const int nval = thermo ? GPU_EOS4D_NVAL : 1;

    for (int corner = 0; corner < 16; corner++) {
        const int bT = (corner >> 0) & 1;
        const int bB = (corner >> 1) & 1;
        const int bQ = (corner >> 2) & 1;
        const int bS = (corner >> 3) & 1;
        const int iT = i0[0] + bT, iB = i0[1] + bB;
        const int iQ = i0[2] + bQ, iS = i0[3] + bS;
        // EOS_4D::index: ((i_T*N_mus + i_mus)*N_muq + i_muq)*N_mub + i_mub
        const size_t node = ((static_cast<size_t>(iT) * p.N_mus + iS)
                             * p.N_muq + iQ) * p.N_mub + iB;
        const float* v = tab + node * GPU_EOS4D_NVAL;

        const float wT = bT ? w1[0] : 1.f - w1[0];
        const float wB = bB ? w1[1] : 1.f - w1[1];
        const float wQ = bQ ? w1[2] : 1.f - w1[2];
        const float wS = bS ? w1[3] : 1.f - w1[3];
        const float w = wT * wB * wQ * wS;
        Pc[corner] = __ldg(&v[0]);
        val[0] += w * Pc[corner];
        for (int k = 1; k < nval; k++) val[k] += w * __ldg(&v[k]);
    }

    // the derivative along direction d: the weighted mean over the eight
    // corner pairs that differ in d of P(upper) - P(lower), divided by the
    // spacing. FourDLInterp sums each face first and subtracts the two sums,
    // which loses most digits in float (the faces differ by little); taking
    // the pair differences first does not.
    float dP[4] = {0.f, 0.f, 0.f, 0.f};
    if (deriv) {
        for (int d = 0; d < 4; d++) {
            for (int corner = 0; corner < 16; corner++) {
                if ((corner >> d) & 1) continue;
                float o = 1.f;
                for (int dd = 0; dd < 4; dd++) {
                    if (dd == d) continue;
                    o *= ((corner >> dd) & 1) ? w1[dd] : 1.f - w1[dd];
                }
                dP[d] += o * (Pc[corner | (1 << d)] - Pc[corner]);
            }
        }
    }

    const float small_eps = 1.e-16f;
    s.P = fmaxf(small_eps, val[0]);
    if (thermo) {
        s.T   = fmaxf(small_eps, val[1]);
        s.muB = val[2];
        s.muQ = val[3];
        s.muS = val[4];
        s.cs2 = fminf(1.f / 3.f, fmaxf(0.01f, val[5] * val[5]));
    }
    if (deriv) {
        // T~ clamped to the table, as FourDLInterp does for the derivatives
        const float Tc = fmaxf(p.T0, fminf(p.T_max, tv[0]));
        const float T1 = p.T0 + (float)i0[0] * p.dT;
        const float T2 = T1 + p.dT;
        const float T1_5 = T1*T1*T1*T1*T1, T2_5 = T2*T2*T2*T2*T2;
        const float dXdT = 5.f * dP[0] / (T2_5 - T1_5) * (Tc*Tc*Tc*Tc);
        const float dXdB = dP[1] / p.dmub;
        const float dXdQ = dP[2] / p.dmuq;
        const float dXdS = dP[3] / p.dmus;
        // 3/(19 pi^2): de/dT~ of the parton gas, with alpha_Nf = 19 pi^2/36
        const float pi2 = 9.8696044010893586f;
        s.dPde    = 3.f / (19.f * pi2 * Tc*Tc*Tc) * dXdT;
        const float iT2 = 1.f / (Tc * Tc);
        s.dPdrhob = ( 5.f*dXdB -     dXdQ + 2.f*dXdS) * iT2;
        s.dPdrhoq = (-    dXdB + 2.f*dXdQ -     dXdS) * iT2;
        s.dPdrhos = ( 2.f*dXdB -     dXdQ + 2.f*dXdS) * iT2;
    }
}

// entropy density [1/fm^3] from the thermodynamic relation, as
// EOS_base::get_entropy for EOS 20
DFI float gpu_eos4d_entropy(float e, float rhob, float rhoq, float rhos,
                            const GPUEos4DState& s) {
    const float f = (e + s.P - s.muB*rhob - s.muS*rhos - s.muQ*rhoq)
                    / (s.T + 1.e-16f);
    return fmaxf(1.e-16f, f);
}

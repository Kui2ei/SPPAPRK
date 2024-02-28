// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdio.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__
#endif
#if defined(__CUDA_ARCH__) && !defined(__SPPARK_FF_MONT_T_CUH__)
# define __SPPARK_FF_MONT_T_CUH__

# include <cstddef>
# include <cstdint>

# define inline __device__ __forceinline__
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

// 0x0e0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d
// 0x0e0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d
namespace device {
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_P[8] = {
        TO_CUDA_T(0x3c208c16d87cfd47), TO_CUDA_T(0x97816a916871ca8d),
        TO_CUDA_T(0xb85045b68181585d), TO_CUDA_T(0x30644e72e131a029)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_RR[8] = { /* (1<<512)%P */
        TO_CUDA_T(0xf32cfc5b538afa89), TO_CUDA_T(0xb5e71911d44501fb),
        TO_CUDA_T(0x47ab1eff0a417ff6), TO_CUDA_T(0x06d89f71cab8351f),
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_one[8] = { /* (1<<256)%P */
        TO_CUDA_T(0xd35d438dc58f0d9d), TO_CUDA_T(0x0a78eb28f5c70b3d),
        TO_CUDA_T(0x666ea36f7879462c), TO_CUDA_T(0x0e0a77c19a07df2f)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_Px4[8] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xf082305b61f3f51c), TO_CUDA_T(0x5e05aa45a1c72a34),
        TO_CUDA_T(0xe14116da06056176), TO_CUDA_T(0xc19139cb84c680a6)
    };
    //n`(rr_1-nn`=1) 0xd76b780d00bad4acbbc3ac183dcfc3f073dd0cc126dc4ece43b5e10de0f8076
    static __device__ __constant__ const uint32_t ALT_BN128_M0 = 0xe4866389;


    static __device__ __constant__ __align__(16) const uint32_t MNT4753_Fr_P[24] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xd90776e240000001), TO_CUDA_T(0x4ea099170fa13a4f),
        TO_CUDA_T(0xd6c381bc3f005797), TO_CUDA_T(0xb9dff97634993aa4),
        TO_CUDA_T(0x3eebca9429212636), TO_CUDA_T(0xb26c5c28c859a99b),
        TO_CUDA_T(0x99d124d9a15af79d), TO_CUDA_T(0x07fdb925e8a0ed8d),
        TO_CUDA_T(0x5eb7e8f96c97d873), TO_CUDA_T(0xb7f997505b8fafed),
        TO_CUDA_T(0x10229022eee2cdad), TO_CUDA_T(0x0001c4c62d92c411)
    };

    
    static __device__ __constant__ const uint32_t MNT4753_Fr_M0 = 0x3fffffff;


    static __device__ __constant__ __align__(16) const uint32_t BLS12_377_P[12] = {
        TO_CUDA_T(0x8508c00000000001), TO_CUDA_T(0x170b5d4430000000),
        TO_CUDA_T(0x1ef3622fba094800), TO_CUDA_T(0x1a22d9f300f5138f),
        TO_CUDA_T(0xc63b05c06ca1493b), TO_CUDA_T(0x01ae3a4617c510ea)
    };

    
    static __device__ __constant__ /*const*/ uint32_t BLS12_377_M0 = 0xffffffff;

         static __device__ __constant__ __align__(16) const uint32_t curve448_P[14] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xfffffffeffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff)
    };


    static __device__ __constant__ /*const*/ uint32_t curve448_M0 = 0x1;


 static __device__ __constant__ __align__(16) const uint32_t Curve1024_P0[32] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xfffffffffffffffe), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff)
    };



    static __device__ __constant__ /*const*/ uint32_t Curve1024_M0 = 0x1;

 static __device__ __constant__ __align__(16) const uint32_t Curve2048_P0[64] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xfffffffffffffffe), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff)
    };



    static __device__ __constant__ /*const*/ uint32_t Curve2048_M0 = 0x1;

    static __device__ __constant__ __align__(16) const uint32_t Curve4096_P0[128] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xfffffffffffffffe), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff)
    };



    static __device__ __constant__ /*const*/ uint32_t Curve4096_M0 = 0x1;
}

void __global__  func1(uint32_t res[]);
//
// To instantiate declare modulus as __device__ __constant___ const and
// complement it with its factual bit-length and the corresponding 32-bit
// Motgomery factor. Bit-length has to be such that (N+31)/32 is even
// and not less than 4.
//
// Special note about M0 being declared as uint32_t& [as opposed to just
// uint32_t]. It was noted that if M0 is 0xffffffff, CUDA compiler
// generates suboptimal code for Montgomery reduction. The way to work
// around the problem is to prevent compiler from viewing it as constant.
// For this reason it's suggested to declare the parameter as following:
//
//    __device__ __constant__ /*const*/ my_M0 = <literal>;
//
template<const size_t N, const uint32_t MOD[(N+31)/32], const uint32_t& M0,
         const uint32_t RR[(N+31)/32], const uint32_t ONE[(N+31)/32],
         const uint32_t MODx[(N+31)/32] = MOD>
class __align__(((N+63)/64)&1 ? 8 : 16) mont_t {
        
    public:
        static const size_t nbits = N;
        static constexpr size_t __device__ bit_length() { return N; }
        static const uint32_t degree = 1;
        using mem_t = mont_t;
        static const size_t n = (N+31)/32;
        
    public:
        uint32_t even[n];

    private:
        static inline void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi,
        size_t n_=n)
        {
            for (size_t j = 0; j < n_; j += 2)
                asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
                    : "=r"(acc[j]), "=r"(acc[j+1])
                    : "r"(a[j]), "r"(bi));
        }

        static inline void cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi,
                                size_t n_=n)
        {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1])
                : "r"(a[0]), "r"(bi));
            for (size_t j = 2; j < n_; j += 2)
                asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
                    : "+r"(acc[j]), "+r"(acc[j+1])
                    : "r"(a[j]), "r"(bi));
            // return carry flag
        }

        static inline void cadd_n(uint32_t* acc, const uint32_t* a, size_t n_=n)
        {
            asm("add.cc.u32 %0, %0, %1;" : "+r"(acc[0]) : "r"(a[0]));
            for (size_t i = 1; i < n_; i++)
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(acc[i]) : "r"(a[i]));
            // return carry flag
        }

        class wide_t {
        private:
            union {
                uint32_t even[2*n];
                mont_t s[2];
            };

        public:
            inline uint32_t& operator[](size_t i)               { return even[i]; }
            inline const uint32_t& operator[](size_t i) const   { return even[i]; }
            inline operator mont_t()
            {
                s[0].mul_by_1();
                return s[0] + s[1];
            }
            inline void final_sub(uint32_t carry, uint32_t* tmp)
            {   s[1].final_sub(carry, tmp);   }

            inline wide_t() {}

        private:
            static inline void mad_row(uint32_t* odd, uint32_t* even,
                                    const uint32_t* a, uint32_t bi, size_t n_=n)
            {
                cmad_n(odd, a+1, bi, n_-2);
                asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
                    : "=r"(odd[n_-2]), "=r"(odd[n_-1])
                    : "r"(a[n_-1]), "r"(bi));

                cmad_n(even, a, bi, n_);
                asm("addc.u32 %0, %0, 0;" : "+r"(odd[n_-1]));
            }

        public:
            inline wide_t(const mont_t& a, const mont_t& b)     //// |a|*|b|
            {
                size_t i = 0;
                uint32_t odd[2*n-2];

                mul_n(even, &a[0], b[0]);
                mul_n(odd,  &a[1], b[0]);
                ++i; mad_row(&even[i+1], &odd[i-1], &a[0], b[i]);

                #pragma unroll
                while (i < n-2) {
                    ++i; mad_row(&odd[i],    &even[i],  &a[0], b[i]);
                    ++i; mad_row(&even[i+1], &odd[i-1], &a[0], b[i]);
                }

                // merge |even| and |odd|
                cadd_n(&even[1], &odd[0], 2*n-2);
                asm("addc.u32 %0, %0, 0;" : "+r"(even[2*n-1]));
            }
        private:
            static inline void qad_row(uint32_t* odd, uint32_t* even,
                                    const uint32_t* a, uint32_t bi, size_t n)
            {
                cmad_n(odd, a, bi, n-2);
                asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
                    : "=r"(odd[n-2]), "=r"(odd[n-1])
                    : "r"(a[n-2]), "r"(bi));

                cmad_n(even, a+1, bi, n-2);
                asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
            }
        public:
            inline wide_t(const mont_t& a)                      //// |a|**2
            {
                size_t i = 0, j;
                uint32_t odd[2*n-2];

                // perform |a[i]|*|a[j]| for all j>i
                mul_n(even+2, &a[2], a[0], n-2);
                mul_n(odd,    &a[1], a[0], n);

                #pragma unroll
                while (i < n-4) {
                    ++i; mad_row(&even[2*i+2], &odd[2*i], &a[i+1], a[i], n-i-1);
                    ++i; qad_row(&odd[2*i], &even[2*i+2], &a[i+1], a[i], n-i);
                }

                asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
                    : "=r"(even[2*n-4]), "=r"(even[2*n-3])
                    : "r"(a[n-1]), "r"(a[n-3]));
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
                    : "+r"(odd[2*n-6]), "+r"(odd[2*n-5])
                    : "r"(a[n-2]), "r"(a[n-3]));
                asm("addc.u32 %0, %0, 0;" : "+r"(even[2*n-3]));

                asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
                    : "=r"(odd[2*n-4]), "=r"(odd[2*n-3])
                    : "r"(a[n-1]), "r"(a[n-2]));

                // merge |even[2:]| and |odd[1:]|
                cadd_n(&even[2], &odd[1], 2*n-4);
                asm("addc.u32 %0, %1, 0;" : "=r"(even[2*n-2]) : "r"(odd[2*n-3]));

                // double |even|
                even[0] = 0;
                asm("add.cc.u32 %0, %1, %1;" : "=r"(even[1]) : "r"(odd[0]));
                for (j = 2; j < 2*n-1; j++)
                    asm("addc.cc.u32 %0, %0, %0;" : "+r"(even[j]));
                asm("addc.u32 %0, 0, 0;" : "=r"(even[j]));

                // accumulate "diagonal" |a[i]|*|a[i]| product
                i = 0;
                asm("mad.lo.cc.u32 %0, %2, %2, %0; madc.hi.cc.u32 %1, %2, %2, %1;"
                    : "+r"(even[2*i]), "+r"(even[2*i+1])
                    : "r"(a[i]));
                for (++i; i < n; i++)
                    asm("madc.lo.cc.u32 %0, %2, %2, %0; madc.hi.cc.u32 %1, %2, %2, %1;"
                        : "+r"(even[2*i]), "+r"(even[2*i+1])
                        : "r"(a[i]));
            }
        };

    private:
        inline operator const uint32_t*() const             { return even;    }
        inline operator uint32_t*()                         { return even;    }

    public:
        inline uint32_t& operator[](size_t i)               { return even[i]; }
        inline const uint32_t& operator[](size_t i) const   { return even[i]; }
        inline size_t len() const                           { return n;       }

        inline mont_t() {}
        // write in 
        inline mont_t(const uint32_t *p)
        {
            for (size_t i = 0; i < n; i++)
                even[i] = p[i];
        }


    // write back
        inline void store(uint32_t *p) const
        {
            for (size_t i = 0; i < n; i++)
                p[i] = even[i];
        }

        inline mont_t& operator+=(const mont_t& b)
        {
            cadd_n(&even[0], &b[0]);
            final_subc();
            return *this;
        }
        friend inline mont_t operator+(mont_t a, const mont_t& b)
        {   return a += b;   }

        friend inline mont_t operator<<(mont_t a, unsigned l)
        {   return a <<= l;   }

        inline mont_t& operator>>=(unsigned r)
        {
            size_t i;
            uint32_t tmp[n+1];

            while (r--) {
                tmp[n] = 0 - (even[0]&1);
                for (i = 0; i < n; i++)
                    tmp[i] = MOD[i] & tmp[n];

                cadd_n(&tmp[0], &even[0]);
                if (N%32 == 0)
                    asm("addc.u32 %0, 0, 0;" : "=r"(tmp[n]));

                for (i = 0; i < n-1; i++)
                    asm("shf.r.wrap.b32 %0, %1, %2, 1;"
                        : "=r"(even[i]) : "r"(tmp[i]), "r"(tmp[i+1]));
                if (N%32 == 0)
                    asm("shf.r.wrap.b32 %0, %1, %2, 1;"
                        : "=r"(even[i]) : "r"(tmp[i]), "r"(tmp[i+1]));
                else
                    even[i] = tmp[i] >> 1;
            }

            return *this;
        }
        friend inline mont_t operator>>(mont_t a, unsigned r)
        {   return a >>= r;   }

        inline mont_t& operator-=(const mont_t& b)
        {
            size_t i;
            uint32_t tmp[n], borrow;

            asm("sub.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(b[0]));
            for (i = 1; i < n; i++)
                asm("subc.cc.u32 %0, %0, %1;" : "+r"(even[i]) : "r"(b[i]));
            asm("subc.u32 %0, 0, 0;" : "=r"(borrow));

            asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
            for (i = 1; i < n-1; i++)
                asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
            asm("addc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));

            asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(borrow));
            for (i = 0; i < n; i++)
                asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));
            asm("}");

            return *this;
        }
        friend inline mont_t operator-(mont_t a, const mont_t& b)
        {   return a -= b;   }

        inline mont_t operator-() const
        {   return cneg(*this, true);   }

    private:
        static inline void madc_n_rshift(uint32_t* odd, const uint32_t *a, uint32_t bi)
        {
            for (size_t j = 0; j < n-2; j += 2)
                asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
                    : "=r"(odd[j]), "=r"(odd[j+1])
                    : "r"(a[j]), "r"(bi), "r"(odd[j+2]), "r"(odd[j+3]));
            asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
                : "=r"(odd[n-2]), "=r"(odd[n-1])
                : "r"(a[n-2]), "r"(bi));
        }

        static inline void mad_n_redc(uint32_t *even, uint32_t* odd,
                                    const uint32_t *a, uint32_t bi, bool first=false)
        {
            if (first) {
                mul_n(odd, a+1, bi);
                mul_n(even, a,  bi);
            } else {
                asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
                madc_n_rshift(odd, a+1, bi);
                cmad_n(even, a, bi);
                asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
            }

            uint32_t mi = even[0] * M0;

            cmad_n(odd, MOD+1, mi);
            cmad_n(even, MOD,  mi);
            asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
        }

    public:
        friend inline mont_t operator*(const mont_t& a, const mont_t& b)
        {
            if (N%32 == 0) {
                return wide_t{a, b};
            } else {
                mont_t even, odd;

                #pragma unroll
                for (size_t i = 0; i < n; i += 2) {
                    mad_n_redc(&even[0], &odd[0], &a[0], b[i], i==0);
                    mad_n_redc(&odd[0], &even[0], &a[0], b[i+1]);
                }

                // merge |even| and |odd|
                cadd_n(&even[0], &odd[1], n-1);
                asm("addc.u32 %0, %0, 0;" : "+r"(even[n-1]));

                even.final_sub(0, &odd[0]);

                return even;
            }
        }
        inline mont_t& operator*=(const mont_t& a)
        {   return *this = *this * a;   }

        inline mont_t& sqr()
        {   return *this = wide_t{*this};   }

        inline mont_t operator()(uint32_t p)
        {   return *this^p;   }

        // raise to a constant power, e.g. x^7, to be unrolled at compile time
        inline mont_t& operator^=(int p)
        {
            if (p < 2)
                asm("trap;");

            mont_t sqr = *this;
            if ((p&1) == 0) {
                do {
                    sqr.sqr();
                    p >>= 1;
                } while ((p&1) == 0);
                *this = sqr;
            }
            for (p >>= 1; p; p >>= 1) {
                sqr.sqr();
                if (p&1)
                    *this *= sqr;
            }
            return *this;
        }
        friend inline mont_t operator^(mont_t a, int p)
        {   return p == 2 ? (mont_t)wide_t{a} : a ^= p;   }
        inline mont_t operator()(int p)
        {   return *this^p;   }
        friend inline mont_t sqr(const mont_t& a)
        {   return a^2;   }

        inline void to()    { mont_t t = RR * *this; *this = t; }
        inline void to(const uint32_t a[2*n], bool host_order = true)
        {
            size_t i;

            // load the most significant half
            if (host_order) {
                for (i = 0; i < n; i++)
                    even[i] = a[n + i];
            } else {
                for (i = 0; i < n; i++)
                    asm("prmt.b32 %0, %1, %1, 0x0123;" : "=r"(even[i]) : "r"(a[n - 1 - i]));
            }
            to();

            mont_t lo;

            // load the least significant half
            if (host_order) {
                for (i = 0; i < n; i++)
                    lo[i] = a[i];
            } else {
                for (i = 0; i < n; i++)
                    asm("prmt.b32 %0, %1, %1, 0x0123;" : "=r"(lo[i]) : "r"(a[2*n - 1 - i]));
            }

            cadd_n(&even[0], &lo[0]);
            final_subc();
            to();
        }
        inline void from()  { mont_t t = *this; t.mul_by_1(); *this = t; }
        inline void from(const uint32_t a[2*n], bool host_order = true)
        {
            size_t i;

            // load the least significant half
            if (host_order) {
                for (i = 0; i < n; i++)
                    even[i] = a[i];
            } else {
                for (i = 0; i < n; i++)
                    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(even[i]) : "r"(a[2*n - 1 -i]));
            }
            mul_by_1();

            mont_t hi;

            // load the most significant half
            if (host_order) {
                for (i = 0; i < n; i++)
                    hi[i] = a[n + i];
            } else {
                for (i = 0; i < n; i++)
                    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(hi[i]) : "r"(a[n - 1 - i]));
            }

            cadd_n(&even[0], &hi[0]);
            final_subc();
            to();
        }

    public:
        static inline void mul_by_1_row(uint32_t* even, uint32_t* odd, bool first=false)
        {
            uint32_t mi;

            if (first) {
                mi = even[0] * M0;
                mul_n(odd, MOD+1, mi);
                cmad_n(even, MOD,  mi);
                asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
            } else {
                asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
    # if 1      // do we trust the compiler to *not* touch the carry flag here?
                mi = even[0] * M0;
    # else
                asm("mul.lo.u32 %0, %1, %2;" : "=r"(mi) : "r"(even[0]), "r"(M0));
    # endif
                madc_n_rshift(odd, MOD+1, mi);
                cmad_n(even, MOD, mi);
                asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
            }
        }
        inline void mul_by_1()
        {
            mont_t odd;

            #pragma unroll
            for (size_t i = 0; i < n; i += 2) {
                mul_by_1_row(&even[0], &odd[0], i==0);
                mul_by_1_row(&odd[0], &even[0]);
            }

            cadd_n(&even[0], &odd[1], n-1);
            asm("addc.u32 %0, %0, 0;" : "+r"(even[n-1]));
        }

        inline void final_sub(uint32_t carry, uint32_t* tmp)
        {
            size_t i;
            asm("{ .reg.pred %top;");

            asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
            for (i = 1; i < n; i++)
                asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
            if (N%32 == 0)
                asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(carry));
            else
                asm("subc.u32 %0, 0, 0; setp.eq.u32 %top, %0, 0;" : "=r"(carry));

            for (i = 0; i < n; i++)
                asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));

            asm("}");
        }

        inline void final_subc()
        {
            uint32_t carry, tmp[n];

            asm("addc.u32 %0, 0, 0;" : "=r"(carry));

            asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
            for (size_t i = 1; i < n; i++)
                asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
            asm("subc.u32 %0, %0, 0;" : "+r"(carry));

            asm("{ .reg.pred %top;");
            asm("setp.eq.u32 %top, %0, 0;" :: "r"(carry));
            for (size_t i = 0; i < n; i++)
                asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));
            asm("}");
        }

};


typedef mont_t<254, device::ALT_BN128_P, device::ALT_BN128_M0,
                    device::ALT_BN128_RR, device::ALT_BN128_one,
                    device::ALT_BN128_Px4> fp_mont;


typedef mont_t<753, device::MNT4753_Fr_P, device::MNT4753_Fr_M0,
                    device::MNT4753_Fr_P, device::MNT4753_Fr_P,
                    device::MNT4753_Fr_P> MNT4753_Fr;

typedef mont_t<1024, device::Curve1024_P0, device::Curve1024_M0,
                    device::Curve1024_P0, device::Curve1024_P0,
                    device::Curve1024_P0> CURVE1024_Fr;

typedef mont_t<2047, device::Curve2048_P0, device::Curve2048_M0,
                    device::Curve2048_P0, device::Curve2048_P0,
                    device::Curve2048_P0> CURVE2048_Fr;

typedef mont_t<4095, device::Curve4096_P0, device::Curve4096_M0,
                    device::Curve4096_P0, device::Curve4096_P0,
                    device::Curve4096_P0> CURVE4096_Fr;






void __global__ __launch_bounds__(1024)  func1(uint32_t res[]){
    const uint32_t* p;
    const uint32_t* p1;
    uint32_t arr[128] =  {0x9c0a6f42
        ,0xebb6a0ba
        ,0x37dd94ef
        ,0x6d79fb73
        ,0x7983207e
        ,0x0b9e00b7
        ,0xd8fa3981
        ,0xbf6b3ed2
        ,0xec47b3b6
        ,0x02b47949
        ,0xdcc7abf4
        ,0x0f18044b
        ,0x0906e354
        ,0x672cfe00
        ,0x65a9f075
        ,0xd0caa5f4
        ,0x9a73eaff
        ,0x87604c0d
        ,0x740a5b97
        ,0xb424b881
        ,0xa415e28c
        ,0x2b0009ce
        ,0x95ba33c9
        ,0x25671aba
        ,0x68a56e13
        ,0xae44df47
        ,0x6d6ba5f2
        ,0x08bfd4fe
        ,0xa3697a7e
        ,0x62509cd1
        ,0xc81350fb
        ,0x80b6959c
        ,0xcead05e5
        ,0xd246923c
        ,0xb8a36df5
        ,0xfc512028
        ,0x80dc54fa
        ,0xca57a9b0
        ,0xd504f7bd
        ,0x23596578
        ,0x635d3b55
        ,0x0bb6b26e
        ,0x1676d5bb
        ,0xcb6a7548
        ,0x8dc10ee7
        ,0x5b7c2a1f
        ,0xff779b41
        ,0x3b2848ab
        ,0xcc55cc16
        ,0xcefa60c1
        ,0xd65c561c
        ,0xbb70f64d
        ,0x16d831ba
        ,0x3d436af7
        ,0x77f44368
        ,0xd5f011cc
        ,0x48cdef86
        ,0x7aafa9c6
        ,0x107e0360
        ,0x9c362b0c
        ,0xba8f2d7d
        ,0xc087ce45
        ,0x802c7068
        ,0x7981ddd6
        ,0x05a91b19
        ,0xcadd0630
        ,0xf19bbfdb
        ,0xdf78d97e
        ,0x1ce8d6a8
        ,0x73a8d43e
        ,0xeb02456d
        ,0x5011b85d
        ,0x371112b9
        ,0x8e7d207b
        ,0x433e21ff
        ,0xe48a2325
        ,0xa322cd23
        ,0xd4ac2f85
        ,0x0dbc3dc9
        ,0xcb2d7570
        ,0x1ffc433a
        ,0x7f647e33
        ,0x8fe481c9
        ,0x88cf8dba
        ,0x28f3ec37
        ,0x2a42faf9
        ,0xe891add9
        ,0x7f861ba8
        ,0x4e2ffe49
        ,0xf920f7d1
        ,0x2597f4c6
        ,0x3dc82277
        ,0x868f4226
        ,0xe732b14b
        ,0x5de21876
        ,0xa0f5e592
        ,0x45dbe2ca
        ,0xb566d1a5
        ,0xa705ead2
        ,0x69beaf1c
        ,0x90504600
        ,0xdfd5be24
        ,0xff88d1e9
        ,0xd460c1e8
        ,0x439e2934
        ,0xd97b88e2
        ,0x2de2d2c5
        ,0x746cc9a6
        ,0x8fedd66d
        ,0xe70e2a4b
        ,0x731c389b
        ,0xb44c21b9
        ,0xe48dcf39
        ,0xd73aae49
        ,0x0d585594
        ,0xf8e90c92
        ,0x4ef6861a
        ,0x752c7c90
        ,0x5ba3a788
        ,0xe078f1a0
        ,0xcb291037
        ,0x07309e0c
        ,0xa8cb7709
        ,0x40642ff8
        ,0x3c779f00
        ,0xaaaeee78
        ,0xf0ed822d
        ,0xc0b19ee5};

    // uint32_t arr[8] =  {0xd87cfd47,0x3c208c16,0x6871ca8d,0x97816a91,0x8181585d,0xb85045b6,0xe131a029,0x30644e72};//p,module
    // uint32_t arr[8] =  {0x11111111,0x11111111,0x11111111,0x11111111,0x11111111,0x11111111,0x11111111,0x11111111};
    p = arr;
    CURVE4096_Fr exampleInstance(p);


    // uint32_t arr1[8] = {0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1};
    // R
        // TO_CUDA_T(0xd35d438dc58f0d9d), TO_CUDA_T(0x0a78eb28f5c70b3a),
        // TO_CUDA_T(0x666ea36f7879462c), TO_CUDA_T(0x0e0a77c19a07df2f)
    // uint32_t arr1[8] ={0x538afa89,0xf32cfc5b,0xd44501fb,0xb5e71911,0x0a417ff6,0x47ab1eff,0xcab8351f,0x06d89f71};//RR
    uint32_t arr1[128] ={0x5c285796
        ,0x8d149319
        ,0x490a1357
        ,0x52db1454
        ,0x3939def3
        ,0xa251ea68
        ,0xa692d350
        ,0x355ef91e
        ,0xf3e941d8
        ,0xe6a1c9a1
        ,0xfd199199
        ,0x6af6cf58
        ,0xda500a44
        ,0x666ecd51
        ,0x30e94bc3
        ,0xcd87ca7b
        ,0x314f8097
        ,0xc01a4b6f
        ,0xcb2ff23e
        ,0x502bd0c8
        ,0x6f0ed927
        ,0xb4a6e691
        ,0x0053b87c
        ,0xd73d030b
        ,0x0c6c3f49
        ,0x025c3a54
        ,0x1b84ed45
        ,0xe1810db6
        ,0x47033d9b
        ,0xb8a18bb4
        ,0xb5dcfe75
        ,0x8b9b3738
        ,0x0845dc46
        ,0xea5795c7
        ,0x6b2c9e60
        ,0x70b3b87f
        ,0xc11d49be
        ,0x32211c2f
        ,0xea06a6be
        ,0x03416da9
        ,0xbe4d2f01
        ,0x556e7ba8
        ,0x79d15e48
        ,0xc4f58684
        ,0xb288ba55
        ,0x63ec61dd
        ,0xa1d1d5d5
        ,0x2bd4de93
        ,0x91985701
        ,0xa5585098
        ,0x5358cf43
        ,0x3d5e8964
        ,0xeb06da5c
        ,0x39c77210
        ,0x71017c47
        ,0x0aa1929e
        ,0xbb07f827
        ,0x97707483
        ,0x30633109
        ,0x1bd60690
        ,0x0086fa8f
        ,0x97ec267b
        ,0xc2e4ba81
        ,0x970ddddf
        ,0x91ec010f
        ,0xd9da115f
        ,0xdcdd5ebb
        ,0x10bcb89e
        ,0xbcf79830
        ,0xde664755
        ,0x13f0c4e6
        ,0x027940bc
        ,0x999b7504
        ,0xe3c0b470
        ,0x35b16d65
        ,0xa762d0ae
        ,0xce06ea6b
        ,0xa83a6bde
        ,0x7832d36e
        ,0x13eff3af
        ,0xa4d7760d
        ,0x2f2e226e
        ,0xcbcdd58a
        ,0x4be5c82f
        ,0x42039e4b
        ,0x5c946e19
        ,0xe867a5a0
        ,0x3821cabc
        ,0xa911702d
        ,0x1a6da9e7
        ,0xa6fafdac
        ,0x533eef0a
        ,0x2baa583b
        ,0x600aaa64
        ,0x8a068ee5
        ,0xcc4462a0
        ,0xaa7413fc
        ,0xf80f5f25
        ,0x54020445
        ,0x557170b7
        ,0xad64a4fe
        ,0x6560445a
        ,0x4426fa51
        ,0x75b8768f
        ,0xa36ffaf6
        ,0x2dd20150
        ,0xb729d555
        ,0xc1e5f401
        ,0xd6226ee2
        ,0x10390572
        ,0x1f342167
        ,0x4a857f94
        ,0xd0d260df
        ,0x04762b8a
        ,0xd469af7d
        ,0xaa83490f
        ,0x3961b48d
        ,0x23f9eae6
        ,0xf0b7811d
        ,0x6fed5763
        ,0x6efe2512
        ,0x378f015b
        ,0x492144e3
        ,0x88448a18
        ,0x284374c6
        ,0xa06f5a0c
        ,0x277800b2
        ,0xdfeebc2f};
    p1 = arr1;
    CURVE4096_Fr exampleInstance1(p1);
    exampleInstance*=exampleInstance1;
    for(int i=0;i<exampleInstance.n;i++){
        res[i]=exampleInstance[i];
    }
    
     return;
}

# undef inline
# undef asm
#endif



#include<iostream>
int main(){
    uint32_t res[128]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    uint32_t *res_d;
    cudaMalloc((void**)&res_d,128*sizeof(uint32_t));
    cudaMemcpy(res_d, res,128*sizeof(uint32_t), cudaMemcpyHostToDevice);



int GRIDEDIM = 1;
while (true)
{
    printf("gridDIM = %d\n",GRIDEDIM);


int NUM_REPEATS = 10;
 float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaEventQuery(start);

    func1<<<GRIDEDIM,1024>>>(res_d);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    
    printf("Time = %g +- %g ms.\n", t_ave, t_err);



GRIDEDIM=GRIDEDIM*10;
if(GRIDEDIM==10000000){
    break;
}
}



    
    cudaMemcpy(res,res_d,128*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    for(int i=0;i<128;i++){
        std::cout<<std::hex<<res[i]<<std::endl;
    }
    return 0;
}
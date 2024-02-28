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


    static __device__ __constant__ __align__(16) const uint32_t Curve896_P0[28] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xfffffffffffffffe),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff)
    };



    static __device__ __constant__ /*const*/ uint32_t Curve896_M0 = 0x1;

}

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

typedef mont_t<1023, device::Curve1024_P0, device::Curve1024_M0,
                    device::Curve1024_P0, device::Curve1024_P0,
                    device::Curve1024_P0> CURVE1024_Fr;

typedef mont_t<895, device::Curve896_P0, device::Curve896_M0,
                    device::Curve896_P0, device::Curve896_P0,
                    device::Curve896_P0> CURVE896_Fr;

void __global__ __launch_bounds__(1024) func1(uint32_t res[]){
    const uint32_t* p;
    const uint32_t* p1;
 uint32_t arr[28] =  {0x197958e9
    ,0x5b06dee7
    ,0x942faea1
    ,0xa0f76754
    ,0x2e80f0d5
    ,0x04748ffa
    ,0x892bf79e
    ,0xbb0b295c
    ,0x556edc31
    ,0x9921918c
    ,0xdda073ef
    ,0x9bda0b5f
    ,0x3ef9763f
    ,0x0ff103bc
    ,0x47a18384
    ,0x499b63de
    ,0x0e3e8f37
    ,0x97ff8158
    ,0x79e55fd3
    ,0x0848545f
    ,0x6ec08783
    ,0x32cb88e9
    ,0xb5106359
    ,0x90fffb20
    ,0x536e6db9
    ,0x1e5c9100
    ,0x7bcb2c8c
    ,0xfc13f9c8};

    // uint32_t arr[8] =  {0xd87cfd47,0x3c208c16,0x6871ca8d,0x97816a91,0x8181585d,0xb85045b6,0xe131a029,0x30644e72};//p,module
    // uint32_t arr[8] =  {0x11111111,0x11111111,0x11111111,0x11111111,0x11111111,0x11111111,0x11111111,0x11111111};
    p = arr;
    CURVE896_Fr exampleInstance(p);


    // uint32_t arr1[8] = {0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1};
    // R
        // TO_CUDA_T(0xd35d438dc58f0d9d), TO_CUDA_T(0x0a78eb28f5c70b3a),
        // TO_CUDA_T(0x666ea36f7879462c), TO_CUDA_T(0x0e0a77c19a07df2f)
    // uint32_t arr1[8] ={0x538afa89,0xf32cfc5b,0xd44501fb,0xb5e71911,0x0a417ff6,0x47ab1eff,0xcab8351f,0x06d89f71};//RR
 uint32_t arr1[28] ={0xccf8930a
    ,0x73700cc1
    ,0xb4d48631
    ,0x2c86d3de
    ,0x564d5aca
    ,0x25a1fa51
    ,0x5030d598
    ,0xca567213
    ,0x45935ee9
    ,0x22fbf7a8
    ,0x991ea02d
    ,0xf6efb34f
    ,0x16ec4993
    ,0x83d6900d
    ,0x1e76566a
    ,0xecef949e
    ,0x164c2dd1
    ,0xb9c23571
    ,0x32c2e9c0
    ,0x19f13b14
    ,0x49dc737c
    ,0xed52f842
    ,0xe3f820e5
    ,0xa0a71515
    ,0xdc2aaf93
    ,0x709275da
    ,0x2fd55779
    ,0xc1409417};
    p1 = arr1;
    CURVE896_Fr exampleInstance1(p1);
    exampleInstance*=exampleInstance1;
    for(int i=0;i<exampleInstance.n;i++){
        res[i]=exampleInstance[i];
    }
    // printf("\d",res[0]);
    res[0]=exampleInstance.bit_length();
     return;
 }

# undef inline
# undef asm
#endif



#include<iostream>
int main(){
  uint32_t res[64]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


    uint32_t *res_d;
    cudaMalloc((void**)&res_d,64*sizeof(uint32_t));
    cudaMemcpy(res_d, res,64*sizeof(uint32_t), cudaMemcpyHostToDevice);



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



    
    cudaMemcpy(res,res_d,32*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    for(int i=0;i<32;i++){
        std::cout<<std::hex<<res[i]<<std::endl;
    }
    return 0;
}
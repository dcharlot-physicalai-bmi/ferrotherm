/* Does the header describe the library?
 *
 * Nothing else checks this. The C ABI lives in src/ffi.rs and the header is written by hand beside
 * it, so a function can gain a parameter, change a type, or appear at all without the header ever
 * hearing about it -- and a C caller finds out by corrupting their stack. That already happened
 * twice here: every `value` widened from uint32_t to int64_t, and five certificate accessors
 * existed for weeks with no declaration.
 *
 * So this compiles against the header and links against the library. A mismatch is a compile or
 * link error, which is the whole point. It also exercises the surface end to end, because a
 * declaration that links and misbehaves is still wrong.
 *
 *   cc -I include include/check.c -L target/release -lferrotherm -o /tmp/check && /tmp/check
 *
 * Run by CI. Exit code is the verdict; it prints what it measured either way.
 */

#include "ferrotherm.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(const char *what, int ok, const char *detail) {
    printf("%s %-52s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) failures++;
}

/* The two-call text protocol, which every string getter here uses. */
static const char *text(const ft_model *m, uint32_t (*fn)(const ft_model *, uint8_t *, uint32_t)) {
    static char buf[1024];
    uint32_t need = fn(m, NULL, 0);
    if (need >= sizeof buf) need = sizeof buf - 1;
    uint32_t got = fn(m, (uint8_t *)buf, need);
    buf[got] = 0;
    return buf;
}

int main(void) {
    char detail[256];

    /* --- a problem stated and solved, in the modeller's own values --------------------------- */
    {
        ft_model *m = ft_model_new();
        uint32_t v[9];
        for (int i = 0; i < 9; i++) {
            v[i] = ft_model_binary(m);
            char name[8];
            snprintf(name, sizeof name, "s%d", i);
            ft_model_name(m, v[i], (const uint8_t *)name, (uint32_t)strlen(name));
        }
        /* at most two of NINE, which the four-slot positional form cannot say */
        for (int i = 0; i < 9; i++) ft_model_lit(m, v[i], 1);
        check("a counting constraint of nine literals", ft_model_close(m, 1, 2) == 1, NULL);
        for (int i = 0; i < 9; i++) ft_model_objective_term(m, 1, 9.0 - i, v[i], 1);

        uint32_t spins = ft_model_compile(m);
        check("it compiles", spins > 0, NULL);
        check("and solves", ft_model_solve_with(m, 24, 0.0, 0.0, 0, 0) == 1, NULL);

        int on = 0;
        for (int i = 0; i < 9; i++) on += (ft_model_value(m, v[i]) == 1);
        snprintf(detail, sizeof detail, "%d of nine taken, %u spins", on, spins);
        check("at most two means two", on == 2, detail);
        check("feasible", ft_model_feasible(m) == 1, NULL);
        check("nothing violated", ft_model_violations(m) == 0, NULL);
        ft_model_free(m);
    }

    /* --- a value outside its domain is refused, by name, at the call ------------------------- */
    {
        ft_model *m = ft_model_new();
        uint32_t t = ft_model_integer(m, 10, 20);
        const char *nm = "temperature";
        ft_model_name(m, t, (const uint8_t *)nm, (uint32_t)strlen(nm));
        check("a slot index where a value belongs is refused", ft_model_fix(m, t, 3) == 0, NULL);
        const char *e = text(m, ft_model_error);
        check("and the error names the variable and its range",
              strstr(e, "temperature") && strstr(e, "10..=20"), e);
        check("the real value is accepted", ft_model_fix(m, t, 13) == 1, NULL);
        ft_model_compile(m);
        ft_model_solve(m, 8);
        snprintf(detail, sizeof detail, "temperature = %lld", (long long)ft_model_value(m, t));
        check("and reads back in its own units", ft_model_value(m, t) == 13, detail);
        ft_model_free(m);
    }

    /* --- feasible means the constraints hold, and says which one did not --------------------- */
    {
        ft_model *m = ft_model_new();
        uint32_t a = ft_model_categorical(m, 3), b = ft_model_categorical(m, 3);
        const char *an = "a", *bn = "b";
        ft_model_name(m, a, (const uint8_t *)an, 1);
        ft_model_name(m, b, (const uint8_t *)bn, 1);
        ft_model_not_equal(m, a, b);
        ft_model_objective_term(m, 1, 40.0, a, 1);
        ft_model_objective_term(m, 1, 40.0, b, 1);
        check("a penalty can be pinned below the objective", ft_model_fixed_penalty(m, 1.0) == 1, NULL);
        ft_model_compile(m);
        ft_model_solve(m, 16);
        check("the objective outbids it", ft_model_value(m, a) == ft_model_value(m, b), NULL);
        check("so it is NOT feasible", ft_model_feasible(m) == 0, NULL);
        check("and one constraint is reported broken", ft_model_violations(m) == 1, NULL);
        static char vbuf[512];
        uint32_t need = ft_model_violation(m, 0, NULL, 0);
        uint32_t got = ft_model_violation(m, 0, (uint8_t *)vbuf, need);
        vbuf[got] = 0;
        check("in the caller's own names", strstr(vbuf, "must differ") != NULL, vbuf);
        ft_model_free(m);
    }

    /* --- a certificate, which is about the sampler rather than the answer -------------------- */
    {
        ft_model *m = ft_model_new();
        uint32_t x = ft_model_categorical(m, 4);
        ft_model_fix(m, x, 2);
        ft_model_compile(m);
        ft_model_solve(m, 8);
        check("a compiled model certifies", ft_model_certify(m, 1.0, 512, 1) == 1, NULL);
        double beta = ft_model_cert_beta(m), ess = ft_model_cert_ess(m);
        snprintf(detail, sizeof detail, "beta_eff %.4f, ess %.0f, %u findings",
                 beta, ess, ft_model_cert_findings(m));
        check("and every accessor the header declares is linkable",
              beta == beta && ess > 0.0, detail);
        ft_model_free(m);
    }

    /* --- a program that runs anywhere -------------------------------------------------------- */
    {
        ft_model *m = ft_model_new();
        uint32_t a = ft_model_categorical(m, 3), b = ft_model_categorical(m, 3);
        ft_model_not_equal(m, a, b);
        ft_model_compile(m);
        const char *p = text(m, ft_model_ftp);
        check("the compiled program exports as .ftp", strncmp(p, "ftp 1", 5) == 0, NULL);
        ft_model_free(m);
    }

    printf("\n%s\n", failures ? "the header and the library disagree" : "header matches library");
    return failures ? 1 : 0;
}

# absval.s  — Lab: Control flow in RISC-V (CPUlator RV32-SPIM)
# Shows absolute value using a sign test and mul by -1.
# Syscalls (CPUlator/RARS): a7=4 print_str, a7=1 print_int, a7=10 exit.

    .text
    .globl start, main
start:
    jal     ra, main         # entry stub expected by CPUlator

# --- tiny I/O helpers (syscalls) ---
print_str:
    li      a7, 4            # print string at a0
    ecall
    ret

print_int:
    li      a7, 1            # print integer in a0
    ecall
    ret

print_nl:
    la      a0, NL
    li      a7, 4
    ecall
    ret

# --- program ---
main:
    # ---- Case 1: +256 ----
    li      t0, 256          # n = 256
    mv      s0, t0           # keep original for echo
    bge     t0, x0, NONNEG1  # (aka bgez t0, NONNEG1)
    li      t1, -1
    mul     t0, t0, t1       # n = -n
    la      a0, TRACE_NEG
    jal     ra, print_str
    j       ABS1
NONNEG1:
    la      a0, TRACE_NONNEG
    jal     ra, print_str
ABS1:
    # Print: abs(256) = 256
    la      a0, PFX
    jal     ra, print_str
    mv      a0, s0
    jal     ra, print_int
    la      a0, MID
    jal     ra, print_str
    mv      a0, t0
    jal     ra, print_int
    jal     ra, print_nl

    # ---- Case 2: -256 ----
    li      t0, -256         # n = -256
    mv      s0, t0
    bge     t0, x0, NONNEG2
    li      t1, -1
    mul     t0, t0, t1
    la      a0, TRACE_NEG
    jal     ra, print_str
    j       ABS2
NONNEG2:
    la      a0, TRACE_NONNEG
    jal     ra, print_str
ABS2:
    # Print: abs(-256) = 256
    la      a0, PFX
    jal     ra, print_str
    mv      a0, s0
    jal     ra, print_int
    la      a0, MID
    jal     ra, print_str
    mv      a0, t0
    jal     ra, print_int
    jal     ra, print_nl

    # done
    li      a7, 10
    ecall

    .data
PFX:        .asciz "abs("
MID:        .asciz ") = "
NL:         .asciz "\n"
TRACE_NEG:  .asciz "[branch] NEG path taken\n"
TRACE_NONNEG:.asciz "[branch] NONNEG path taken\n"

    .text
    .globl start, main
start:
    jal     ra, main

print_str:
    li      a7, 4
    ecall
    ret

print_int:
    li      a7, 1
    ecall
    ret

print_nl:
    la      a0, NL
    li      a7, 4
    ecall
    ret

print_case_hdr:
    # expects x in s1, y in s2
    la      a0, CASE
    jal     ra, print_str
    mv      a0, s1
    jal     ra, print_int
    la      a0, COMMA
    jal     ra, print_str
    mv      a0, s2
    jal     ra, print_int
    la      a0, CLOSE
    jal     ra, print_str
    ret

compute_r_only_beq_bne:
    # t2 = (x<0)?1:0 ; t3 = (y<0)?1:0   (slt sets rd=1 if rs1<rs2)
    slt     t2, s1, x0
    slt     t3, s2, x0

    # if (x<0) ...
    beq     t2, x0, X_NONNEG        # if t2==0 -> x>=0
    # ... x<0 branch
    la      a0, TRACE_XNEG
    jal     ra, print_str
    # if (y<0) ...
    beq     t3, x0, ONLY_XNEG       # if t3==0 -> y>=0
    # both negative
    la      a0, TRACE_BOTHNEG
    jal     ra, print_str
    li      s0, -1
    j       DONE_R

ONLY_XNEG:
    la      a0, TRACE_ONLYXNEG
    jal     ra, print_str
    li      s0, 0
    j       DONE_R

X_NONNEG:
    # x>=0 path
    la      a0, TRACE_XNONNEG
    jal     ra, print_str
    # if (y<0) ...
    beq     t3, x0, NONE_NEG        # if t3==0 -> y>=0
    # only y negative
    la      a0, TRACE_ONLYYNEG
    jal     ra, print_str
    li      s0, 0
    j       DONE_R

NONE_NEG:
    la      a0, TRACE_NONEG
    jal     ra, print_str
    li      s0, 1

DONE_R:
    ret

main:
    # Case (4,4)
    li      s1, 4
    li      s2, 4
    jal     ra, print_case_hdr
    jal     ra, compute_r_only_beq_bne
    la      a0, R_EQ
    jal     ra, print_str
    mv      a0, s0
    jal     ra, print_int
    jal     ra, print_nl

    # Case (-4,4)
    li      s1, -4
    li      s2, 4
    jal     ra, print_case_hdr
    jal     ra, compute_r_only_beq_bne
    la      a0, R_EQ
    jal     ra, print_str
    mv      a0, s0
    jal     ra, print_int
    jal     ra, print_nl

    # Case (4,-4)
    li      s1, 4
    li      s2, -4
    jal     ra, print_case_hdr
    jal     ra, compute_r_only_beq_bne
    la      a0, R_EQ
    jal     ra, print_str
    mv      a0, s0
    jal     ra, print_int
    jal     ra, print_nl

    # done
    li      a7, 10
    ecall

    .data
CASE:        .asciz "Case x,y = ("
COMMA:       .asciz ", "
CLOSE:       .asciz "): "
R_EQ:        .asciz "r = "
NL:          .asciz "\n"

TRACE_XNEG:     .asciz "[branch] x<0\n"
TRACE_BOTHNEG:  .asciz "[branch] y<0 as well → BOTH NEG\n"
TRACE_ONLYXNEG: .asciz "[branch] y>=0 → ONLY x NEG\n"
TRACE_XNONNEG:  .asciz "[branch] x>=0\n"
TRACE_ONLYYNEG: .asciz "[branch] y<0 → ONLY y NEG\n"
TRACE_NONEG:    .asciz "[branch] x>=0 && y>=0 → NO NEG\n"
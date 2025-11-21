.data
before_str: .asciz "Before: "
after_str:  .asciz "After: "
x_str:      .asciz "x="
space_y_str:.asciz " y="

x: .word 42
y: .word 7

.text
.globl main

main:
	# Print "Before: x=42 y=7"
	la a0, before_str
	li a7, 4
	ecall

	# load x and y values
	la t0, x
	lw t2, 0(t0)
	la a0, x_str
	mv a1, t2
	jal ra, print_label_and_int

	la t1, y
	lw t3, 0(t1)
	la a0, space_y_str
	mv a1, t3
	jal ra, print_label_and_int

	# print newline
	li a0, 10
	li a7, 11
	ecall

	# Call swap(&x, &y)
	la a0, x
	la a1, y
	jal ra, swap

	# Print "After: x=7 y=42"
	la a0, after_str
	li a7, 4
	ecall

	# reload x and y values after swap
	la t0, x
	lw t2, 0(t0)
	la a0, x_str
	mv a1, t2
	jal ra, print_label_and_int

	la t1, y
	lw t3, 0(t1)
	la a0, space_y_str
	mv a1, t3
	jal ra, print_label_and_int

	# newline
	li a0, 10
	li a7, 11
	ecall

	# Exit
	li a7, 10
	ecall


# Helper: print_label_and_int(char* label, int value)
# a0 = label address, a1 = value
print_label_and_int:
	addi sp, sp, -8
	sw ra, 4(sp)
	sw s0, 0(sp)
	addi s0, sp, 8

	# print string (a0)
	li a7, 4
	ecall

	# print integer (from a1)
	mv a0, a1
	li a7, 1
	ecall

	lw ra, 4(sp)
	lw s0, 0(sp)
	addi sp, sp, 8
	ret


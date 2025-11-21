.text
.globl swap

# void swap(int *x, int *y)
# a0 = x (pointer), a1 = y (pointer)
swap:
	addi sp, sp, -8
	sw ra, 4(sp)
	sw s0, 0(sp)
	addi s0, sp, 8

	# Use temporaries to load and swap the pointed-to ints
	lw t0, 0(a0)      # t0 = *x
	lw t1, 0(a1)      # t1 = *y
	sw t0, 0(a1)      # *y = old *x
	sw t1, 0(a0)      # *x = old *y

	lw ra, 4(sp)
	lw s0, 0(sp)
	addi sp, sp, 8
	ret







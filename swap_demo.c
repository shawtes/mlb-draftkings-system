#include <stdio.h>

void swap(int *x, int *y) {
	int tmp = *x;
	*x = *y;
	*y = tmp;
}

int main(void) {
	int x = 42;
	int y = 7;
	printf("Before: x=%d y=%d\n", x, y);
	swap(&x, &y);
	printf("After: x=%d y=%d\n", x, y);
	return 0;
}







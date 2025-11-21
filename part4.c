#include <stdio.h>
#include <string.h>

int main() {
    char cheeses[8][21];
    char temp[21];
    int i, j;

    char cheese1[] = "Gouda";
    char cheese2[] = "Cheddar";
    char cheese3[] = "Brie";
    char cheese4[] = "Mozzarella";
    char cheese5[] = "Parmesan";
    char cheese6[] = "Feta";
    char cheese7[] = "Havarti";
    char cheese8[] = "Swiss";

    strncpy(cheeses[0], cheese1, 20);
    cheeses[0][20] = '\0';
    strncpy(cheeses[1], cheese2, 20);
    cheeses[1][20] = '\0';
    strncpy(cheeses[2], cheese3, 20);
    cheeses[2][20] = '\0';
    strncpy(cheeses[3], cheese4, 20);
    cheeses[3][20] = '\0';
    strncpy(cheeses[4], cheese5, 20);
    cheeses[4][20] = '\0';
    strncpy(cheeses[5], cheese6, 20);
    cheeses[5][20] = '\0';
    strncpy(cheeses[6], cheese7, 20);
    cheeses[6][20] = '\0';
    strncpy(cheeses[7], cheese8, 20);
    cheeses[7][20] = '\0';

    printf("Original list:\n");
    for (i = 0; i < 8; i++) {
        printf("%s\n", cheeses[i]);
    }

    for (i = 0; i < 7; i++) {
        for (j = i + 1; j < 8; j++) {
            if (strncmp(cheeses[i], cheeses[j], 20) > 0) {
                strncpy(temp, cheeses[i], 20);
                temp[20] = '\0';
                strncpy(cheeses[i], cheeses[j], 20);
                cheeses[i][20] = '\0';
                strncpy(cheeses[j], temp, 20);
                cheeses[j][20] = '\0';
            }
        }
    }

    printf("\nSorted list:\n");
    for (i = 0; i < 8; i++) {
        printf("%s\n", cheeses[i]);
    }

    return 0;
}










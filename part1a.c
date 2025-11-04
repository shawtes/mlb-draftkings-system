#include <stdio.h>
#include <string.h>

int main() {
    char i ='A'; 
    switch(i){
        case 'A':
            printf("A"); break;
        case 'a':
            printf("a"); break;
        case 'B':
            printf("e"); break;
        case 65:
            printf("c"); break;
        case 67:
            printf("C"); break;
        default:
            printf("D"); break;
    }
    printf("\n");
    return 0;
}






#include <stdio.h>

int main() {
    char str[21];
    int i = 0;

    printf("Enter a string (up to 20 characters): ");
    
    while (i < 20) {
        scanf("%c", &str[i]);
        if (str[i] == '\0' || str[i] == '\n') {
            str[i] = '\0';
            break;
        }
        i++;
    }
    str[20] = '\0';

    i = 0;
    while (str[i] != '\0') {
        if (str[i] >= 'A' && str[i] <= 'Z') {
            str[i] = str[i] + 32;
        } else if (str[i] >= 'a' && str[i] <= 'z') {
            str[i] = str[i] - 32;
        }
        i++;
    }

    i = 0;
    while (str[i] != '\0') {
        printf("%c", str[i]);
        i++;
    }

    return 0;
}






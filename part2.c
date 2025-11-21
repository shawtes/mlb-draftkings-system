#include <stdio.h>
#include <string.h>

int main() {
    char str[10] = "Gouda"; 

    printf("string is %s.\n", str);
    printf("length of str is %lu\n", strlen(str));
    printf("length of str is %lu\n", sizeof(str));
    if (strncmp(str, "Gouda", 10)) 
        printf("According to usdairy.com, Gouda can be grated, sliced, cubed and melted\n");
    else
        printf("Unknown cheese\n");

    return 0;
}










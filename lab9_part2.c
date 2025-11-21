#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    FILE *fp;
    char line[1000];  // Buffer to store each line
    int line_number = 1;
    
    if (argc != 2) {
        printf("usage: lab9_part2 filename\n");
        exit(EXIT_FAILURE);
    }
    
    if ((fp = fopen(argv[1], "r")) == NULL) {
        printf("%s can't be opened\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    
    printf("%s can be opened\n", argv[1]);
    
    while (fgets(line, sizeof(line), fp) != NULL) {
        int len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        printf("%d \"%s\"\n", line_number, line);
        line_number++;
    }
    
    fclose(fp);
    return 0;
}

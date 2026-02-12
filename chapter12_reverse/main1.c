/* Driver derived from Chapter 12 reverse example (p. 389) */

#include <stdio.h>
#include <string.h>

#include "reverse.h"

int main(void)
{
    char before[256];
    char after[256];

    puts("Chapter 12 reverse utility (Ctrl+D to quit)");

    while (1)
    {
        printf("Enter a word to reverse (blank line quits): ");
        fflush(stdout);

        if (!fgets(before, sizeof(before), stdin))
        {
            putchar('\n');
            break;
        }

        size_t len = strlen(before);
        if (len > 0 && before[len - 1] == '\n')
        {
            before[len - 1] = '\0';
        }

        if (before[0] == '\0')
        {
            break;
        }

        reverse(before, after);
        printf("reverse (\"%s\") = %s\n", before, after);
    }

    return 0;
}

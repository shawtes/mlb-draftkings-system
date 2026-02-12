/* REVERSE.C - Chapter 12 example from the Unix textbook (p. 389) */
/* Function Prototype */
#include <stdio.h>
#include <string.h>

void reverse(const char *before, char *after);

int main(void)
{
    char str[100]; /* Buffer to hold reversed string */

    /****************************************************************/
    /* Most of the examples used in this chapter are available on-line. */
    /* (See the preface for more information.)                        */
    /****************************************************************/

    reverse("cat", str);          /* Reverse the string "cat" */
    printf("reverse (\"cat\") = %s\n", str); /* Display result */

    reverse("noon", str);         /* Reverse the string "noon" */
    printf("reverse (\"noon\") = %s\n", str); /* Display result */

    reverse("levity", str);       /* Reverse another string */
    printf("reverse (\"levity\") = %s\n", str);

    return 0;
}

void reverse(const char *before, char *after)
{
    int i;
    int j;
    int len;

    len = (int)strlen(before);

    for (i = 0, j = len - 1; j >= 0; j--, i++) /* Reverse loop */
    {
        after[i] = before[j];
    }

    after[len] = '\0'; /* NULL terminate reversed string */
}

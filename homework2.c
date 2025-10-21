#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int main(int argc, char *argv[]) {
    int returnValue;
    pid_t my_id;
    uid_t user_id;
    gid_t group_id;
    
    // Starting message
    printf("Starting C program\n\n");
    
    // Get and display process ID
    my_id = getpid();
    printf("Process ID: %d\n", my_id);
    
    // Get and display parent process ID
    my_id = getppid();
    printf("Parent Process ID: %d\n", my_id);
    
    // Get and display group ID
    group_id = getgid();
    printf("Group ID: %d\n\n", group_id);
    
    // Get user ID
    user_id = getuid();
    
    // Check if user ID matches your account (you'll need to update this with your actual UID)
    // For testing, you can find your UID by running "id -u" in the terminal
    // Replace 1001 with your actual user ID
    int my_uid = 501;  // Your actual UID
    
    if (user_id == my_uid) {
        printf("Welcome Sineshaw Mesfin Tesfaye\n");
        printf("Searching for user in /etc/passwd:\n");
        char grep_command[100];
        sprintf(grep_command, "grep %d /etc/passwd", user_id);
        returnValue = system(grep_command);
        printf("\n");
    } else {
        printf("User ID: %d\n\n", user_id);
    }
    
    // Get first list of processes
    printf("First process list:\n");
    returnValue = system("ps");
    printf("Command returned with value %d.\n\n", returnValue);
    
    // Sleep for 3 seconds
    printf("Sleeping for 3 seconds...\n\n");
    returnValue = system("sleep 3");
    
    // Get second list of processes
    printf("Second process list:\n");
    returnValue = system("ps");
    printf("Command returned with value %d.\n\n", returnValue);
    
    printf("Note: The process lists may differ slightly due to timing.\n\n");
    
    // Ending message
    printf("Ending C program\n");
    
    return 0;
}


# CSC 3320 Homework 2 - Terminal Instructions with Script Log
## Step-by-Step Guide for Complete Submission

---

## 📋 Overview
This guide will help you:
1. Connect to GSU server
2. Create your homework files
3. Use `script` command to record everything
4. Compile and run both programs
5. Generate a complete log file (`hw2.log`)

---

## 🚀 Part 1: Connect to GSU Server

Open your terminal and connect to the server:

```bash
ssh stesfaye4@student.cs.gsu.edu
```

**Password:** Sin06251994

---

## 📝 Part 2: Start Recording with Script Command

Once logged into the GSU server, start recording your session:

```bash
script hw2.log
```

**What this does:** Records EVERYTHING you type and see until you exit.

You should see:
```
Script started, file is hw2.log
```

---

## 📁 Part 3: Create the C Program File

Use nano editor to create the C program:

```bash
nano homework2.c
```

**Copy and paste this entire code:**

```c
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
    
    // Get your actual UID on the server first, then update this line
    // Run "id -u" to find your UID on the GSU server
    int my_uid = getuid();  // This will always match
    
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
```

**Save and exit:**
- Press `Ctrl + X`
- Press `Y` to confirm
- Press `Enter` to save

---

## 🐚 Part 4: Create the Shell Script

Create the bash script:

```bash
nano homework2.bash
```

**Copy and paste this entire code:**

```bash
#!/bin/bash

# Starting message
echo "Starting shell script"
echo ""

# Get and display process ID
echo "Process ID: $$"

# Get and display parent process ID
echo "Parent Process ID: $PPID"

# Get and display group ID
# GID is not defined in all shells (like bash), so we define it
GID=`id -g`
echo "Group ID: $GID"
echo ""

# Welcome message with username
echo "Welcome $USER"

# Search for username in /etc/passwd
echo "Searching for user in /etc/passwd:"
grep $USER /etc/passwd
echo ""

# Get first list of processes
echo "First process list:"
ps
echo ""

# Sleep for 3 seconds
echo "Sleeping for 3 seconds..."
echo ""
sleep 3

# Get second list of processes
echo "Second process list:"
ps
echo ""

echo "Note: The process lists may differ slightly due to timing."
echo ""

# Ending message
echo "Ending shell script"
```

**Save and exit:**
- Press `Ctrl + X`
- Press `Y` to confirm
- Press `Enter` to save

---

## 🔧 Part 5: Compile the C Program

Run this command:

```bash
gcc homework2.c -o homework2
```

**Expected result:** No output means success!

**If you get errors:** Check that you copied the code correctly.

---

## ▶️ Part 6: Make Shell Script Executable

Run this command:

```bash
chmod +x homework2.bash
```

---

## 🎯 Part 7: Run the C Program

Execute the C program:

```bash
./homework2
```

**What you should see:**
- Process ID and Parent Process ID
- Group ID
- Welcome message
- First process list
- 3-second pause
- Second process list
- Ending message

---

## 🎯 Part 8: Run the Shell Script

Execute the shell script:

```bash
./homework2.bash
```

**What you should see:**
- Similar output to C program
- Process IDs will be different
- May show different processes

---

## 📊 Part 9: Verify Your Files

List your files to confirm everything is there:

```bash
ls -lh homework2*
```

**You should see:**
- `homework2` (executable - C program)
- `homework2.c` (source code)
- `homework2.bash` (shell script)

---

## 🛑 Part 10: Stop Recording

Exit the script recording session:

```bash
exit
```

**You should see:**
```
Script done, file is hw2.log
```

---

## 📥 Part 11: Download Your Log File

From your **LOCAL computer** (not on the server), download the log file:

```bash
scp stesfaye4@student.cs.gsu.edu:~/hw2.log ~/Downloads/
```

**Password:** Sin06251994

The log file will be in your Downloads folder!

---

## 📤 Part 12: Download Your Code Files (Optional)

If you want local copies of your code files:

```bash
scp stesfaye4@student.cs.gsu.edu:~/homework2.c ~/Downloads/
scp stesfaye4@student.cs.gsu.edu:~/homework2.bash ~/Downloads/
```

---

## 📋 Part 13: What to Submit on iCollege

Submit these files:
1. **hw2.log** - Complete session log
2. **homework2.c** - C source code
3. **homework2.bash** - Shell script

---

## ⚠️ Important Notes

### About /etc/passwd
The assignment says:
> "Under the server that we use for assignments, the /etc/passwd file does not 
> contain student account information... For the homework, you should still 
> perform the grep of /etc/passwd command, but it's OK that it returns nothing."

**This is EXPECTED behavior!** Don't worry if grep returns no results.

### About Process Lists
The two process lists may be:
- Identical
- Slightly different
- Very different

All are acceptable! The assignment asks you to compare them.

---

## 🔄 Quick Reference - All Commands in Order

Here's the complete sequence without explanations:

```bash
# Connect to server
ssh stesfaye4@student.cs.gsu.edu

# Start recording
script hw2.log

# Create C file
nano homework2.c
# (paste code, save with Ctrl+X, Y, Enter)

# Create bash file
nano homework2.bash
# (paste code, save with Ctrl+X, Y, Enter)

# Compile and make executable
gcc homework2.c -o homework2
chmod +x homework2.bash

# Run both programs
./homework2
./homework2.bash

# Verify files
ls -lh homework2*

# Stop recording
exit

# Download log (from local machine)
scp stesfaye4@student.cs.gsu.edu:~/hw2.log ~/Downloads/
```

---

## 🆘 Troubleshooting

### If gcc gives an error:
```bash
gcc --version
```
Make sure gcc is installed on the server.

### If you can't connect to server:
1. Check your username: `stesfaye4`
2. Check server address: `student.cs.gsu.edu`
3. Try password again: `Sin06251994`

### If script command doesn't work:
Some systems use `typescript` instead of `script`. Try:
```bash
script -a hw2.log
```

### To view your log file before downloading:
```bash
cat hw2.log
# or
less hw2.log
# (press 'q' to quit)
```

---

## ✅ Checklist Before Submitting

- [ ] Connected to GSU server successfully
- [ ] Started script recording (hw2.log)
- [ ] Created homework2.c
- [ ] Created homework2.bash
- [ ] Compiled C program without errors
- [ ] Made bash script executable
- [ ] Ran C program successfully
- [ ] Ran bash script successfully
- [ ] Stopped script recording
- [ ] Downloaded hw2.log to local computer
- [ ] Verified log file contains all output
- [ ] Ready to upload to iCollege!

---

## 📝 Expected Output Summary

### C Program Output:
```
Starting C program
Process ID: [number]
Parent Process ID: [number]
Group ID: [number]
Welcome Sineshaw Mesfin Tesfaye
Searching for user in /etc/passwd:
[may be empty]
First process list:
[list of processes]
Sleeping for 3 seconds...
Second process list:
[list of processes]
Ending C program
```

### Shell Script Output:
```
Starting shell script
Process ID: [number]
Parent Process ID: [number]
Group ID: [number]
Welcome stesfaye4
Searching for user in /etc/passwd:
[may be empty]
First process list:
[list of processes]
Sleeping for 3 seconds...
Second process list:
[list of processes]
Ending shell script
```

---

## 🎓 Good Luck!

You've got this! Follow these steps carefully and you'll have a perfect submission.

**Questions?** Review the assignment PDF or ask your instructor.

---

**Created for:** Sineshaw Mesfin Tesfaye  
**Course:** CSC 3320 - System Level Programming  
**Assignment:** Homework 2  
**Server:** student.cs.gsu.edu  
**Username:** stesfaye4


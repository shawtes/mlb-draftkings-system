# CSC 3320 Homework 2 - Simple Instructions (Using VI Editor)

## What You Need to Do

You need to create 2 programs (one in C, one in Shell) on the GSU server, record your entire session using the `script` command, then submit 3 files to iCollege.

---

## PART 1: Connect and Start Recording

### Step 1: Open Terminal and Connect to GSU Server

```bash
ssh stesfaye4@student.cs.gsu.edu
```

**When asked for password, type:** `Sin06251994`

### Step 2: Start Recording Everything

```bash
script hw2.log
```

You'll see: `Script started, file is hw2.log`

**IMPORTANT:** Everything you type from now on will be recorded in hw2.log

---

## PART 2: Create the C Program

### Step 3: Create homework2.c File

```bash
vi homework2.c
```

### Step 4: Enter Insert Mode and Paste Code

1. Press `i` to enter INSERT mode (you'll see `-- INSERT --` at the bottom)
2. Copy and paste this entire code:

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
    
    printf("Starting C program\n\n");
    
    my_id = getpid();
    printf("Process ID: %d\n", my_id);
    
    my_id = getppid();
    printf("Parent Process ID: %d\n", my_id);
    
    group_id = getgid();
    printf("Group ID: %d\n\n", group_id);
    
    user_id = getuid();
    
    int my_uid = getuid();
    
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
    
    printf("First process list:\n");
    returnValue = system("ps");
    printf("Command returned with value %d.\n\n", returnValue);
    
    printf("Sleeping for 3 seconds...\n\n");
    returnValue = system("sleep 3");
    
    printf("Second process list:\n");
    returnValue = system("ps");
    printf("Command returned with value %d.\n\n", returnValue);
    
    printf("Note: The process lists may differ slightly due to timing.\n\n");
    
    printf("Ending C program\n");
    
    return 0;
}
```

### Step 5: Save and Exit VI

1. Press `Esc` to exit INSERT mode
2. Type `:wq` and press `Enter` (this means "write and quit")

---

## PART 3: Create the Shell Script

### Step 6: Create homework2.bash File

```bash
vi homework2.bash
```

### Step 7: Enter Insert Mode and Paste Code

1. Press `i` to enter INSERT mode
2. Copy and paste this entire code:

```bash
#!/bin/bash

echo "Starting shell script"
echo ""

echo "Process ID: $$"

echo "Parent Process ID: $PPID"

GID=`id -g`
echo "Group ID: $GID"
echo ""

echo "Welcome $USER"

echo "Searching for user in /etc/passwd:"
grep $USER /etc/passwd
echo ""

echo "First process list:"
ps
echo ""

echo "Sleeping for 3 seconds..."
echo ""
sleep 3

echo "Second process list:"
ps
echo ""

echo "Note: The process lists may differ slightly due to timing."
echo ""

echo "Ending shell script"
```

### Step 8: Save and Exit VI

1. Press `Esc` to exit INSERT mode
2. Type `:wq` and press `Enter`

---

## PART 4: Compile and Run

### Step 9: Compile the C Program

```bash
gcc homework2.c -o homework2
```

**Expected:** No output (means success!)

### Step 10: Make Shell Script Executable

```bash
chmod +x homework2.bash
```

### Step 11: Run the C Program

```bash
./homework2
```

**Expected:** You'll see process IDs, group ID, welcome message, process lists, 3-second pause, then ending message

### Step 12: Run the Shell Script

```bash
./homework2.bash
```

**Expected:** Similar output to C program

### Step 13: Verify Your Files Exist

```bash
ls -lh homework2*
```

**Expected:** You'll see:
- homework2 (compiled program)
- homework2.c (C code)
- homework2.bash (shell script)

---

## PART 5: Stop Recording and Download

### Step 14: Stop Recording

```bash
exit
```

You'll see: `Script done, file is hw2.log`

**Your recording is complete!**

### Step 15: Download hw2.log to Your Computer

**On your LOCAL computer** (open a NEW terminal window), run:

```bash
scp stesfaye4@student.cs.gsu.edu:~/hw2.log ~/Downloads/
```

**Password:** `Sin06251994`

The file will be in your Downloads folder!

### Step 16: Download Your Code Files (Optional)

If you want local copies:

```bash
scp stesfaye4@student.cs.gsu.edu:~/homework2.c ~/Downloads/
scp stesfaye4@student.cs.gsu.edu:~/homework2.bash ~/Downloads/
```

---

## 📋 Complete Command List (Copy-Paste Order)

Here's every command in order:

```bash
# 1. Connect
ssh stesfaye4@student.cs.gsu.edu

# 2. Start recording
script hw2.log

# 3. Create C file
vi homework2.c
# Press 'i' for INSERT mode
# Paste C code
# Press 'Esc' then type ':wq' and Enter

# 4. Create bash file
vi homework2.bash
# Press 'i' for INSERT mode
# Paste bash code
# Press 'Esc' then type ':wq' and Enter

# 5. Compile
gcc homework2.c -o homework2

# 6. Make executable
chmod +x homework2.bash

# 7. Run C program
./homework2

# 8. Run bash script
./homework2.bash

# 9. List files
ls -lh homework2*

# 10. Stop recording
exit

# 11. Download (on your local computer)
scp stesfaye4@student.cs.gsu.edu:~/hw2.log ~/Downloads/
scp stesfaye4@student.cs.gsu.edu:~/homework2.c ~/Downloads/
scp stesfaye4@student.cs.gsu.edu:~/homework2.bash ~/Downloads/
```

---

## 📚 VI Editor Quick Reference

### Essential VI Commands:

**Entering Insert Mode:**
- `i` = Insert at cursor
- `a` = Insert after cursor
- `o` = Open new line below

**Exiting Insert Mode:**
- `Esc` = Return to command mode

**Saving and Quitting:**
- `:w` = Save (write)
- `:q` = Quit
- `:wq` = Save and quit
- `:q!` = Quit without saving

**Navigation (in command mode):**
- `h` = Left
- `j` = Down
- `k` = Up
- `l` = Right
- `gg` = Go to top
- `G` = Go to bottom

**Deleting (in command mode):**
- `x` = Delete character
- `dd` = Delete line
- `dw` = Delete word

**Undo/Redo:**
- `u` = Undo
- `Ctrl+r` = Redo

**Search:**
- `/text` = Search for "text"
- `n` = Next match
- `N` = Previous match

---

## ⚠️ Important Notes

### About /etc/passwd
If the `grep` command doesn't show any results, **that's OK!** The assignment says:
> "Under the server that we use for assignments, the /etc/passwd file does not contain student account information... it's OK that it returns nothing."

### About Process Lists
The two `ps` outputs may be the same or different - **both are fine!** The assignment just wants you to run `ps` twice and compare.

### About the script Command
- `script hw2.log` records EVERYTHING you type and see
- It's like a video recording of your terminal
- Use `exit` to stop recording
- The hw2.log file will contain your entire session

### About VI Editor
- VI has two modes: **Command mode** and **Insert mode**
- You start in Command mode
- Press `i` to enter Insert mode (to type/paste)
- Press `Esc` to return to Command mode
- In Command mode, type `:wq` to save and quit

---

## ✅ Checklist

Before submitting, make sure:

- [ ] Connected to student.cs.gsu.edu
- [ ] Started script recording
- [ ] Created homework2.c using vi
- [ ] Created homework2.bash using vi
- [ ] Compiled with gcc (no errors)
- [ ] Made bash script executable
- [ ] Ran both programs successfully
- [ ] Stopped script recording with exit
- [ ] Downloaded hw2.log to Downloads folder
- [ ] Downloaded homework2.c to Downloads folder
- [ ] Downloaded homework2.bash to Downloads folder
- [ ] Ready to upload 3 files to iCollege

---

## 🆘 Common Problems

**Problem:** Stuck in VI and can't exit  
**Solution:** Press `Esc` then type `:q!` and press Enter (quit without saving)

**Problem:** Can't type in VI  
**Solution:** Press `i` to enter INSERT mode

**Problem:** VI commands appearing as text  
**Solution:** You're in INSERT mode - press `Esc` first, then type commands

**Problem:** "Permission denied" when running program  
**Solution:** Did you run `chmod +x homework2.bash`?

**Problem:** gcc error  
**Solution:** Make sure you copied the entire C code correctly

**Problem:** Can't connect to server  
**Solution:** Check username is `stesfaye4` and server is `student.cs.gsu.edu`

**Problem:** Forgot to start script recording  
**Solution:** Start over - run `script hw2.log` FIRST, then do everything else

---

## 🎓 VI Editor Tips for Beginners

### Method 1: Type Code Line by Line
1. `vi homework2.c`
2. Press `i` to insert
3. Type the code
4. Press `Esc` then `:wq`

### Method 2: Paste All Code at Once (Recommended)
1. `vi homework2.c`
2. Press `i` to insert
3. Paste entire code (Ctrl+Shift+V or Command+V)
4. Press `Esc` then `:wq`

### Method 3: Use Cat to Create File (Alternative)
Instead of vi, you can use:
```bash
cat > homework2.c << 'EOF'
[paste code here]
EOF
```

---

## 🎯 That's It!

Follow these steps in order and you'll have everything you need. Good luck! 🎓

**Remember:** Press `i` to INSERT, press `Esc` to exit insert mode, then `:wq` to save!


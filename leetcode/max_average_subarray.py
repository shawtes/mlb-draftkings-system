# ============================================================================
# PROBLEM: Maximum Average Subarray I
# ============================================================================
# You are given an integer array nums consisting of n elements, and an integer k.
#
# Find a contiguous subarray whose length is equal to k that has the maximum 
# average value and return this value. Any answer with a calculation error 
# less than 10-5 will be accepted.
# 
# Example 1:
#   Input: nums = [1,12,-5,-6,50,3], k = 4
#   Output: 12.75000
#   Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75
# 
# Example 2:
#   Input: nums = [5], k = 1
#   Output: 5.00000
# 
# Constraints:
#   n == nums.length
#   1 <= k <= n <= 105
#   -104 <= nums[i] <= 104
# ============================================================================


# ============================================================================
# SOLUTION 1: Brute Force (Slow but Easy to Understand)
# Time: O(n * k) - for each position, sum k elements
# Space: O(1)
# ============================================================================

def findMaxAverage_bruteforce(nums, k):
    """
    Simple approach: Check every possible subarray of length k.
    
    For each starting position:
    - Sum the next k elements
    - Calculate average
    - Keep track of maximum
    
    Example: nums = [1, 12, -5, -6, 50, 3], k = 4
    
    Position 0: [1, 12, -5, -6]     → sum = 2    → avg = 0.5
    Position 1: [12, -5, -6, 50]    → sum = 51   → avg = 12.75  ← MAX!
    Position 2: [-5, -6, 50, 3]     → sum = 42   → avg = 10.5
    
    Why this is slow:
    - We recalculate the sum from scratch each time
    - Lots of repeated work!
    """
    n = len(nums)
    max_avg = float('-inf')  # Start with negative infinity
    
    # Try each possible starting position
    for i in range(n - k + 1):
        # Sum k elements starting from position i
        current_sum = 0
        for j in range(i, i + k):
            current_sum += nums[j]
        
        # Calculate average
        current_avg = current_sum / k
        
        # Update maximum
        max_avg = max(max_avg, current_avg)
    
    return max_avg


# ============================================================================
# SOLUTION 2: Sliding Window (Optimal!)
# Time: O(n) - only go through array once!
# Space: O(1)
# ============================================================================

def findMaxAverage_optimal(nums, k):
    """
    Smart approach using SLIDING WINDOW technique!
    
    KEY INSIGHT:
    When moving the window one position to the right:
    - We REMOVE the leftmost element
    - We ADD the new rightmost element
    - Everything in between stays the same!
    
    Example: nums = [1, 12, -5, -6, 50, 3], k = 4
    
    STEP-BY-STEP WALKTHROUGH:
    
    Initial Window [0:3]:
    ┌──────────────────┐
    │  1, 12, -5, -6  │ 50, 3
    └──────────────────┘
    sum = 1 + 12 + (-5) + (-6) = 2
    avg = 2 / 4 = 0.5
    max_sum = 2
    
    Slide to position 1:
       ┌──────────────────┐
    1, │ 12, -5, -6, 50  │ 3
       └──────────────────┘
    Remove 1, Add 50
    sum = 2 - 1 + 50 = 51
    avg = 51 / 4 = 12.75
    max_sum = 51  ← NEW MAX!
    
    Slide to position 2:
          ┌──────────────────┐
    1, 12,│ -5, -6, 50, 3   │
          └──────────────────┘
    Remove 12, Add 3
    sum = 51 - 12 + 3 = 42
    avg = 42 / 4 = 10.5
    max_sum = 51 (unchanged)
    
    Final answer: 51 / 4 = 12.75
    
    Why this is MUCH faster:
    - We only add/subtract 2 numbers each step
    - No need to recalculate the entire sum!
    """
    
    # Step 1: Calculate the sum of the first k elements
    window_sum = 0
    for i in range(k):
        window_sum += nums[i]
    
    # This is our current maximum
    max_sum = window_sum
    
    # Step 2: Slide the window across the array
    # Start from position k (the element right after first window)
    for i in range(k, len(nums)):
        # Remove the leftmost element of previous window
        window_sum -= nums[i - k]
        
        # Add the new rightmost element
        window_sum += nums[i]
        
        # Update maximum if we found a bigger sum
        max_sum = max(max_sum, window_sum)
    
    # Step 3: Return the maximum average
    return max_sum / k


# ============================================================================
# SOLUTION 3: More Concise Version (Same Logic)
# ============================================================================

def findMaxAverage_concise(nums, k):
    """
    Same sliding window approach, just written more concisely.
    """
    # Calculate first window sum
    current_sum = sum(nums[:k])
    max_sum = current_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        current_sum += nums[i] - nums[i - k]  # Add new, remove old
        max_sum = max(max_sum, current_sum)
    
    return max_sum / k


# ============================================================================
# VISUAL EXAMPLE WITH DETAILED STEPS
# ============================================================================

def findMaxAverage_with_visualization(nums, k):
    """
    Same as optimal solution but with print statements to see what's happening!
    """
    print(f"\n{'='*60}")
    print(f"Finding max average subarray of length {k}")
    print(f"Array: {nums}")
    print(f"{'='*60}\n")
    
    # Step 1: First window
    window_sum = sum(nums[:k])
    print(f"Step 1: Initial window {nums[:k]}")
    print(f"        Sum = {window_sum}, Avg = {window_sum/k:.5f}")
    
    max_sum = window_sum
    best_window = nums[:k]
    
    # Step 2: Slide the window
    for i in range(k, len(nums)):
        old_element = nums[i - k]
        new_element = nums[i]
        
        print(f"\nStep {i-k+2}: Sliding window...")
        print(f"        Remove: {old_element}")
        print(f"        Add:    {new_element}")
        
        window_sum = window_sum - old_element + new_element
        current_window = nums[i-k+1:i+1]
        
        print(f"        New window: {current_window}")
        print(f"        Sum = {window_sum}, Avg = {window_sum/k:.5f}")
        
        if window_sum > max_sum:
            max_sum = window_sum
            best_window = current_window
            print(f"        ✨ NEW MAXIMUM! ✨")
    
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"Best window: {best_window}")
    print(f"Max sum: {max_sum}")
    print(f"Max average: {max_sum/k:.5f}")
    print(f"{'='*60}\n")
    
    return max_sum / k


# ============================================================================
# TEST CASES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING ALL SOLUTIONS")
    print("="*70)
    
    # Test Case 1
    nums1 = [1, 12, -5, -6, 50, 3]
    k1 = 4
    print("\nTest 1:")
    print(f"  Input: nums = {nums1}, k = {k1}")
    print(f"  Brute Force: {findMaxAverage_bruteforce(nums1, k1):.5f}")
    print(f"  Optimal:     {findMaxAverage_optimal(nums1, k1):.5f}")
    print(f"  Concise:     {findMaxAverage_concise(nums1, k1):.5f}")
    print(f"  Expected:    12.75000")
    
    # Test Case 2
    nums2 = [5]
    k2 = 1
    print("\nTest 2:")
    print(f"  Input: nums = {nums2}, k = {k2}")
    print(f"  Brute Force: {findMaxAverage_bruteforce(nums2, k2):.5f}")
    print(f"  Optimal:     {findMaxAverage_optimal(nums2, k2):.5f}")
    print(f"  Expected:    5.00000")
    
    # Test Case 3: All negative numbers
    nums3 = [-1, -2, -3, -4, -5]
    k3 = 2
    print("\nTest 3 (all negative):")
    print(f"  Input: nums = {nums3}, k = {k3}")
    print(f"  Brute Force: {findMaxAverage_bruteforce(nums3, k3):.5f}")
    print(f"  Optimal:     {findMaxAverage_optimal(nums3, k3):.5f}")
    print(f"  Expected:    -1.50000 (from [-1, -2])")
    
    # Test Case 4: Larger array
    nums4 = [0, 1, 1, 3, 3]
    k4 = 4
    print("\nTest 4:")
    print(f"  Input: nums = {nums4}, k = {k4}")
    print(f"  Brute Force: {findMaxAverage_bruteforce(nums4, k4):.5f}")
    print(f"  Optimal:     {findMaxAverage_optimal(nums4, k4):.5f}")
    print(f"  Expected:    2.00000 (from [1, 1, 3, 3])")
    
    # Visualization Example
    print("\n" + "="*70)
    print("DETAILED VISUALIZATION")
    print("="*70)
    result = findMaxAverage_with_visualization([1, 12, -5, -6, 50, 3], 4)


# ============================================================================
# WHY SLIDING WINDOW IS BETTER
# ============================================================================
"""
Brute Force Approach:
- For each position, sum k elements
- Time: O(n * k)
- If array has 10,000 elements and k = 1,000:
  → 10,000 * 1,000 = 10,000,000 operations 😱

Sliding Window Approach:
- Calculate first sum once
- Then just add/subtract one number each step
- Time: O(n)
- Same array (10,000 elements):
  → Only 10,000 operations 🚀

Sliding window is k times faster!

For k = 1,000, that's 1,000x speedup! 🔥

SLIDING WINDOW PATTERN:
This technique works whenever you need to:
1. Find something in a fixed-size contiguous subarray
2. The answer depends on the sum/product/etc of elements

Examples:
- Maximum sum of k consecutive elements
- Average of k consecutive elements  
- Minimum product of k consecutive elements
- etc.

The key: Instead of recalculating everything, 
         just update by removing old and adding new!
"""




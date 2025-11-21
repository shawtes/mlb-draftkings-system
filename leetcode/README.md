# 🎯 LeetCode Solutions

This folder contains well-documented LeetCode problem solutions with easy-to-understand explanations.

## 📁 Problems Solved

### 1️⃣ Squares of a Sorted Array
**File:** `squared_array.py`

**Problem:** Given a sorted array, return the squares sorted.

**Example:**
```
Input:  [-4, -1, 0, 3, 10]
Output: [0, 1, 9, 16, 100]
```

**Solutions:**
- ✅ **Easy Approach:** Square then sort - O(n log n)
- ✅ **Optimal Approach:** Two pointers - O(n) ⚡

**Key Concept:** Two Pointers Technique

---

### 2️⃣ Maximum Average Subarray
**File:** `max_average_subarray.py`

**Problem:** Find the contiguous subarray of length k with the maximum average.

**Example:**
```
Input:  nums = [1, 12, -5, -6, 50, 3], k = 4
Output: 12.75000
Explanation: [12, -5, -6, 50] has sum 51, avg = 51/4 = 12.75
```

**Solutions:**
- ✅ **Brute Force:** Check all subarrays - O(n * k)
- ✅ **Optimal Approach:** Sliding window - O(n) ⚡
- ✅ **Bonus:** Visualization version to see how it works!

**Key Concept:** Sliding Window Technique

---

## 🚀 Running the Solutions

Each file can be run independently:

```bash
# Run squares problem
python3 squared_array.py

# Run max average problem with visualization
python3 max_average_subarray.py
```

## 📚 Key Techniques Learned

### Two Pointers
- Use when array is sorted
- Compare elements from both ends
- Efficient for many sorted array problems

### Sliding Window
- Use for contiguous subarray problems
- Maintain a "window" of fixed or variable size
- Slide it across the array by adding/removing elements
- Avoids recalculating everything from scratch

## ⚡ Performance Comparison

| Problem | Brute Force | Optimal | Speedup |
|---------|-------------|---------|---------|
| Squared Array | O(n log n) | O(n) | ~log(n)x faster |
| Max Average | O(n * k) | O(n) | k times faster |

For large inputs, these optimizations make a HUGE difference! 🔥

---

## 💡 Why These Solutions?

Each solution includes:
- ✅ Commented problem description
- ✅ Multiple approaches (easy → optimal)
- ✅ Step-by-step walkthrough with examples
- ✅ Visual diagrams in comments
- ✅ Test cases that verify correctness
- ✅ Time/space complexity analysis
- ✅ Explanations of WHY the optimal solution is better

Perfect for learning AND interview prep! 🎓















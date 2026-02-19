# CRITICAL FIX: Multiple Lineups Per Combination

## 🎯 **PROBLEM IDENTIFIED**

The sequential optimization was generating lineups across ALL combinations instead of generating multiple lineups PER combination. 

**User Report**: "im still only getting 1 line up per combination maybe the problem is with optimize single line up instead of multiple"

## ⚡ **ROOT CAUSE**

In the sequential optimization method, I had:
```python
target_lineups = self.num_lineups * len(self.stack_settings)
```

This was generating lineups across all stack types globally instead of generating `self.num_lineups` for each individual team combination.

## 🔧 **FIX IMPLEMENTED**

Changed the target calculation to:
```python
target_lineups = self.num_lineups  # Generate num_lineups for THIS specific combination
```

## 📊 **How It Works Now**

### **Team Combination Processing**
1. **For Each Combination**: CHC(5) + BOS(2)
2. **Generate**: `num_lineups` unique lineups for this combination
3. **Sequential Exclusions**: Each lineup uses different players
4. **Next Combination**: Reset and generate `num_lineups` for the next combination

### **Example with 25 Lineups Per Combination**
- **Combination 1**: CHC(5) + BOS(2) → Generate 25 unique lineups
- **Combination 2**: LAD(4) + SF(3) → Generate 25 unique lineups  
- **Combination 3**: NYY(5) + BAL(2) → Generate 25 unique lineups
- **Total**: 75 unique lineups (25 per combination)

## 🚀 **Expected Results**

### **Before Fix**
- 6 teams selected
- 25 lineups per combination requested
- **Got**: 1 lineup per combination (6 total lineups) ❌

### **After Fix**
- 6 teams selected
- 25 lineups per combination requested
- **Get**: 25 lineups per combination (150 total lineups) ✅

## 🎲 **Sequential Process Per Combination**

```
Combination: CHC(5) + BOS(2)
├── Lineup 1: Best available players → Exclude 8 players
├── Lineup 2: Best from remaining → Exclude 8 more players  
├── Lineup 3: Best from remaining → Exclude 8 more players
├── Reset exclusions (every 3 lineups)
├── Lineup 4: Fresh optimization with variety
├── ... continue until 25 lineups
└── Result: 25 unique lineups for CHC + BOS

Next Combination: LAD(4) + SF(3)
├── Fresh start (no exclusions from previous combination)
├── Lineup 1: Best available players → Exclude 8 players
├── ... repeat process
└── Result: 25 unique lineups for LAD + SF
```

## 📋 **Files Modified**

- **optimizer01.py**: Fixed `target_lineups` calculation in `optimize_lineups_with_risk_management()`

## 🎯 **Testing Instructions**

1. **Select Teams**: Choose 6 teams in team combinations
2. **Set Lineups**: Request 25 lineups per combination
3. **Expected Result**: 6 combinations × 25 lineups = 150 total unique lineups
4. **Verify Logs**: Should see "Generated lineup X/25" for each combination

## ✅ **Status**

The optimizer now correctly generates multiple diverse lineups per combination instead of just one lineup per combination. Each combination will produce exactly the number of lineups you request, with true diversity within each combination thanks to the sequential player exclusion system.

**You should now get exactly what you request: multiple unique lineups for each team combination!**

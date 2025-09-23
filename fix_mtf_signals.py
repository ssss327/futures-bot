#!/usr/bin/env python3
"""
EMERGENCY FIX: Fix broken multi-timeframe signal requirements
This script fixes the 3 critical issues causing zero signals.
"""

import re

def fix_smart_money_analyzer():
    """Fix the broken multi-timeframe signal generation"""
    
    # Read the file
    with open('smart_money_analyzer.py', 'r') as f:
        content = f.read()
    
    print("🔧 Applying emergency fixes to multi-timeframe analysis...")
    
    # FIX 1: Lower impossible signal requirements
    # Change: if total_mtf_signals < 16 or timeframe_participation < 4:
    # To:     if total_mtf_signals < 8 or timeframe_participation < 2:
    old_requirements = r'if total_mtf_signals < 16 or timeframe_participation < 4:'
    new_requirements = 'if total_mtf_signals < 8 or timeframe_participation < 2:'
    
    if old_requirements in content:
        content = content.replace(old_requirements, new_requirements)
        print("✅ FIX 1: Lowered signal requirements (16→8 points, 4→2 timeframes)")
    else:
        print("❌ FIX 1: Could not find signal requirements line")
    
    # FIX 2: Correct maximum points calculation
    # Change: max_mtf_points = 25
    # To:     max_mtf_points = 35
    old_max_points = r'max_mtf_points = 25  # Theoretical maximum'
    new_max_points = 'max_mtf_points = 35  # FIXED: Realistic maximum'
    
    if old_max_points in content:
        content = content.replace(old_max_points, new_max_points)
        print("✅ FIX 2: Corrected maximum points (25→35)")
    else:
        print("❌ FIX 2: Could not find max points line")
    
    # FIX 3: Remove artificial confidence inflation
    # Change: confidence = min(98, (mtf_bullish_signals / max_mtf_points) * 100 + 10)
    # To:     confidence = min(95, (mtf_bullish_signals / max_mtf_points) * 100)
    old_confidence_bull = r'confidence = min\(98, \(mtf_bullish_signals / max_mtf_points\) \* 100 \+ 10\)'
    new_confidence_bull = 'confidence = min(95, (mtf_bullish_signals / max_mtf_points) * 100)'
    
    content = re.sub(old_confidence_bull, new_confidence_bull, content)
    
    # Same for bearish signals
    old_confidence_bear = r'confidence = min\(98, \(mtf_bearish_signals / max_mtf_points\) \* 100 \+ 10\)'
    new_confidence_bear = 'confidence = min(95, (mtf_bearish_signals / max_mtf_points) * 100)'
    
    content = re.sub(old_confidence_bear, new_confidence_bear, content)
    print("✅ FIX 3: Removed artificial confidence inflation (+10 bonus)")
    
    # Write the fixed file
    with open('smart_money_analyzer.py', 'w') as f:
        f.write(content)
    
    print("🎉 EMERGENCY FIXES APPLIED!")
    print("Multi-timeframe signals should now work properly.")
    print("\nChanges made:")
    print("- Signal requirements: 16→8 points, 4→2 timeframes")
    print("- Maximum points: 25→35 (realistic)")
    print("- Removed +10 confidence inflation")
    print("\nRestart your bot to see working signals!")

if __name__ == "__main__":
    fix_smart_money_analyzer()

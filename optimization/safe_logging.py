"""
Windows-safe logging utility for DFS optimizer
Handles Unicode emoji characters that cause issues on Windows console
"""

import logging

def safe_log_info(message):
    """Log info message with Windows-safe Unicode handling"""
    # Replace Unicode emojis with text equivalents for Windows compatibility
    emoji_replacements = {
        '🎯': '[TARGET]',
        '🔍': '[DEBUG]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🚀': '[ENHANCED]',
        '⚪': '[UNSELECTED]',
        '🔄': '[CHANGED]',
        '📊': '[RESULT]',
        '🧪': '[TEST]',
        '🔧': '[FIX]',
        '💡': '[TIP]',
        '🎉': '[COMPLETE]',
        '⭐': '[STAR]',
        '🆘': '[HELP]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    logging.info(safe_message)

def safe_log_debug(message):
    """Log debug message with Windows-safe Unicode handling"""
    emoji_replacements = {
        '🎯': '[TARGET]',
        '🔍': '[DEBUG]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🚀': '[ENHANCED]',
        '⚪': '[UNSELECTED]',
        '🔄': '[CHANGED]',
        '📊': '[RESULT]',
        '🧪': '[TEST]',
        '🔧': '[FIX]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    logging.debug(safe_message)

def safe_log_warning(message):
    """Log warning message with Windows-safe Unicode handling"""
    emoji_replacements = {
        '🎯': '[TARGET]',
        '🔍': '[DEBUG]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🚀': '[ENHANCED]',
        '⚪': '[UNSELECTED]',
        '🔄': '[CHANGED]',
        '📊': '[RESULT]',
        '🧪': '[TEST]',
        '🔧': '[FIX]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    logging.warning(safe_message)

def safe_log_error(message):
    """Log error message with Windows-safe Unicode handling"""
    emoji_replacements = {
        '🎯': '[TARGET]',
        '🔍': '[DEBUG]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🚀': '[ENHANCED]',
        '⚪': '[UNSELECTED]',
        '🔄': '[CHANGED]',
        '📊': '[RESULT]',
        '🧪': '[TEST]',
        '🔧': '[FIX]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    logging.error(safe_message)

# This script fixes enum syntax in your UI file
import re

# Read the UI file
with open('ui.ui', 'r', encoding='utf-8') as file:
    content = file.read()

# Pattern for finding problematic enum values
pattern = r'([A-Za-z0-9]+)::[A-Za-z0-9]+::([A-Za-z0-9]+)'
replacement = r'\1::\2'

# Apply the fix
fixed_content = re.sub(pattern, replacement, content)

# Save the fixed file
with open('ui.ui', 'w', encoding='utf-8') as file:
    file.write(fixed_content)

print("UI file has been fixed!")
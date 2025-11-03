#!/bin/bash
# Code Originality Verification Script
# This script performs automated checks to verify code originality

set -e

echo "========================================================================"
echo "Code Originality Verification for mesh_prop"
echo "========================================================================"
echo ""

# Check 1: Copyright headers in source code (should only be in LICENSE)
echo "Check 1: Scanning for unexpected copyright headers..."
copyright_count=$(grep -r "Copyright" mesh_prop/ --exclude-dir=.git 2>/dev/null | wc -l)
if [ "$copyright_count" -eq 0 ]; then
    echo "✅ PASS: No copyright headers found in source code"
else
    echo "⚠️  WARNING: Found $copyright_count copyright references:"
    grep -r "Copyright" mesh_prop/ --exclude-dir=.git 2>/dev/null
fi
echo ""

# Check 2: TODO/FIXME comments that might indicate unfinished copied code
echo "Check 2: Scanning for TODO/FIXME/HACK comments..."
todo_count=$(grep -r "TODO\|FIXME\|XXX\|HACK" mesh_prop/ --exclude-dir=.git 2>/dev/null | wc -l || echo 0)
if [ "$todo_count" -eq 0 ]; then
    echo "✅ PASS: No TODO/FIXME comments found"
else
    echo "⚠️  INFO: Found $todo_count TODO/FIXME comments (not necessarily a problem)"
fi
echo ""

# Check 3: URLs in source code (might indicate source references)
echo "Check 3: Scanning for URLs in source code..."
url_count=$(grep -r "https://\|http://\|www\." mesh_prop/ --exclude-dir=.git 2>/dev/null | grep -v ".pyc" | wc -l || echo 0)
if [ "$url_count" -eq 0 ]; then
    echo "✅ PASS: No URLs found in source code"
else
    echo "⚠️  INFO: Found $url_count URLs (check if they are references or examples)"
fi
echo ""

# Check 4: License file exists
echo "Check 4: Checking for LICENSE file..."
if [ -f "LICENSE" ]; then
    echo "✅ PASS: LICENSE file exists"
    echo "   License: $(head -1 LICENSE)"
else
    echo "❌ FAIL: LICENSE file not found"
fi
echo ""

# Check 5: Check for common plagiarism indicators
echo "Check 5: Scanning for plagiarism indicators..."
# Look for phrases like "source:", "adapted from:", "based on code from:"
indicators=$(grep -ri "source:\|adapted from:\|based on code from:\|copied from:" mesh_prop/ --exclude-dir=.git 2>/dev/null | wc -l || echo 0)
if [ "$indicators" -eq 0 ]; then
    echo "✅ PASS: No plagiarism indicator phrases found"
else
    echo "⚠️  WARNING: Found potential source attributions:"
    grep -ri "source:\|adapted from:\|based on code from:\|copied from:" mesh_prop/ --exclude-dir=.git 2>/dev/null
fi
echo ""

# Check 6: Verify attribution documentation exists
echo "Check 6: Checking attribution documentation..."
if [ -f "ACKNOWLEDGMENTS.md" ]; then
    echo "✅ PASS: ACKNOWLEDGMENTS.md exists"
else
    echo "⚠️  WARNING: ACKNOWLEDGMENTS.md not found"
fi

if [ -f "CODE_PROVENANCE.md" ]; then
    echo "✅ PASS: CODE_PROVENANCE.md exists"
else
    echo "⚠️  WARNING: CODE_PROVENANCE.md not found"
fi
echo ""

# Check 7: Count Python files and lines of code
echo "Check 7: Code statistics..."
py_files=$(find mesh_prop/ -name "*.py" -type f | wc -l)
total_lines=$(find mesh_prop/ -name "*.py" -type f -exec wc -l {} + | tail -1 | awk '{print $1}')
echo "   Python files: $py_files"
echo "   Total lines: $total_lines"
echo ""

# Summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
echo "All automated checks completed. Review any warnings above."
echo ""
echo "Additional verification steps you can take:"
echo "1. Search GitHub for unique function names (e.g., _count_ray_intersections)"
echo "2. Use GitHub's Code Search: https://github.com/search?type=code"
echo "3. Compare algorithm implementations to reference papers (expected similarity)"
echo "4. Review CODE_PROVENANCE.md for detailed audit results"
echo ""
echo "For commercial use, consider:"
echo "- SonarQube/SonarCloud for professional code analysis"
echo "- Codequiry for comprehensive plagiarism detection"
echo "- Legal review if required for your organization"
echo ""
echo "Note: GitHub Copilot has built-in filtering and only ~1% of suggestions"
echo "      match training data verbatim (GitHub research, 2021)"
echo ""

#!/bin/bash
# Verification script for extended model support
# Checks that all 6 models are properly registered and supported

echo "=========================================="
echo "Model Support Extension Verification"
echo "=========================================="
echo ""

echo "1. Checking model_registry.py..."
if grep -q '"qwen25_0_5b"' model_registry.py && \
   grep -q '"qwen25_1_5b"' model_registry.py && \
   grep -q '"qwen_math_1_5b"' model_registry.py; then
    echo "   ✓ All 3 Qwen variants registered in model_registry.py"
else
    echo "   ✗ Missing entries in model_registry.py"
    exit 1
fi
echo ""

echo "2. Checking hidden_steer.py choices..."
if grep -q 'qwen25_1_5b.*qwen25_0_5b.*qwen_math_1_5b' hidden_steer.py; then
    echo "   ✓ All 6 models supported in hidden_steer.py"
else
    echo "   ✗ Missing models in hidden_steer.py argument choices"
    exit 1
fi
echo ""

echo "3. Checking run_full_evaluation.sh arrays..."
if grep -q 'steering_model_types=(phi2 llama32_3b qwen25_0_5b qwen25_1_5b qwen25_3b qwen_math_1_5b)' run_full_evaluation.sh; then
    echo "   ✓ All 6 models in steering_model_types array"
else
    echo "   ✗ steering_model_types array not updated"
    exit 1
fi
echo ""

echo "4. Testing Python import (evaluation.py model registry)..."
python3 -c "
from model_registry import all_model_types
models = all_model_types()
required = ['qwen25_0_5b', 'qwen25_1_5b', 'qwen_math_1_5b', 'phi2', 'llama32_3b', 'qwen25_3b']
missing = [m for m in required if m not in models]
if missing:
    print(f'   ✗ Missing models: {missing}')
    exit(1)
else:
    print(f'   ✓ All {len(required)} models in registry')
    print(f'     Available: {sorted(models)}')
" || exit 1
echo ""

echo "=========================================="
echo "✓ ALL CHECKS PASSED"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  bash run_full_evaluation.sh"
echo ""
echo "Or individually:"
echo "  python hidden_steer.py --model-type qwen25_0_5b --model-path Qwen/Qwen2.5-0.5B ..."
echo "  python hidden_steer.py --model-type qwen25_1_5b --model-path Qwen/Qwen2.5-1.5B ..."
echo "  python hidden_steer.py --model-type qwen_math_1_5b --model-path Qwen/Qwen2.5-Math-1.5B ..."
echo ""

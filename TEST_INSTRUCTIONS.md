# GLM-4.5 Parser Testing Instructions

**Purpose:** Execute end-to-end tests for GLM-4.5 tool calling and reasoning extraction features.

**Prerequisites:** GLM-4.5-Air model in `/opt/tabbyAPI/models/GLM-4.5-Air/`

**NOTE: Remember to kill tabbyAPI server process and make sure it is stopped before ending your session!!**
**NOTE: Use `nvidia-smi` to check the GPU(s), if they have more than 1GB each still loaded in VRAM, tabbyAPI is still running!**
---

## 1. Start TabbyAPI Server

### 1.1 Navigate to Project Directory
```bash
cd /opt/GLM_tabbyAPI
```

### 1.2 Activate Virtual Environment
```bash
source venv/bin/activate
```

### 1.3 Launch Server in Background
```bash
python start.py > /tmp/tabby_server.log 2>&1 &
```

**Store PID for later cleanup:**
```bash
echo $! > /tmp/tabby_server.pid
```

### 1.4 Monitor Server Startup

**Initial check (first 30 seconds):**
```bash
tail -f /tmp/tabby_server.log
```

**Watch for these log entries in sequence:**

1. Backend initialization:
   ```
   INFO: Using backend exllamav3
   INFO: exllamav3 version: 0.0.7
   ```

2. Parser initialization (check for errors):
   ```
   WARNING: Failed to initialize GLM-4.5 parsers: ...
   # OR
   INFO: Initialized GLM-4.5 parsers for model: ...
   ```

3. Model loading:
   ```
   INFO: Loading model: /opt/tabbyAPI/models/GLM-4.5-Air
   INFO: Loading with tensor parallel
   ```

4. Model loaded (wait ~2-3 minutes):
   ```
   INFO: Model successfully loaded.
   Loading model modules ━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 49/49
   ```

5. Server ready:
   ```
   INFO: Started server process [PID]
   INFO: Waiting for application startup.
   INFO: Application startup complete.
   INFO: Uvicorn running on http://10.1.1.10:8080
   ```

**Press `Ctrl+C` to stop following logs once "Application startup complete" appears.**

### 1.5 Verify Server Ready

```bash
curl -s http://10.1.1.10:8080/v1/models | jq .
```

**Expected:** JSON response with model list. If connection refused, wait 10s and retry.

---

## 2. Run Tests

### 2.1 Environment Setup

Ensure pytest installed:
```bash
python -m pip list | grep pytest || python -m pip install pytest requests --break-system-packages
```

### 2.2 Test Execution Commands

#### Run All Tests
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py -v --tb=short 2>&1 | tee /tmp/test_run_$(date +%Y%m%d_%H%M%S).log
```

#### Run Specific Test Categories

**Server health checks:**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_server_health_check -v
python -m pytest tests/e2e/test_glm4_full_flow.py::test_model_loaded -v
```

**Tool calling (non-streaming):**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_tool_calling_weather_example -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_tool_calling_multiple_functions -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_tool_calling_with_conversation_context -v -s
```

**Reasoning extraction:**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_reasoning_extraction_basic -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_reasoning_with_math_problem -v -s
```

**Streaming tests:**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_streaming_basic -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_streaming_tool_calls -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_streaming_reasoning -v -s
```

**Combined features:**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_combined_reasoning_and_tools -v -s
```

**OpenAI SDK compatibility:**
```bash
# Install SDK first
python -m pip install openai --break-system-packages

# Run tests
python -m pytest tests/e2e/test_glm4_full_flow.py::test_openai_sdk_compatibility -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_openai_sdk_tool_calling -v -s
```

**Error handling:**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_invalid_tool_schema -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_empty_messages -v -s
python -m pytest tests/e2e/test_glm4_full_flow.py::test_very_long_context -v -s
```

### 2.3 Viewing Detailed Output

**With stdout/stderr (`-s` flag):**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_tool_calling_weather_example -v -s
```

Shows model responses, tool calls, reasoning content in console.

**With increased verbosity (`-vv` flag):**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py -vv
```

Shows detailed assertion failures and full diffs.

---

## 3. Analyze Results

### 3.1 Check Test Summary

Look for final line in test output:
```
================== 14 passed, 2 skipped in 182.50s ===================
```

### 3.2 Examine Server Logs for Parser Issues

```bash
grep -i "parser" /tmp/tabby_server.log
```

**Critical issue to check:**
```bash
grep "Failed to initialize GLM-4.5 parsers" /tmp/tabby_server.log
```

If found, parsers are NOT working (known bug).

### 3.3 Inspect Test Responses

**For tool calling tests, check if extraction worked:**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_tool_calling_weather_example -v -s 2>&1 | grep -A 5 "Tool calls:"
```

**Expected if parsers work:** `Tool calls: [{"type": "function", ...}]`
**Actual if broken:** `Tool calls: None`

**For reasoning tests:**
```bash
python -m pytest tests/e2e/test_glm4_full_flow.py::test_reasoning_extraction_basic -v -s 2>&1 | grep -A 5 "Reasoning:"
```

**Expected if parsers work:** `Reasoning: "The user is asking..."`
**Actual if broken:** `Reasoning: None`

---

## 4. Update Testing Worklog

### 4.1 Gather Information

**After each test run, collect:**

1. **Run Number:** Increment from last entry in `TESTING_WORKLOG.md`

2. **Timestamp:** Generate with:
   ```bash
   date +%Y%m%d-%H%M%S
   ```

3. **Test Results:** Pass/fail counts from pytest summary

4. **Problems Found:** Check server logs:
   ```bash
   grep -E "(ERROR|WARNING|Failed)" /tmp/tabby_server.log | tail -20
   ```

5. **Error Messages:** Copy exact error text

6. **Suspected Root Cause:** Based on error analysis

### 4.2 Append Entry to Worklog

```bash
# Open worklog in editor
nano TESTING_WORKLOG.md

# OR append programmatically (example):
cat >> TESTING_WORKLOG.md << 'EOF'

---

## Run #2

**Timestamp:** 20251004-143022
**Duration:** 185.3s
**Test Command:** `python -m pytest tests/e2e/test_glm4_full_flow.py -v`

### Results
- **Passed:** 14
- **Failed:** 0
- **Skipped:** 2

### Problems Identified

1. **Parser initialization failure**
   - **Error:** `'Tokenizer' object has no attribute 'get_vocab'`
   - **Impact:** Tool calls and reasoning not extracted
   - **Evidence:** `tool_calls: None`, `reasoning_content: None` in all test responses

### Error Messages
```
2025-10-04 14:28:47.325 WARNING: Failed to initialize GLM-4.5 parsers:
'Tokenizer' object has no attribute 'get_vocab'. Tool calling and reasoning
extraction will be unavailable.
```

### Short Description
Parser initialization fails on server startup due to incorrect tokenizer API call. Model generates correct XML but parsers cannot extract tool calls or reasoning.

### Suspected Problem
API incompatibility: Parser calls `tokenizer.get_vocab()` but ExllamaV3 Tokenizer has `get_vocab_dict()` method instead. Located in `common/parsers/abstract_tool_parser.py:40`.

### Actions Taken
- Documented root cause in TEST_RESULTS.md
- Identified one-line fix required
- Verified model generates correct XML output

EOF
```

### 4.3 Worklog Entry Template

Copy this template for each new run:

```markdown
---

## Run #[NUMBER]

**Timestamp:** [YYYYMMDD-HHMMSS]
**Duration:** [X.Xs]
**Test Command:** `[command used]`

### Results
- **Passed:** [count]
- **Failed:** [count]
- **Skipped:** [count]

### Problems Identified

1. **[Problem title]**
   - **Error:** `[exact error message]`
   - **Impact:** [what broke]
   - **Evidence:** [how you know]

### Error Messages
```
[paste relevant error logs]
```

### Short Description
[2-3 sentence summary of what happened]

### Suspected Problem
[Analysis of root cause with file/line references]

### Actions Taken
- [what you did]
- [what you discovered]
```

---

## 5. Stop Server

### 5.1 Graceful Shutdown

```bash
# If PID was saved:
kill $(cat /tmp/tabby_server.pid)

# OR find and kill:
pkill -f "python start.py"
```

### 5.2 Force Kill (if unresponsive)

```bash
pkill -9 -f "python start.py"
```

### 5.3 Verify Stopped

```bash
ps aux | grep "python start.py" | grep -v grep
```

Should return nothing if server stopped.

### 5.4 Archive Logs

```bash
# Save logs with timestamp
cp /tmp/tabby_server.log logs/tabby_server_$(date +%Y%m%d_%H%M%S).log
```

---

## 6. Troubleshooting

### Server Won't Start

**Check port availability:**
```bash
lsof -i :8080
```

If occupied, kill process or change port in `config.yml`.

**Check model path:**
```bash
ls -la /opt/tabbyAPI/models/GLM-4.5-Air/
```

Must contain model files (safetensors, config.json, etc.).

**Check GPU availability:**
```bash
nvidia-smi
```

### Tests Fail to Connect

**Verify server listening:**
```bash
netstat -tuln | grep 8080
```

**Test direct connection:**
```bash
curl -v http://10.1.1.10:8080/v1/models
```

**Check firewall:**
```bash
sudo iptables -L | grep 8080
```

### Parser Issues

**Check parser initialization in logs:**
```bash
grep -C 3 "GLM-4.5 parsers" /tmp/tabby_server.log
```

**Verify tokenizer API:**
```bash
python << 'EOF'
import sys
sys.path.insert(0, 'venv/lib/python3.12/site-packages')
from exllamav3 import Tokenizer
print("Methods:", [m for m in dir(Tokenizer) if 'vocab' in m.lower()])
EOF
```

Expected: `['get_vocab_dict', ...]` (NOT `get_vocab`)

### Model Loading Hangs

**Monitor GPU memory:**
```bash
watch -n 1 nvidia-smi
```

**Check for OOM errors:**
```bash
dmesg | tail -50 | grep -i "out of memory"
```

**Reduce cache size in config.yml:**
```yaml
model:
  cache_size: 65536  # Reduce from 131072
```

---

## 7. Quick Reference

### One-Command Test Cycle

```bash
# Stop any running server
pkill -f "python start.py"

# Start server
cd /opt/GLM_tabbyAPI && \
source venv/bin/activate && \
python start.py > /tmp/tabby_server.log 2>&1 &

# Wait for startup (watch logs)
tail -f /tmp/tabby_server.log
# Press Ctrl+C when you see "Application startup complete"

# Run tests
python -m pytest tests/e2e/test_glm4_full_flow.py -v --tb=short 2>&1 | tee /tmp/test_run_$(date +%Y%m%d_%H%M%S).log

# Check for parser issues
grep "Failed to initialize GLM-4.5 parsers" /tmp/tabby_server.log

# Update worklog (manually)
nano TESTING_WORKLOG.md
```

### Important File Locations

- **Server logs:** `/tmp/tabby_server.log`
- **Test logs:** `/tmp/test_run_*.log`
- **Config:** `/opt/GLM_tabbyAPI/config.yml`
- **Model:** `/opt/tabbyAPI/models/GLM-4.5-Air/`
- **Test file:** `/opt/GLM_tabbyAPI/tests/e2e/test_glm4_full_flow.py`
- **Test results:** `/opt/GLM_tabbyAPI/TEST_RESULTS.md`
- **Worklog:** `/opt/GLM_tabbyAPI/TESTING_WORKLOG.md`

### Key Log Patterns

| Pattern | Meaning |
|---------|---------|
| `Failed to initialize GLM-4.5 parsers` | Parsers not working (CRITICAL) |
| `Model successfully loaded` | Model ready for inference |
| `Application startup complete` | Server ready for requests |
| `tool_calls: None` in test output | Parser extraction failed |
| `reasoning_content: None` | Reasoning not extracted |
| `<tool_call>` in content field | Model generates XML but parsers don't extract |

---

## 8. Test Validation Checklist

After each run, verify:

- [ ] Server started without errors
- [ ] Model loaded successfully (check logs)
- [ ] Parser initialization status checked (success/failure)
- [ ] All tests executed (14 passed + 2 skipped OR different results)
- [ ] Tool call extraction tested (check if `tool_calls` populated)
- [ ] Reasoning extraction tested (check if `reasoning_content` populated)
- [ ] Streaming functionality verified
- [ ] Error responses validated (empty messages, invalid schema)
- [ ] Worklog updated with run details
- [ ] Server stopped cleanly

---

**Last Updated:** 2025-10-04
**Version:** 1.0
**Tested With:** ExllamaV3 0.0.7, GLM-4.5-Air, TabbyAPI GLM4 branch

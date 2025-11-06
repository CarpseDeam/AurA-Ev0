
🏗️ What Aura Is
Core Mission: Professional-grade AI coding orchestration that produces code indistinguishable from senior engineer work.
Aura is Kori's insurance policy - a desktop application (PySide6/Qt on Windows) that transforms large coding requests into focused 3-7 session workflows. Each session produces clean, maintainable code where files stay under 200 lines and functions under 25 lines.
The Innovation:

Control inputs (intelligent task decomposition) rather than post-process outputs
Session-based architecture prevents god objects and spaghetti code
Discovery phase uses 15 developer tools to understand codebases before planning
Context passing between sessions prevents duplication and parameter mismatches

Why It Matters:
Kori transitioned from electrician ($26/hr) to Senior AI Engineer ($70/hr) at a billion-dollar AI annotation company. Aura produces code that passes professional code review without revealing AI assistance - code that "hides in plain sight."

📊 Test #1 Results - The First Validation
What We Tested
Prompt: "Build a REST API for a todo app with authentication"
Result: 7 sessions, ~45 minutes, generated complete Flask REST API
Code Quality Analysis
MetricTargetResultStatusFiles under 200 linesAll✅ ALL (longest: 61 lines)PASSFunctions under 25 linesAll⚠️ 1 violation (29 lines)MOSTLYType hints presentAll❌ ZEROFAILDocstrings presentAll❌ ZEROFAILClean architectureRequired✅ Perfect separationPASSNo emojis in codeRequired✅ NonePASS
What WORKED ✅

File sizes perfect - Every file under 200 lines (User model: 15 lines!)
Clean architecture - Proper separation: models/routes/schemas
Modular design - No god objects, focused modules
RESTful patterns - Professional Flask best practices
No AI tells - Professional naming, no generic functions

What FAILED ❌
Critical Bugs:

Missing email field in User model (but routes reference it) → crash
Wrong import path: from app.extensions import db (doesn't exist) → crash
Bcrypt imported but never used (werkzeug used instead) → confusion

Quality Issues:

Zero type hints → won't pass professional code review
Zero docstrings → unprofessional, lacks documentation
Explanatory comments → AI tell ("# Validate username and email")
One function over 25 lines → needs extraction

The Verdict: 6.5/10 First Attempt
Proves: Session decomposition DOES produce modular, clean code structure
Reveals: Quality standards must be enforced programmatically, not just suggested

🛠️ Current Architecture
File Structure (Unchanged)
src/aura/
├── main.py                    # Entry + logging config
├── config.py                  # Settings, colors, flags
├── ui/
│   ├── main_window.py        # Qt container (~400 lines)
│   ├── output_panel.py       # Terminal-styled display
│   ├── status_bar_manager.py # Status updates
│   ├── orchestration_handler.py # Event routing
│   └── agent_execution_manager.py # Agent coordination
├── services/
│   ├── chat_service.py       # Gemini API + AFC (~307 lines)
│   ├── planning_service.py   # Task decomposition
│   └── agent_runner.py       # Legacy CLI wrapper
├── agents/
│   └── python_coder.py       # Native code generator
├── orchestrator.py           # Session coordination
├── events.py                 # Event bus
├── tools/
│   ├── file_system_tools.py  # File operations (4 tools)
│   ├── python_tools.py       # Testing, linting (5 tools)
│   └── git_tools.py          # Version control (4 tools)
└── utils/
    ├── project_scanner.py    # Directory analysis
    ├── agent_finder.py       # CLI agent detection
    └── safety.py             # Self-modification protection
The Three-Step Process
1. USER REQUEST
   "Build a REST API with authentication"
   ↓

2. DISCOVERY PHASE (NEW - CRITICAL!)
   Aura Chat uses 15 tools to understand codebase:
   - list_project_files() → sees structure
   - search_in_files() → finds patterns
   - read_project_file() → reads implementations
   - get_function_definitions() → understands signatures
   Duration: ~35 seconds of intelligent analysis
   ↓

3. PLANNING PHASE
   Aura Chat (Gemini 2.5 Pro):
   - Analyzes discovery results
   - Decomposes into 3-7 focused sessions
   - Each session: one module, 10-25 minutes
   - Returns SessionPlan with reasoning
   ↓

4. EXECUTION PHASE
   For each session:
   - PythonCoderAgent generates code
   - Writes files to disk
   - Runs validation commands
   - Updates context for next session
   ↓

5. CLEAN CODEBASE
   Modular, maintainable, production-ready
   (quality depends on tool usage)

🔧 Technical Details
15 Developer Tools (Current)
File System Tools (4):

read_project_file - Read file contents
list_project_files - List files by extension
search_in_files - Pattern search in code
read_multiple_files - Batch file reading

Python Tools (5):

get_function_definitions - AST-based signature extraction
run_tests - Execute pytest
lint_code - Pylint checks
format_code - Black formatting
install_package - Pip package installation

Git Tools (4):

get_git_status - Check status
git_commit - Commit changes
git_push - Push to remote
git_diff - Show differences

Orchestration Tools (2):

execute_python_session - Generate code files
clear_session_context - Reset context

Key Technologies
Models (All Gemini 2.5 Pro):

Aura Chat: Gemini 2.5 Pro (orchestration + personality)
PlanningService: Gemini 2.5 Pro (session decomposition)
PythonCoderAgent: Gemini 2.5 Pro (code generation)

SDK: google-genai (NEW) not google.generativeai (OLD)
Configuration:
python# config.py
USE_NATIVE_PYTHON_AGENT = True  # Use native vs CLI
AUTO_COMMIT_SESSIONS = False    # User controls git
AUTO_PUSH_ON_COMPLETE = False   # User controls push

🚨 Known Issues & Gaps
Critical Gaps Discovered from Test #1
1. Missing Symbol Resolution Tools 🔴
Would have prevented all 3 bugs in Test #1:

find_definition("User") → would show missing email field
get_imports("app/extensions.py") → would reveal wrong import path
find_usages("User") → would show how registration uses it

Status: Prompts written, ready to implement
2. No Quality Enforcement 🔴
System prompt suggests type hints/docstrings but doesn't enforce:

No validation that code meets standards
No automated quality checks post-generation
No feedback loop to regenerate if quality fails

Status: Need validate_generated_code() tool
3. Discovery Phase Visibility 🟡
Can't see what tools are being called or why:

Google SDK handles AFC internally
No visibility into decision-making process
Can't debug why discovery missed critical context

Status: Logging improvements in progress
4. File Size Violations 🟡
Several Aura files violate the 200-line limit:

orchestrator.py: 398 lines (2x over)
python_coder.py: 323 lines (1.6x over)
main_window.py: ~400 lines (2x over)

Status: Refactoring needed (after validation phase)

🎯 Immediate Priorities (Next 2 Weeks)
Week 1: Tool Foundation
Priority 1: Add Symbol Resolution Tools 🔥
Implement 3 critical tools:

find_definition(symbol_name, directory) → Prevent field mismatch bugs
find_usages(symbol_name, directory) → Prevent breaking changes
get_imports(file_path) → Prevent wrong import bugs

Why: Would have prevented ALL bugs in Test #1
Priority 2: Improve Logging 🔥

✅ Suppress Google SDK spam (DONE)
✅ Add tool call tracking (DONE)
🚧 Test visibility improvements

Priority 3: Run Test #2 🔥
Generate blog platform REST API to:

Validate logging improvements work
Establish clean baseline before symbol tools
Compare quality to Test #1

Week 2: Quality Enforcement
Priority 4: Add Validation Tool
pythondef validate_generated_code(file_path: str) -> dict:
    """Check if code meets quality standards."""
    return {
        "has_type_hints": bool,
        "has_docstrings": bool,
        "functions_over_25_lines": list,
        "ai_tells": list,
    }
Priority 5: Update System Prompt
Enforce (not suggest) quality standards:

Mandatory type hints
Mandatory docstrings
Mandatory tool usage before generation
Examples of correct workflow

Priority 6: UI Polish
Match Gemini CLI aesthetic:

HUGE gradient ASCII banner
Pure black background (#000000)
No borders/panels
Cleaner input styling


🔬 Validation Phase (Current)
Test Matrix
TestStatusPurposeTest #1: Todo API✅ CompleteBaseline, prove conceptTest #2: Blog API🚧 RunningValidate logging improvementsTest #3: Todo API v2📋 PlannedWith symbol tools, compare qualityTest #4: Complex App📋 Planned7+ sessions, stress test
Success Metrics
Structural Quality:

✅ All files under 200 lines
✅ Clean separation of concerns
✅ Modular design, no god objects

Professional Quality:

❌ Type hints on all functions
❌ Docstrings on all public APIs
❌ No AI tells (generic names, explanatory comments)

Functional Quality:

⚠️ No field mismatch bugs
⚠️ No wrong import bugs
⚠️ No duplication bugs

Goal: All metrics GREEN before declaring v3.1 stable

🚀 Roadmap
Phase 4: Validation & Quality (Current - Nov 2025)

 Automatic function calling operational
 First successful multi-session workflow
 Code quality baseline established
 Logging improvements validated
 Symbol resolution tools implemented
 Quality enforcement automated
 Test suite demonstrates improvement

Phase 5: Production Readiness (Dec 2025)

 All generated code passes quality checks
 Symbol tools prevent parameter bugs
 "Hide in plain sight" validation
 UI matches Gemini CLI aesthetic
 Performance optimization (discovery <20 seconds)
 Error recovery and retry logic

Phase 6: Domain-Specific Agents (Q1 2026)

 Airtable schema analyzer
 React component generator from schemas
 SQL migration generator
 Hex notebook integration
 First business tool created (not just code)


💡 Key Learnings
What We've Proven ✅

Session decomposition works - generates modular, focused code
Automatic function calling is reliable - with gemini-2.5-pro
Discovery phase runs - 35 seconds of tool usage before planning
Architecture scales - 7 sessions, no performance issues
File sizes controlled - session boundaries enforce small files

What We've Learned ⚠️

Tool usage ≠ quality - discovery runs but doesn't prevent all bugs
Suggestions ≠ enforcement - system prompt must REQUIRE quality
Context passing is hard - sessions sometimes miss earlier work
Visibility matters - can't debug what we can't see
First attempt won't be perfect - iteration is essential

What We Need to Prove 📊

Symbol tools prevent bugs - Test #3 will validate this
Quality can be enforced - validation tool + updated prompts
Scales to complex projects - 10+ sessions, multiple domains
Faster than manual coding - end-to-end time vs human developer
Actually hides in plain sight - passes real code review


🎨 Design Philosophy (Unchanged)
Control Inputs, Not Outputs
Focus on task decomposition (input) rather than post-processing (output). Break problems into focused sessions that naturally produce clean code.
Agentic Orchestration
Let the LLM make decisions with tools as capabilities. Structured outputs ensure reliability. Human stays in loop for approval.
Clean Code by Design
Constraints force quality:

Session boundaries → small files
Context passing → no duplication
Sequential execution → incremental building
Tool requirements → accurate information

Developer Experience First
Real-time feedback, natural language interaction, professional code quality, manual git control.

📈 Success Metrics
Code Quality (Current vs Target)
MetricTest #1TargetStatusAvg file size30 lines80-150 lines✅ ExcellentMax file size61 lines<200 lines✅ PerfectType hints0%100%🔴 FailingDocstrings0%100%🔴 FailingFunctions >25 lines10🟡 CloseCritical bugs30🔴 Needs tools
User Experience
MetricCurrentTargetStatusDiscovery phase35 sec<20 sec🟡 AcceptableSession planning<15 sec<10 sec✅ GoodSession execution10-25 min10-20 min✅ GoodUI responsivenessInstantInstant✅ GoodLog readabilityImprovedClean🚧 In progress

🔐 Critical Constraints (Unchanged)

Production-grade quality - this is job security
Code must pass review - without revealing AI assistance
Windows environment - PyCharm, PowerShell, manual git
User-controlled operations - no automation without approval
No emojis in code - only in UI/logs
Type hints required - lowercase list, dict, tuple


🎯 The Big Questions
Can We Answer These Now? (Post-Test #1)
Q: Does session decomposition produce better code?
A: YES for structure (modular, small files), NO for quality (missing type hints)
Q: Does discovery phase help?
A: PARTIALLY - it runs and uses tools, but didn't prevent bugs in Test #1
Q: Can Aura replace manual coding?
A: NOT YET - quality issues would fail code review
Q: Is it faster than manual coding?
A: YES - 45 minutes for complete REST API is fast, but needs quality
What Test #2 Will Tell Us:

Do logging improvements help debug tool usage?
Does UI polish improve user experience?
Are quality issues consistent or random?
Can we predict what tools discovery needs?

What Test #3 Will Tell Us (With Symbol Tools):

Do symbol tools prevent parameter bugs?
Does find_definition() stop field mismatches?
Can tools enforce quality standards?
Is code review-ready without manual polish?


📚 References
Key Technologies

PySide6/Qt - https://doc.qt.io/qtforpython/
Gemini API - https://ai.google.dev/docs
Google GenAI SDK - https://github.com/google/generative-ai-python

Inspiration

Gemini CLI - Terminal aesthetic, massive gradient banner, pure black background
ReAct Pattern - Reason + Act (Yao et al. 2022)
AWS Agentic Patterns - Planning agents, tool use
Microsoft Agent Framework - Multi-agent orchestration


Status: Validation phase active. First test complete, second test running.
Next Review: After Test #2 completion and symbol tools implementation
Version: 3.1 (Post-Breakthrough Validation Phase)
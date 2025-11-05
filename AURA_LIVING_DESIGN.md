# Aura - Living Design Document
**Version: 3.0 (Native Agent Architecture)**  
**Last Updated: 2025-01-06**

---

## 🎯 Implementation Snapshot (2025-01-06)

### What Changed in v3.0
**MAJOR ARCHITECTURE SHIFT:** Replaced subprocess CLI wrappers with native Python agents.

**Before (v2.0):**
- Orchestrator → subprocess → Gemini CLI → stdout parsing → pain

**After (v3.0):**
- Aura Chat (Gemini API) → function calling → PythonCoderAgent → structured results → clean

### Current Architecture
- **PySide6 desktop shell** with dark theme, real-time output streaming
- **Aura Chat LLM** (Gemini 2.5 Pro) as orchestrating brain with personality
- **Function calling** via Gemini native API - tools registered directly
- **PythonCoderAgent** - custom Python agent with its own LLM for code generation
- **Structured outputs** - Pydantic models, no stdout parsing
- **Qt signals** for live UI updates throughout the stack
- **Git integration** via tools (read files, commit, push)

### File Structure
```
src/aura/
├── main.py                    # Application entry point
├── config.py                  # Colors, fonts, settings
├── ui/
│   ├── main_window.py        # Main Qt window (~400 lines)
│   └── agent_settings_dialog.py
├── services/
│   ├── chat_service.py       # Gemini API + tools (~250 lines)
│   ├── planning_service.py   # Task decomposition (~150 lines)
│   └── agent_runner.py       # Legacy CLI wrapper (fallback)
├── agents/
│   └── python_coder.py       # 🆕 Native coding agent (~200 lines)
├── orchestrator.py           # Session coordination (~300 lines)
├── events.py                 # Event bus for coordination
├── tools.py                  # Git helpers
└── utils/
    ├── project_scanner.py    # Directory analysis
    ├── agent_finder.py       # Detect available CLI agents
    └── safety.py            # Prevent self-modification
```

---

## 🧠 What Aura Actually Does

### The Three-Step Process (Updated for v3.0)

```
1. USER REQUEST
   "Build a REST API with authentication"
   ↓

2. AURA CHAT LLM (Gemini 2.5 Pro + Function Calling)
   - Reads existing files: read_project_file()
   - Plans sessions: Decomposes into 3-7 focused tasks
   - Executes code: execute_python_session()
   - Verifies results: read_project_file() again
   - Commits work: git_commit(), git_push()
   ↓

3. PYTHONCODERAGENT (Worker Agent)
   - Called as a tool by Aura Chat
   - Generates code using Gemini API
   - Writes files to disk
   - Runs validation commands
   - Returns structured AgentResult
   ↓

4. CLEAN CODEBASE
   Small files, clear separation, maintainable
```

### Example Session Flow

**Session 1: "Create User model ONLY"**
```python
Aura Chat thinks:
1. read_project_file("models/user.py") → doesn't exist
2. execute_python_session(
     "Create User model with bcrypt password hashing"
   )
3. PythonCoderAgent creates models/user.py (87 lines)
4. Returns: {"success": true, "files_created": ["models/user.py"]}
5. read_project_file("models/user.py") → verify it's good
```

**Session 2: "Create login/logout routes using existing User model"**
```python
Aura Chat thinks:
1. read_project_file("models/user.py") → exists! Don't recreate
2. execute_python_session(
     "Create auth routes using existing User model at models/user.py"
   )
3. PythonCoderAgent creates routes/auth.py (134 lines)
4. Returns structured results
```

**Key Innovation: The LLM Sees Everything**
- Aura Chat reads files before and after sessions
- It UNDERSTANDS what changed and why
- It can adapt the plan based on what actually happened
- It decides when to commit code, when to iterate

---

## 🏗️ Architecture Diagram (v3.0)

```
┌──────────────────────────────────────────────────┐
│           Aura GUI (PySide6/Qt)                  │
│  ┌────────────────────────────────────────────┐  │
│  │   QTextEdit (live output stream)           │  │
│  │   - Colored, formatted HTML                │  │
│  │   - Auto-scroll, timestamps                │  │
│  └────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │   QLineEdit (user input)                   │  │
│  └────────────────────────────────────────────┘  │
└──────────────────┬───────────────────────────────┘
                   │ Qt Signals
                   ↓
┌──────────────────────────────────────────────────┐
│        Aura Chat LLM (Gemini 2.5 Pro)            │
│                                                  │
│  "You are Aura, an AI orchestrator..."          │
│                                                  │
│  Tools Available:                                │
│  ✓ read_project_file(path)                      │
│  ✓ list_project_files(directory)                │
│  ✓ execute_python_session(prompt, dir) ← NEW!   │
│  ✓ get_git_status()                             │
│  ✓ git_commit(message)                          │
│  ✓ git_push(remote, branch)                     │
│                                                  │
│  Function Calling: Native Gemini API            │
└──────────────────┬───────────────────────────────┘
                   │ Tool Call
                   ↓
┌──────────────────────────────────────────────────┐
│       PythonCoderAgent (Custom Worker)           │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │  Gemini 1.5 Pro (Code Generation Brain)   │ │
│  │  - Generates code from session prompt     │ │
│  │  - Returns JSON plan: {files, commands}   │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  Actions:                                        │
│  • Writes files to working_dir                   │
│  • Runs commands (pytest, git, etc.)             │
│  • Emits Qt signals for progress                 │
│                                                  │
│  Returns: AgentResult (Pydantic)                 │
│    {success, files_created, files_modified,      │
│     commands_run, output_lines, errors}          │
└──────────────────────────────────────────────────┘
```

---

## 🔧 Technical Decisions (v3.0)

### Core Tech Stack (Updated)

**GUI Framework: PySide6 (Qt)**
- ✅ Native performance
- ✅ Cross-platform
- ✅ Rich widgets (HTML support in QTextEdit)
- ✅ Built-in threading (QThread)
- ✅ Signal/slot pattern for async updates

**Orchestration Brain: Gemini 2.5 Pro API + Function Calling**
- ✅ Native function calling (no MCP needed yet)
- ✅ Large context window (1M tokens)
- ✅ Can read files, plan, execute, verify in one flow
- ✅ Structured outputs via JSON mode
- ✅ Agentic - makes decisions autonomously

**Coding Worker: PythonCoderAgent (Custom)**
- ✅ We own the code (no subprocess mysteries)
- ✅ Returns structured data (Pydantic models)
- ✅ Direct Gemini API integration
- ✅ Testable, mockable, controllable
- ✅ Qt signals for live updates
- ❌ No more stdout parsing hell!

### Key Architectural Choices

#### 1. Function Calling Over Subprocess Wrappers

**Old Approach (v2.0):**
```python
# Parse subprocess stdout with regex 😭
process = subprocess.Popen(["gemini", "-p", prompt])
for line in process.stdout:
    if "Creating" in line:  # Fragile!
        file = regex_magic(line)
```

**New Approach (v3.0):**
```python
# Structured tool response 🎉
result = execute_python_session(
    "Create calculator.py",
    "/path/to/project"
)
# result = {
#   "success": True,
#   "files_created": ["calculator.py"],
#   "summary": "Created 1 file"
# }
```

**Why This Is Better:**
- ✅ No regex parsing of stdout
- ✅ Type-safe structured data
- ✅ Proper error handling
- ✅ Testable (mock tool responses)
- ✅ LLM sees results and adapts

#### 2. Aura Chat IS the Orchestrator

**Philosophy:** The orchestrating LLM should be agentic, not just a planner.

**What Aura Chat Can Do:**
1. **Read project context** - sees what files exist
2. **Plan sessions** - decomposes tasks intelligently
3. **Execute code** - calls PythonCoderAgent
4. **Verify results** - reads created files
5. **Iterate** - if something's wrong, fix it
6. **Commit & push** - handles git operations
7. **Explain** - tells user what happened in friendly language

**Example Conversation:**
```
User: "Add error handling to calculator"

Aura: Let me check the current code...
[reads calculator.py]

I see the divide function doesn't handle zero. Let me fix that.
[executes coding session]

Done! Added ZeroDivisionError handling. Want me to commit this?

User: yes

Aura: [commits with message "Add zero division error handling"]
Pushed to GitHub! ✅
```

#### 3. Structured Outputs Everywhere

**Every interaction uses Pydantic models or JSON schemas:**

```python
# SessionContext (input to agent)
@dataclass(frozen=True)
class SessionContext:
    working_dir: Path
    session_prompt: str
    previous_work: Sequence[str]
    project_files: Sequence[str]

# AgentResult (output from agent)
@dataclass(frozen=True)
class AgentResult:
    success: bool
    files_created: Sequence[str]
    files_modified: Sequence[str]
    commands_run: Sequence[str]
    output_lines: Sequence[str]
    errors: Sequence[str]
    duration_seconds: float
```

**Benefits:**
- Type safety at compile time
- No parsing errors
- Easy to test
- Self-documenting
- IDE autocomplete works

#### 4. Qt Signals for Everything

**All async updates use Qt's signal/slot pattern:**
```python
# In PythonCoderAgent
progress_update = Signal(str)  # "Creating calculator.py..."
command_executed = Signal(str)  # "Ran: pytest tests/"

# In MainWindow
agent.progress_update.connect(self.display_output)
```

**Why:**
- ✅ Thread-safe GUI updates
- ✅ Decoupled components
- ✅ Multiple listeners possible
- ✅ Built into Qt (no extra deps)

#### 5. No MCP Yet (But Ready For It)

**Current:** Native Gemini function calling  
**Future:** Can add MCP when needed for:
- External tool integrations (VS Code, etc.)
- Agent marketplace
- Cross-LLM compatibility

**For now:** Native function calling is simpler and works great!

---

## 🎨 Design Principles (Unchanged)

### What Makes Aura's GUI "Slick"

**Typography First**
- JetBrains Mono everywhere
- Proper size hierarchy (12px body, 14px headers)
- Generous line height (1.6)

**Color Palette (Dark Theme)**
```
Background:  #1e1e1e
Surface:     #2c2c2c
Text:        #e0e0e0
Accent:      #64B5F6 (blue)
Success:     #66BB6A (green)
Warning:     #FFB74D (orange)
Agent:       #FFD27F (gold)
```

**Real-time Feedback**
- Every progress update streams immediately
- Smooth auto-scroll
- Progress indicators (Session 2/4)
- Time elapsed counters

**Minimal Chrome**
- Focus on content
- Input box at bottom (ChatGPT style)
- Everything else is output

---

## 📋 The Session Planning Prompt (Updated)

### How Aura's Brain Works Now

Aura Chat uses its tools to gather context, then plans:

```python
# Aura Chat workflow
1. list_project_files() → see what exists
2. read_project_file() → understand key files
3. Plan 3-7 sessions internally
4. Display plan to user for approval
5. For each session:
   - execute_python_session()
   - read_project_file() to verify
   - Update context for next session
6. git_commit() when done
7. Celebrate! 🎉
```

**Planning Criteria:**
- 3-7 sessions per request
- 10-25 minutes per session
- 2-4 files per session
- No file over 200 lines
- Clear dependencies

---

## 🔄 The Session Execution Flow (v3.0)

### Detailed Step-by-Step

**1. User Submits Request**
```
Input: "Build a REST API for a todo app with user auth"
```

**2. Aura Chat Gathers Context**
```
🤖 Aura Chat thinks:
- Calls list_project_files(".") → sees empty project
- Internally plans 4 sessions
```

**3. Display Plan to User**
```
📋 I've planned 4 focused sessions:

Session 1: User Authentication Model (~12 min)
  • models/user.py - User class with password hashing
  • utils/jwt.py - JWT token utilities

Session 2: Auth Endpoints (~10 min)  
  • routes/auth.py - login/logout/register routes

Session 3: Todo CRUD Operations (~15 min)
  • models/todo.py - Todo model
  • routes/todo.py - CRUD endpoints

Session 4: Testing & Integration (~8 min)
  • tests/ - Pytest suite

Total: ~45 minutes

Type 'start' when ready! 🚀
```

**4. User Approves → Aura Executes Session 1**
```
User: start

Aura: ▶️ Session 1/4: User Authentication Model

[calls execute_python_session with prompt]
[PythonCoderAgent generates code]
[returns structured result]

✅ Created 2 files:
  • models/user.py (87 lines)
  • utils/jwt.py (45 lines)

[reads models/user.py to understand what was created]
```

**5. Context Passed to Session 2**
```
Aura: ▶️ Session 2/4: Auth Endpoints

[calls execute_python_session with context:]
"Create auth routes using the existing User model at 
models/user.py. Do NOT recreate the User class."

[PythonCoderAgent sees context, doesn't duplicate]

✅ Created 1 file:
  • routes/auth.py (134 lines)
```

**6. All Sessions Complete**
```
Aura: 🎉 All sessions complete! (43m 12s)

Created 12 files totaling 1,247 lines.
Average file size: 104 lines.

Want me to commit this to git?

User: yes

Aura: [commits with descriptive message]
      [pushes to GitHub]
      
✅ Pushed to GitHub! Your todo API is ready!
```

---

## 🛣️ Development Roadmap (Updated)

### Phase 1: Core MVP ✅ COMPLETE
- [x] Basic Qt GUI
- [x] Single session execution
- [x] Dark theme styling
- [x] Real-time output streaming

### Phase 2: Multi-Session Orchestration ✅ COMPLETE
- [x] Gemini planning integration
- [x] Sequential session execution
- [x] Context passing
- [x] Progress display

### Phase 3: Native Agent Architecture ✅ COMPLETE (v3.0)
- [x] PythonCoderAgent built
- [x] Function calling integration
- [x] Structured outputs
- [x] Git tool integration
- [x] File reading tools

### Phase 4: Polish & Refinement 🚧 IN PROGRESS
- [ ] Wire agent into orchestrator
- [ ] Full end-to-end testing
- [ ] Error recovery
- [ ] Session name quality improvements
- [ ] Better prompt engineering

### Phase 5: Power Features (Next)
- [ ] Edit session plans before execution
- [ ] Pause/resume sessions
- [ ] Session history viewer
- [ ] Templates for common architectures
- [ ] Custom planning prompts
- [ ] Auto-commit toggle

### Future: Advanced Features
- [ ] Domain-specific agents (Airtable, SQL, Hex)
- [ ] MCP integration for external tools
- [ ] Watch mode (auto-run on changes)
- [ ] Session branching (try different approaches)
- [ ] Cost tracking
- [ ] Plugin system

---

## 🔬 Technical Deep Dives

### How PythonCoderAgent Works

```python
class PythonCoderAgent(QObject):
    def execute_session(self, context: SessionContext) -> AgentResult:
        # 1. Build prompt with context
        prompt = self._build_prompt(context)
        
        # 2. Call Gemini to get code plan (JSON)
        plan = self._request_plan(prompt)  # Returns: {files: [...], commands: [...]}
        
        # 3. Parse plan into file operations
        operations = self._parse_file_operations(plan['files'])
        
        # 4. Write files to disk
        for op in operations:
            op.path.write_text(op.content)
            self.progress_update.emit(f"✅ Created {op.path}")
        
        # 5. Run validation commands
        for cmd in plan.get('commands', []):
            result = subprocess.run(cmd, ...)
            self.command_executed.emit(f"Ran: {cmd}")
        
        # 6. Return structured results
        return AgentResult(
            success=True,
            files_created=[...],
            duration_seconds=elapsed
        )
```

### How Function Calling Works

```python
# In chat_service.py

# 1. Define tool as Python function
def execute_python_session(session_prompt: str, working_directory: str) -> dict:
    agent = PythonCoderAgent(api_key=GEMINI_API_KEY)
    result = agent.execute_session(SessionContext(...))
    return {
        "success": result.success,
        "files_created": result.files_created,
        # ... LLM can understand this
    }

# 2. Register with Gemini
model = genai.GenerativeModel(
    "gemini-2.5-pro",
    tools=[
        read_project_file,
        execute_python_session,  # ← Registered!
        git_commit,
    ]
)

# 3. Gemini decides when to call
response = model.generate_content("Build a calculator")
# Gemini: "I should call execute_python_session(...)"

# 4. We execute the tool and return result
tool_result = execute_python_session(...)

# 5. Gemini sees result and responds to user
# Gemini: "✅ Created calculator.py! Here's what I built..."
```

---

## 📊 Architecture Comparison

### v2.0 (Subprocess) vs v3.0 (Native Agent)

| Aspect | v2.0 Subprocess | v3.0 Native Agent |
|--------|----------------|-------------------|
| **Execution** | subprocess.Popen | Direct Python API call |
| **Output** | stdout text stream | Structured Pydantic models |
| **Parsing** | Regex on stdout 😭 | No parsing needed 🎉 |
| **Context** | GEMINI.md files | Function parameters |
| **Errors** | Parse stderr | Exception handling |
| **Testing** | Mock subprocess | Mock API responses |
| **Control** | Limited (CLI interface) | Full (we own the code) |
| **Updates** | Real-time stdout | Qt signals |
| **Reliability** | Fragile (CLI changes break us) | Robust (we control it) |

---

## 🎯 Success Metrics

### What "Good" Looks Like

**Code Quality:**
- Average file size: 80-150 lines
- No files over 200 lines
- Clear separation of concerns
- Minimal code duplication

**User Experience:**
- Request to first session: <15 seconds
- Session execution: 10-25 minutes each
- Real-time feedback streaming
- Natural language interaction

**Reliability:**
- Session success rate: >90%
- No crashes on edge cases
- Graceful error handling
- Clear error messages

**Performance:**
- GUI stays responsive
- Streaming updates <100ms latency
- API calls complete in <5 seconds
- File operations instant

---

## 🐛 Known Issues & Future Fixes

### Current Known Issues
1. **Session name quality** - Sometimes vague names like "Core Implementation"
2. **Context population** - previous_work and project_files not fully wired
3. **Command security** - Using shell=True with LLM commands (needs review)
4. **Integration gap** - PythonCoderAgent not yet used by Orchestrator

### Planned Fixes (Priority Order)
1. ✅ Enforce JSON responses from Gemini (use structured output)
2. ✅ Add error handling in execute_python_session
3. ✅ Populate project_files context
4. ✅ Update system prompt to match new architecture
5. 🚧 Wire PythonCoderAgent into Orchestrator
6. 🚧 Pass session context properly
7. 🚧 Improve planning prompt for better session names

---

## 💡 Design Philosophy

### Why This Architecture?

**Control Inputs, Not Outputs**
- Focus on task decomposition (input) vs post-processing (output)
- Break problems into small, focused tasks
- Each task produces clean code naturally

**Agentic Orchestration**
- Let the LLM make decisions
- Tools give it capabilities
- Structured outputs ensure reliability
- Human stays in the loop for approval

**Clean Code by Design**
- Constraints force clean code (file size limits, focused sessions)
- Context passing prevents duplication
- Sequential execution builds incrementally

**Developer Experience First**
- Real-time feedback (see what's happening)
- Natural language interaction
- Professional code quality
- "Hide in plain sight" - looks human-written

---

## 📚 References & Resources

### Key Technologies
- **PySide6** - https://doc.qt.io/qtforpython/
- **Gemini API** - https://ai.google.dev/docs
- **Function Calling** - https://ai.google.dev/docs/function_calling

### Related Patterns
- **ReAct** (Reason + Act) - Yao et al. 2022
- **Planning agents** - AWS agentic patterns
- **Tool use** - LangChain agent patterns
- **Multi-agent orchestration** - Microsoft Agent Framework

---

**Last Reviewed:** 2025-01-06  
**Next Review:** After Phase 4 completion

---

## End of Design Doc v3.0
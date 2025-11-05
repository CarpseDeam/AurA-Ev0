Aura - Living Design Document
Version: 2.0 (Simplified Architecture)
Last Updated: 2025-01-05


Implementation Snapshot (2025-11-05)
- PySide6 desktop shell in `src/aura/ui/main_window.py` with dark theme, status bar, toolbar, and live output stream
- Background `AgentRunner` QThread executes Gemini CLI with real-time stdout capture and status signaling (`src/aura/services/agent_runner.py`)
- Configuration, project scanning, and entry-point plumbing live under `src/aura/` with `requirements.txt` pinning PySide6 (>=6.7,<7)
- Prompt wrapper now injects working-directory context and validates Gemini CLI availability before each run

Session 1: "Create User model ONLY"
→ models/user.py (87 lines, clean)
Session 2: "Create login/logout routes ONLY, using existing User model"
→ routes/auth.py (134 lines, clean)
Session 3: "Create JWT middleware ONLY"
→ middleware/auth.py (56 lines, clean)

Each session produces clean code because the task is inherently small.

🧠 What Aura Actually Does
The Three-Step Process
1. USER REQUEST
   ↓
2. AURA BRAIN (Gemini 2.5 Pro)
   Analyzes request
   Decomposes into 3-7 focused sessions
   Each session = one module/feature
   ↓
3. SESSIONS RUN SEQUENTIALLY
   Session 1: Creates User model → passes to Session 2
   Session 2: Uses User model, creates routes → passes to Session 3
   Session 3: Uses routes, creates tests
   ↓
4. CLEAN CODEBASE
   Small files, clear separation, maintainable
Key Innovation: Context Passing
Each session sees:

✅ What files were created in previous sessions
✅ Summary of what's been done
✅ File tree structure
❌ NOT the full code (too much context)

Why this works:

AI knows what exists, doesn't recreate
AI builds on previous work
Each task remains focused
Architecture stays clean


👤 The User Experience
What It Feels Like to Use Aura
Initial Request:
User: Build a blog with authentication

Aura: 🧠 Thinking...

I've decomposed this into 4 focused sessions:

📋 Session 1: User Authentication System (Est. 15 min)
   Creates: models/user.py, utils/password.py
   
📋 Session 2: Auth Routes & Middleware (Est. 12 min)
   Creates: routes/auth.py, middleware/jwt.py
   
📋 Session 3: Blog CRUD Operations (Est. 18 min)
   Creates: models/post.py, routes/blog.py
   
📋 Session 4: Frontend & Integration (Est. 10 min)
   Creates: templates/, static/js/blog.js

Total: ~55 minutes

[Start Building]
During Execution:
✨ Session 1/4: User Authentication System

🤖 Gemini CLI
Using: 1 GEMINI.md file

> Using tool: write_file
> Writing to: models/user.py
✓ Wrote 87 lines

> Using tool: write_file  
> Writing to: utils/password.py
✓ Wrote 45 lines

> Using tool: run_command
> Running: pytest tests/test_user.py
✓ All 8 tests passed


───────────────────────────────────────


┌─────────────────────────────────────┐
│         Main Window (Qt)            │
│  ┌───────────────────────────────┐  │
│  │   Output Display              │  │
│  │   (QTextEdit + formatting)    │  │
│  │                               │  │
│  │   Real-time agent output      │  │
│  │   Colored, formatted          │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │   Input Box                   │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         ↓ Qt Signals
┌─────────────────────────────────────┐
│      Orchestrator (Python)          │
│  ┌─────────────────────────────┐   │
│  │  Session Planner            │   │
│  │  (Gemini 2.5 Pro API)       │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  Session Runner             │   │
│  │  (subprocess + threading)   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         ↓ subprocess.Popen
┌─────────────────────────────────────┐
│      Gemini CLI (Google)            │
│   Executes coding tasks             │
│   Streams verbose output            │
│   Creates files in project          │
└─────────────────────────────────────┘
Key Components:

Main Window (Qt - ~150 lines)

Dark theme, JetBrains Mono font
QTextEdit for output (formatted HTML)
QLineEdit for input
Clean, minimal design


Session Planner (~50 lines)

Calls Gemini 2.5 Pro API
Sends user request + planning prompt
Parses response into session list
Each session: name, estimated time, files to create


Session Runner (~80 lines)

Runs in QThread (keeps UI responsive)
Uses subprocess.Popen to run Gemini CLI
Captures stdout line-by-line
Emits Qt signals for each line
Parses Gemini's verbose output for structure


Output Formatter (~40 lines)

Parses Gemini CLI output
Detects: file creation, tool usage, status
Adds colors, icons, formatting
Creates clean HTML for display



Total Core: ~320 lines

🎨 Design Principles
What Makes Aura's GUI "Slick"

Typography First

JetBrains Mono everywhere
Proper size hierarchy (12px body, 14px headers)
Generous line height (1.6)


Color Palette (Dark Theme)

   Background:  #1e1e1e
   Surface:     #2c2c2c
   Text:        #e0e0e0
   Accent:      #64B5F6 (blue)
   Success:     #66BB6A (green)
   Warning:     #FFB74D (orange)
   Agent:       #FFD27F (gold)

Spacing & Rhythm

16px base unit
Consistent padding: 16px, 24px, 32px
Breathing room between elements


Real-time Feedback

Every line from Gemini streams immediately
Smooth auto-scroll
Progress indicators (Session 2/4)
Time elapsed counters


No Chrome

Minimal UI
Focus on output
Input box at bottom (like ChatGPT)
Everything else is content




🔧 Technical Decisions
Core Tech Stack
GUI Framework: PySide6 (Qt)

✅ Native performance
✅ Cross-platform (Windows, Mac, Linux)
✅ Rich widgets (QTextEdit handles HTML/colors easily)
✅ Built-in threading (QThread)
✅ Mature, stable

Orchestration Brain: Gemini 2.5 Pro (API)

✅ Best at planning/reasoning
✅ Large context window (1M tokens)
✅ Good at structured output
✅ Fast

Coding Agent: Gemini CLI (subprocess)

✅ Already does file operations
✅ Has project understanding
✅ Streams beautiful output
✅ Built by Google, maintained
✅ --yolo mode for automation


Key Technical Choices
1. Subprocess, Not Embedded Terminal
We run Gemini CLI as subprocess and capture stdout:
pythonprocess = subprocess.Popen(
    ["gemini", "-p", prompt, "--yolo"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=project_dir,
    text=True,
    bufsize=1  # Line buffered
)

for line in process.stdout:
    self.output_signal.emit(line)
Why this works:

✅ Simple (20 lines)
✅ Direct access to output
✅ No shell echo pollution
✅ Easy to parse
✅ Works cross-platform

2. Qt Signals for Threading
Session runs in QThread, emits signals:
pythonclass SessionWorker(QThread):
    output_line = Signal(str)
    session_complete = Signal(dict)
    
    def run(self):
        # Run subprocess
        # Emit lines as they arrive
        self.output_line.emit(line)
Why:

✅ Thread-safe GUI updates
✅ Non-blocking UI
✅ Built into Qt, no extra deps
✅ Clean separation

3. No --output-format json Flag
We do NOT use gemini --output-format json:
bash# ❌ Bad: Suppresses beautiful output
gemini -p "Build auth" --output-format json

# ✅ Good: Streams verbose, colored output
gemini -p "Build auth" --yolo
```

**Result:**
```
Using tool: write_file
Writing to: models/user.py
✓ Wrote 87 lines
Much better UX than parsing JSON.
4. HTML Formatting in QTextEdit
We insert colored HTML directly:
pythondef add_output(self, text, color="#FFD27F"):
    html = f'<span style="color: {color};">{text}</span><br>'
    self.output.insertHtml(html)
    self.output.ensureCursorVisible()
```

**Why:**
- ✅ Easy colors/formatting
- ✅ Rich text support
- ✅ Clickable links (future)
- ✅ No custom rendering needed

---

## 📋 The Session Planning Prompt

### How Aura's Brain Works

**Input:** User request  
**Output:** Structured session plan

**The Planning Prompt:**
```
You are Aura's architect. Your job is to decompose coding requests into 3-7 focused sessions.

Rules:
1. Each session creates ONE module or feature
2. Sessions run sequentially (can build on previous work)
3. Estimate 10-25 minutes per session
4. Each session should create 2-4 files max
5. No session should produce files over 200 lines
6. Each session has clear deliverables

User request: {user_request}

Output format:
<sessions>
  <session>
    <name>User Authentication System</name>
    <estimated_minutes>15</estimated_minutes>
    <description>Create User model with password hashing and JWT utilities</description>
    <deliverables>
      - models/user.py (User class with validation)
      - utils/password.py (bcrypt hashing)
      - utils/jwt.py (token generation/validation)
    </deliverables>
    <dependencies>none</dependencies>
  </session>
  <session>
    <name>Auth Routes</name>
    <estimated_minutes>12</estimated_minutes>
    <description>Implement login/logout/register endpoints</description>
    <deliverables>
      - routes/auth.py (FastAPI routes)
      - middleware/auth.py (JWT middleware)
    </deliverables>
    <dependencies>Session 1 (needs User model)</dependencies>
  </session>
</sessions>
```

**Why XML not JSON:**
- ✅ LLMs handle XML more reliably
- ✅ Easier to parse with simple regex
- ✅ More flexible structure

---

## 🔄 The Session Execution Flow

### Detailed Step-by-Step

**1. User Submits Request**
```
Input: "Build a REST API for a todo app with user auth"
```

**2. Aura Brain Plans (10 seconds)**
```
🧠 Analyzing request...

Gemini 2.5 Pro receives:
- User request
- Planning prompt
- Example session structure

Returns:
- 4 sessions
- Estimated 45 minutes total
- File tree structure
```

**3. Display Plan to User**
```
📋 Execution Plan

Session 1: User & Auth Models (12 min)
Session 2: Auth Endpoints (10 min)  
Session 3: Todo CRUD Operations (15 min)
Session 4: Testing & Integration (8 min)

[Start] [Cancel]
```

**4. Execute Session 1**
```
Create GEMINI.md for Session 1:
- Task: "Create User model and auth utilities ONLY"
- Context: "Starting fresh project"
- Files to create: models/user.py, utils/password.py
- Success criteria: User class with tests

Run: gemini -p "Follow GEMINI.md" --yolo
Working directory: ~/projects/todo-app/

Stream output:
✨ Gemini CLI
> Using tool: write_file
> Writing to: models/user.py
[Shows full Gemini CLI output in real-time]
```

**5. Extract Session Results**
```
Parse output / check filesystem:
- models/user.py created ✓
- utils/password.py created ✓
- tests/test_user.py created ✓

Generate summary:
{
  "files_created": ["models/user.py", "utils/password.py"],
  "lines_written": 145,
  "tests_passed": 8,
  "duration_seconds": 734
}
```

**6. Prepare Session 2 Context**
```
Create GEMINI.md for Session 2:
- Task: "Create auth endpoints ONLY"
- Context:
  * Session 1 created User model at models/user.py
  * Available: User class with password hashing
  * File tree: [shows current structure]
- Files to create: routes/auth.py
- Must use existing User model, don't recreate
```

**7. Execute Session 2... (repeat)**

**8. All Sessions Complete**
```
✅ All 4 sessions complete! (43m 12s)

Created 12 files:
  models/ (2 files)
  routes/ (3 files)
  utils/ (2 files)
  tests/ (5 files)

Total: 1,247 lines
Average file size: 104 lines

[Open Project] [Run Tests] [New Task]




🛣️ Development Roadmap
Phase 1: Core MVP (Week 1-2)
Goal: Prove the orchestration works
Features:

 Basic Qt GUI (input box, output display)
 Run single Gemini CLI session
 Stream output with basic colors
 Dark theme styling

Deliverable:
Can type "build calculator" and watch Gemini create it in real-time.
Lines of code: ~300

Phase 2: Multi-Session (Week 3-4)
Goal: The core innovation works
Features:

 Call Gemini 2.5 Pro to plan sessions
 Execute multiple sessions sequentially
 Pass context between sessions
 Display progress (Session 2/4)
 Parse session results

Deliverable:
Can type "build blog with auth" and Aura breaks it into 4 sessions, runs each, passes context forward.
Lines of code: +200 (total ~500)

Phase 3: Polish & Usability (Week 5-6)
Goal: Actually pleasant to use
Features:

 Better output parsing (detect files, tests, errors)
 File tree view (show what was created)
 Project directory selector
 Settings (API keys, default model)
 Error recovery (retry failed sessions)
 Export session logs

Deliverable:
Polished app that feels good to use daily.
Lines of code: +150 (total ~650)

Phase 4: Power Features (Month 2)
Goal: Actually useful for real projects
Features:

 Edit/refine session plans before execution
 Pause/resume sessions
 Session history (see past builds)
 Templates (common architectures)
 Custom planning prompts
 Integration with git (auto-commit after each session)

Lines of code: +200 (total ~850)

Future: Advanced (Month 3+)
Potential features:

 Watch mode (auto-run sessions when requirements change)
 Session branching (try different approaches)
 Cost tracking (API usage)
 Model selection per session (fast model for simple, smart for complex)
 MCP integration (project-aware AI)
 Plugin system for custom agents


Last reviewed: 2025-01-05


End of Design Doc

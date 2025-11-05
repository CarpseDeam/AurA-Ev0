Aura - Living Design Document
Version: 2.0 (Simplified Architecture)
Last Updated: 2025-01-05
Status: 🟡 Rebuilding from lessons learned
Author: Solo developer, building for themselves first

🎯 The Vision (One Sentence)
Aura is an AI orchestrator that breaks large coding requests into small, focused sessions so AI agents produce clean, maintainable code instead of 700-line god files.

🔥 The Problem
What's Broken with AI Coding Tools Today
Every AI coding tool has the same fatal flaw:
When you ask an AI to "build a blog with authentication," it tries to do everything in one shot:

Creates massive 700+ line files
Mixes concerns (models + routes + services in one file)
Produces god objects that violate SOLID principles
Generates code that works but is unmaintainable

Why this happens:

AI agents are given tasks that are too large
They lack architectural discipline
They try to solve everything at once
No one is constraining the INPUT - everyone tries to fix the OUTPUT

Existing solutions don't work:

❌ Adding linting rules → ignored by AI
❌ Post-processing fixes → too late, architecture is already broken
❌ Yelling in prompts → doesn't scale, gets ignored
❌ Human review every step → defeats the purpose of automation


💡 The Core Insight
Don't constrain outputs. Control inputs.
The breakthrough:
If you give an AI a small enough task, it naturally produces small, clean files.
Example:
❌ Bad (one big task):

"Build a blog system with auth"
→ AI creates auth_system.py (734 lines)
→ Mixes models, routes, middleware, validation
→ Unmaintainable mess

✅ Good (decomposed into focused tasks):

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

✅ Session 1 complete (14m 23s)
   Created: models/user.py, utils/password.py
   Tests: ✓ 8 passed

───────────────────────────────────────

✨ Session 2/4: Auth Routes & Middleware
[Real-time Gemini CLI output streams here...]
The Magic Moment:
User watches AI build their project module by module, with each piece staying clean and focused. It's like watching a senior engineer implement a well-planned architecture.

🏗️ Technical Architecture (Simplified v2)
What We're NOT Building
Based on painful lessons learned:
❌ Embedded terminal widget (xterm.js + WebSocket)

Too complex (500+ lines)
Fighting shell echo
PTY management hell
Doesn't add real value

❌ Complex event bus

Over-engineered for simple needs
Qt signals work fine

❌ Conversation persistence (for now)

Adds complexity
Not core to value prop
Can add later if needed

What We ARE Building
Aura Core Architecture (v2 - Simplified)

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

Alternative considered: Electron

❌ Massive bundle size
❌ Memory hog
❌ Feels sluggish
❌ Overkill for our needs

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

Alternative considered: Custom agent implementation

❌ Reinventing wheel
❌ More maintenance
❌ Harder to keep updated
❌ Gemini CLI already solves this

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

🎯 Success Metrics
What Does "Success" Look Like?
Code Quality Metrics:

✅ Average file size: 80-150 lines
✅ No files over 250 lines
✅ Single Responsibility: Each file does one thing
✅ Proper separation: models/, routes/, utils/, tests/
✅ Tests included automatically

User Experience:

✅ Real-time progress visible
✅ Estimated time accurate (±20%)
✅ Clear what's happening at each step
✅ Can watch it work (satisfying)
✅ Produces actually maintainable code

Technical:

✅ 90%+ success rate on sessions
✅ Context passing works (no duplicate code)
✅ Each session completes in estimated time
✅ Errors are recoverable


🚫 Anti-Goals (What We're NOT Building)
Scope Boundaries
❌ Not a code editor

Users use their own IDE
We just orchestrate the AI
Output: project directory with files

❌ Not a deployment tool

No CI/CD integration (for now)
No hosting/servers
Just generates code locally

❌ Not a collaboration platform

Single-user for v1
No teams, sharing, permissions
Solo dev workflow only

❌ Not a low-code builder

No drag-and-drop
No visual builders
Text prompts only (like talking to senior dev)

❌ Not trying to replace developers

Tool for developers
Speeds up boilerplate
Human still reviews, refines, extends


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


💰 Business Model (If This Gets Big)
Current: Free, open-source, personal tool
If it takes off:
Option 1: Freemium Desktop App

Free: Basic orchestration, limited sessions/day
Pro ($20/mo): Unlimited, advanced features, priority support
Enterprise: Team features, on-prem

Option 2: API/SDK

Let developers integrate Aura orchestration into their tools
Usage-based pricing

Option 3: Hosted Service

Web version, no install needed
Subscription-based

Not deciding now. Build the tool first, see if people want it.

🧪 Key Assumptions to Validate
What Might Be Wrong
Assumption 1: Multi-session actually produces better code

Risk: Maybe one big session with good prompt is just as good
Test: Build same project both ways, compare quality
Fallback: If wrong, pivot to "better prompts" tool

Assumption 2: Users want to watch it work

Risk: Maybe they just want results fast, don't care about process
Test: Usage metrics, feedback
Fallback: Add "fast mode" that hides details

Assumption 3: Context passing works

Risk: Maybe AI still creates duplicate code or ignores previous work
Test: Monitor for duplicates, test with complex projects
Fallback: More explicit constraints in prompts

Assumption 4: Gemini CLI is reliable enough

Risk: Maybe it fails too often, needs babysitting
Test: Run 100 sessions, measure success rate
Fallback: Add fallback to direct API calls

Assumption 5: Planning with LLM is consistent

Risk: Maybe it plans differently each time, unpredictable
Test: Same request 10 times, compare plans
Fallback: Use few-shot examples, stricter output format


📚 Learning & Iteration
Lessons from V1 (Failed Attempt)
What Went Wrong:

Over-engineered GUI - Embedded terminal was unnecessary complexity
Fighting tools - Tried to hide PowerShell, when we should've just used subprocess
Premature features - Built conversation history before core worked
Lost focus - Got distracted by UI polish instead of orchestration

What We Learned:

Simple is better - Subprocess + QThread beats WebSocket + PTY
Focus on value - Orchestration is the innovation, not the terminal
Build incrementally - Prove core works before adding features
Use existing tools - Gemini CLI already does file operations

Applying to V2:

✅ Start with minimal GUI (200 lines)
✅ Prove orchestration works FIRST
✅ Add features only when needed
✅ Embrace simplicity


🎤 Pitch (How to Explain Aura)
To a Developer

"You know how AI coding tools give you giant 700-line files that are impossible to maintain? Aura fixes that. It breaks your request into focused sessions - like 'build user model', then 'build auth routes', then 'build tests'. Each session produces small, clean files because the task is small enough. It's like having a senior architect plan your project, then junior devs implement each piece."

To a Non-Technical Person

"AI coding tools are like asking a junior developer to build your entire app at once - they make a mess. Aura is like having a senior developer break the work into small, manageable pieces, then supervising multiple AI coders as they each build one piece. The result is clean, organized code instead of spaghetti."

To an Investor (If This Becomes a Thing)

"We're solving the #1 problem with AI-generated code: it's unmaintainable. Every AI tool produces monolithic files because they give the AI tasks that are too large. We're the only tool that decomposes requests into micro-tasks before execution. This isn't a prompt engineering trick - it's architectural discipline baked into the workflow. Early users report 90% reduction in refactoring time. TAM: 20M+ developers worldwide using AI coding tools. GTM: Freemium desktop app, virality through demo videos showing before/after code quality."


✅ Definition of Done (When is Aura "Real"?)
V1.0 Success Criteria
It works when:

✅ I can type "build todo API with auth"
✅ Aura plans 3-5 sessions automatically
✅ Each session runs, creates files, passes context forward
✅ Resulting code has files <200 lines each
✅ Code actually runs (tests pass)
✅ I can use this for my own projects
✅ Process takes <1 hour for medium project
✅ Success rate >80% (doesn't break often)

Personal validation:

Would I use this every day?
Does it save me real time?
Is the code it produces actually good?
Do I trust it enough to use on real projects?

If yes to all → Ship it. Show others.

🤝 Contributing (Future - If Open Source)
Not open source yet. Building in private until it actually works.
If/when we open source:

Clear contribution guidelines
Focus on quality over quantity
No feature creep - stick to core vision
Prioritize simplicity and maintainability


📖 Appendix: Why This Matters
The Bigger Picture
AI coding tools are at an inflection point.
Everyone's adding "AI coding assistants" to their editors. But they all have the same problem: they generate code that works but isn't maintainable.
This is limiting AI adoption. Developers try these tools, see the spaghetti code, and go back to writing manually.
Aura is a bet that orchestration matters more than the underlying model.
It's not about having the best AI. It's about:

Giving the AI the right-sized tasks
Passing context intelligently
Architectural discipline baked into the workflow

If this works, it changes what's possible with AI coding.
Instead of "AI writes first draft, human refactors" → "AI produces maintainable code from the start"
That's worth building.

🔄 Living Document Status
This document should evolve.
As we build and learn:

Update assumptions that prove wrong
Add new insights
Remove what doesn't matter
Keep it true to the vision

Last reviewed: 2025-01-05
Next review: After MVP ships

End of Design Doc
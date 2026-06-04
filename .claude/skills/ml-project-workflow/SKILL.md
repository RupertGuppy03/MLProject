---
name: ml-project-workflow
description: Use when starting any Premier League Match Predictor user story. Walks through the per-story workflow from claude.md — read the story, plan with the user, execute, test, check DoD, mark done, propose a commit message.
---

# ML Project Workflow

For the Premier League Match Predictor. Apply when picking up any Sprint 1–4 user story.

## Source of truth
- `claude.md` — project rules and conventions
- `references/Project_User_Stories.md` — master story library
- `references/sprint{1,2,3,4}.md` — per-sprint quick lookup

The Acceptance Tests + Definition of Done inside each story are the contract. Don't invent extra scope.

## Steps

1. **Read the story.** Open both the master file and the per-sprint file. Quote the Acceptance Tests + DoD back to the user so we agree on scope before designing anything.

2. **Plan.** Enter plan mode. Propose file paths, function signatures, key design decisions, and the test list. Use AskUserQuestion for any genuine ambiguity (locked parameters, edge cases, file conventions). Don't write code until the user exits plan mode with approval.

3. **Execute.** Write the feature + its tests. Stick to the locked schema, the no-leakage rule, snake_case, and the style of the existing codebase. Keep comments brief.

4. **Summarise.** One short paragraph: each file created/modified and what it does. No long explanations.

5. **Tests.** Ask the user to run `pytest`. Only run it yourself if they explicitly say so. If anything fails, debug together — never claim a test passes on your own.

6. **DoD check.** Once tests are green, walk every bullet of the story's Definition of Done. Flag anything missing (artifacts not produced, schema not updated, downstream file untouched). Resolve before declaring done.

7. **Mark done + propose a commit.** Once the user confirms they're happy:
   - Update the story status in `references/sprint{N}.md` (table row + section header: `To Do` → `Done`).
   - Append `— **DONE**` to the Labels line in `references/Project_User_Stories.md`, matching the format of the other completed stories.
   - Propose a short, sentence-case commit message matching the style in `git log`. Do NOT run git — the user commits.

## Rules to enforce
- **User controls git.** Read-only git commands are fine; never commit, push, or reset.
- **One story at a time.** Don't drift into the next one before this one is committed and marked done.
- **No leakage.** Rolling features use `shift(1)`, Elo updates after snapshot, backtest train data strictly before test data.
- **Schema is locked.** Any new feature column must update `artifacts/feature_schema.json` and the inference path together.
- **Random Forest is the main model.** Not XGBoost, not HistGradientBoosting.

-- Adds the human-review lock/pending workflow on top of the existing
-- single human_feedback_* columns on investigations. ADDITIVE ONLY —
-- every statement is guarded, safe to re-run, doesn't touch existing data.
--
-- Why this exists: today, submitting feedback twice on the same
-- investigation silently overwrites the first reviewer's decision, and
-- there is nothing stopping a reviewer from confirming a cause the
-- evidence doesn't actually support. This adds:
--   - evidence_map: the per-cause evidence-strength grading, persisted so
--     it can be checked at review time (previously only existed in the
--     live response right after an investigation ran, never saved).
--   - review_status: 'locked' (final, fed to ML training) or
--     'pending_admin_approval' (saved, not yet final, not yet trained).
--   - human_feedback_role: which profile ('user' or 'admin') submitted
--     the currently-recorded human_feedback_* decision.
--   - pending_reason: why a review didn't lock immediately.
--   - original_proposed_cause: preserves a User's original proposal when
--     an Admin later overrides it, so the audit trail isn't lost.
--   - admin_decision / admin_reviewer / admin_decided_at: records how an
--     Admin resolved a pending item (approved as-is / overridden /
--     rejected), independent of whatever human_feedback_* ends up as.

alter table public.investigations
  add column if not exists evidence_map jsonb not null default '{}'::jsonb;

alter table public.investigations
  add column if not exists review_status text
    check (review_status is null or review_status in ('locked', 'pending_admin_approval'));

alter table public.investigations
  add column if not exists human_feedback_role text
    check (human_feedback_role is null or human_feedback_role in ('user', 'admin'));

alter table public.investigations
  add column if not exists pending_reason text;

alter table public.investigations
  add column if not exists original_proposed_cause text;

alter table public.investigations
  add column if not exists admin_decision text
    check (admin_decision is null or admin_decision in ('approved', 'overridden', 'rejected'));

alter table public.investigations
  add column if not exists admin_reviewer text;

alter table public.investigations
  add column if not exists admin_decided_at timestamptz;

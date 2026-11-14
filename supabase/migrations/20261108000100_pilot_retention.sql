-- Retention, as stated in the privacy policy (/privacy, "How long we keep it"):
--   alert inbox + delivery records: 180 days;
--   feedback + trip logs: two years after the person's last sign-in (or account deletion, which cascades).
-- Server only: callable by the service role / postgres (pg_cron), never by signed-in people.
create or replace function public.prune_pilot_data() returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  n_alerts int;
  n_sent int;
  n_feedback int;
  n_trips int;
begin
  delete from public.alerts where created_at < now() - interval '180 days';
  get diagnostics n_alerts = row_count;
  delete from public.notifications_sent where sent_at < now() - interval '180 days';
  get diagnostics n_sent = row_count;
  delete from public.feedback f using auth.users u
    where f.user_id = u.id and coalesce(u.last_sign_in_at, u.created_at) < now() - interval '2 years';
  get diagnostics n_feedback = row_count;
  delete from public.trip_logs t using auth.users u
    where t.user_id = u.id and coalesce(u.last_sign_in_at, u.created_at) < now() - interval '2 years';
  get diagnostics n_trips = row_count;
  return jsonb_build_object('alerts', n_alerts, 'notifications_sent', n_sent, 'feedback', n_feedback, 'trip_logs', n_trips);
end;
$$;
revoke all on function public.prune_pilot_data() from public, anon, authenticated;

-- Cloud only (after enabling pg_cron), run once:
-- select cron.schedule('neritic-prune-pilot-data', '40 3 * * *', $cron$ select public.prune_pilot_data(); $cron$);

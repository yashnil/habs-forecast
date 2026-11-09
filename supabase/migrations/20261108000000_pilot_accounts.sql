-- Neritic private pilot: accounts, saved places, alert subscriptions, inbox, idempotent fan-out,
-- feedback, trip logs, consent log and account deletion.
--
-- Rules for every table here:
--   * row-level security is ON;
--   * a signed-in person reaches only their own rows: user_id = (select auth.uid());
--   * the anon role gets nothing;
--   * rows written only by the server (alerts, notifications_sent, alert_fact_state) have no
--     insert/update policy for users: only the service role (which bypasses RLS) writes them, and the
--     service role key lives only on the server (never NEXT_PUBLIC_*).
-- Deleting the auth user (delete_my_account) cascades to every row below.

create extension if not exists pgcrypto with schema extensions;

-- ------------------------------------------------------------------------------------ helpers
create or replace function public.touch_updated_at() returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

-- ------------------------------------------------------------------------------------ profiles
create table public.profiles (
  user_id uuid primary key default auth.uid() references auth.users (id) on delete cascade,
  role text not null default 'commercial_fisher'
    check (role in ('commercial_fisher', 'recreational_fisher', 'charter', 'processor', 'scientist', 'agency', 'other')),
  -- FisheryId values from coastwatch-web/src/pilot/contracts.ts
  fisheries text[] not null default '{}'
    check (fisheries <@ array['dungeness_commercial', 'dungeness_recreational', 'rock_crab', 'salmon_commercial', 'groundfish',
                              'anchovy_sardine', 'market_squid', 'bivalve_shellfish', 'spiny_lobster', 'other']::text[]),
  home_place_id text check (home_place_id is null or home_place_id ~ '^[a-z0-9:-]{1,64}$'),
  vessel_length_class text
    check (vessel_length_class is null or vessel_length_class in ('none', 'under_25ft', '25_40ft', '40_60ft', 'over_60ft')),
  -- a profile exists only for someone who attested they are 13 or older (privacy policy, terms)
  age_13_plus boolean not null check (age_13_plus),
  age_attested_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create trigger profiles_touch before update on public.profiles for each row execute function public.touch_updated_at();

-- ------------------------------------------------------------------------------------ saved places
create table public.saved_places (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  place_id text not null check (place_id ~ '^[a-z0-9:-]{1,64}$'),
  label text check (char_length(label) <= 120),
  created_at timestamptz not null default now(),
  unique (user_id, place_id)
);
create index saved_places_user on public.saved_places (user_id);

-- ------------------------------------------------------------------------------------ alert subscriptions
create table public.alert_subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  place_id text not null check (place_id ~ '^[a-z0-9:-]{1,64}$'),
  triggers text[] not null
    check (cardinality(triggers) > 0
           and triggers <@ array['status_change', 'new_toxin_result', 'ramp_action', 'bloom_risk_jump', 'data_stale']::text[]),
  channel text not null check (channel in ('in_app', 'web_push', 'sms')),
  -- SMS: the number, and the consent record (when, and the exact words the person agreed to)
  sms_phone_e164 text check (sms_phone_e164 is null or sms_phone_e164 ~ '^\+[1-9][0-9]{7,14}$'),
  sms_consent_at timestamptz,
  sms_consent_text text check (sms_consent_text is null or char_length(sms_consent_text) <= 2000),
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, place_id, channel),
  constraint sms_needs_consent check (
    channel <> 'sms' or (sms_phone_e164 is not null and sms_consent_at is not null and sms_consent_text is not null)
  )
);
create index alert_subscriptions_place on public.alert_subscriptions (place_id) where active;
create trigger alert_subscriptions_touch before update on public.alert_subscriptions for each row execute function public.touch_updated_at();

-- ------------------------------------------------------------------------------------ web push endpoints
create table public.push_subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  endpoint text not null unique check (endpoint ~ '^https://' and char_length(endpoint) <= 1000),
  p256dh text not null check (char_length(p256dh) <= 200),
  auth text not null check (char_length(auth) <= 100),
  user_agent text check (char_length(user_agent) <= 300),
  created_at timestamptz not null default now()
);
create index push_subscriptions_user on public.push_subscriptions (user_id);

-- ------------------------------------------------------------------------------------ inbox (server-written)
create table public.alerts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  subscription_id uuid references public.alert_subscriptions (id) on delete set null,
  place_id text not null,
  fact_id text not null,
  fact_version text not null,
  trigger text not null check (trigger in ('status_change', 'new_toxin_result', 'ramp_action', 'bloom_risk_jump', 'data_stale')),
  severity text not null check (severity in ('red', 'amber', 'neutral', 'info')),
  title text not null check (char_length(title) <= 300),
  body text not null check (char_length(body) <= 2000),
  basis text,
  verified boolean,
  source_name text,
  source_url text,
  as_of text,
  created_at timestamptz not null default now(),
  read_at timestamptz
);
create index alerts_user_created on public.alerts (user_id, created_at desc);

-- idempotency: one delivery per subscription x fact x fact version (fan-out may run any number of times)
create table public.notifications_sent (
  subscription_id uuid not null references public.alert_subscriptions (id) on delete cascade,
  fact_id text not null,
  fact_version text not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  channel text not null check (channel in ('in_app', 'web_push', 'sms')),
  sent_at timestamptz not null default now(),
  primary key (subscription_id, fact_id, fact_version)
);

-- the last fact version each subscription was compared against (the fan-out's memory)
create table public.alert_fact_state (
  subscription_id uuid not null references public.alert_subscriptions (id) on delete cascade,
  fact_id text not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  fact_version text not null,
  fact jsonb not null,
  seen_at timestamptz not null default now(),
  primary key (subscription_id, fact_id)
);

-- ------------------------------------------------------------------------------------ feedback + trip logs
create table public.feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  fact_id text not null check (char_length(fact_id) <= 200),
  place_id text check (place_id is null or char_length(place_id) <= 64),
  useful boolean not null,
  note text check (note is null or char_length(note) <= 1000),
  created_at timestamptz not null default now()
);
create index feedback_user on public.feedback (user_id);

create table public.trip_logs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  trip_date date not null,
  place_id text not null check (char_length(place_id) <= 64),
  fishery text check (fishery is null or char_length(fishery) <= 40),
  outcome_band text check (outcome_band is null or outcome_band in ('skipped', 'poor', 'fair', 'good')),
  observed_conditions text check (observed_conditions is null or char_length(observed_conditions) <= 1000),
  recommendation_useful boolean,
  created_at timestamptz not null default now()
);
create index trip_logs_user on public.trip_logs (user_id, trip_date desc);

-- ------------------------------------------------------------------------------------ consent log (append-only)
create table public.consent_log (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  kind text not null check (kind in ('terms', 'privacy', 'age_13_plus', 'sms', 'web_push')),
  granted boolean not null,
  text_shown text not null check (char_length(text_shown) <= 2000),
  document_version text check (document_version is null or char_length(document_version) <= 40),
  created_at timestamptz not null default now()
);
create index consent_log_user on public.consent_log (user_id, created_at desc);

-- ------------------------------------------------------------------------------------ RLS
alter table public.profiles enable row level security;
alter table public.saved_places enable row level security;
alter table public.alert_subscriptions enable row level security;
alter table public.push_subscriptions enable row level security;
alter table public.alerts enable row level security;
alter table public.notifications_sent enable row level security;
alter table public.alert_fact_state enable row level security;
alter table public.feedback enable row level security;
alter table public.trip_logs enable row level security;
alter table public.consent_log enable row level security;

-- the anonymous role never reaches pilot data
revoke all on public.profiles, public.saved_places, public.alert_subscriptions, public.push_subscriptions, public.alerts,
  public.notifications_sent, public.alert_fact_state, public.feedback, public.trip_logs, public.consent_log from anon;

-- own rows, full control: profiles, saved places, subscriptions, push endpoints, feedback, trip logs
create policy profiles_own on public.profiles for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy saved_places_own on public.saved_places for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy alert_subscriptions_own on public.alert_subscriptions for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy push_subscriptions_own on public.push_subscriptions for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy feedback_own on public.feedback for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy trip_logs_own on public.trip_logs for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));

-- inbox: read, mark read, delete own; never insert (the server writes alerts)
create policy alerts_select_own on public.alerts for select to authenticated using (user_id = (select auth.uid()));
create policy alerts_update_own on public.alerts for update to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy alerts_delete_own on public.alerts for delete to authenticated using (user_id = (select auth.uid()));
revoke insert, update on public.alerts from authenticated;
grant update (read_at) on public.alerts to authenticated;

-- delivery records: a person may see what was sent to them; only the server writes
create policy notifications_sent_select_own on public.notifications_sent for select to authenticated
  using (user_id = (select auth.uid()));
revoke insert, update, delete on public.notifications_sent from authenticated;

-- the fan-out's memory is server-only (no policy: authenticated sees nothing)
revoke all on public.alert_fact_state from authenticated;

-- consent log: append and read own; never edit or delete (deleting the account removes it)
create policy consent_log_insert_own on public.consent_log for insert to authenticated with check (user_id = (select auth.uid()));
create policy consent_log_select_own on public.consent_log for select to authenticated using (user_id = (select auth.uid()));
revoke update, delete on public.consent_log from authenticated;

-- ------------------------------------------------------------------------------------ account deletion
-- Deletes the caller's auth user; every table above cascades. Callable only by a signed-in person,
-- and only ever for themselves.
create or replace function public.delete_my_account() returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  me uuid := auth.uid();
begin
  if me is null then
    raise exception 'not signed in' using errcode = '42501';
  end if;
  delete from auth.users where id = me;
end;
$$;
revoke all on function public.delete_my_account() from public, anon;
grant execute on function public.delete_my_account() to authenticated;

-- ------------------------------------------------------------------------------------ schedule (cloud only)
-- After the cloud project exists (docs/coastwatch/pilot/accounts.md), enable pg_cron and pg_net in the
-- dashboard, store the fan-out URL and secret in Vault, and run (once) as the postgres role:
--
-- select vault.create_secret('https://<preview-or-site>/api/alerts/fanout', 'alerts_fanout_url');
-- select vault.create_secret('<same value as ALERTS_FANOUT_SECRET>', 'alerts_fanout_secret');
-- select cron.schedule(
--   'neritic-alerts-fanout',
--   '20 * * * *', -- hourly, 20 min past (the data refresh publishes on the hour)
--   $cron$
--   select net.http_post(
--     url := (select decrypted_secret from vault.decrypted_secrets where name = 'alerts_fanout_url'),
--     headers := jsonb_build_object(
--       'content-type', 'application/json',
--       'authorization', 'Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name = 'alerts_fanout_secret')
--     ),
--     body := '{}'::jsonb,
--     timeout_milliseconds := 30000
--   );
--   $cron$
-- );

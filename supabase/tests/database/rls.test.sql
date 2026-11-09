-- RLS isolation for the Neritic pilot tables. Run: npx supabase@2.121.0 test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(40);

-- two people, A and B
insert into auth.users (id, email, aud, role) values
  ('00000000-0000-0000-0000-00000000000a', 'a@example.test', 'authenticated', 'authenticated'),
  ('00000000-0000-0000-0000-00000000000b', 'b@example.test', 'authenticated', 'authenticated');

-- every pilot table has RLS on
select ok((select relrowsecurity from pg_class where oid = ('public.' || t)::regclass), t || ' has RLS enabled')
from unnest(array['profiles', 'saved_places', 'alert_subscriptions', 'push_subscriptions', 'alerts', 'notifications_sent',
                  'alert_fact_state', 'feedback', 'trip_logs', 'consent_log']) as t;

-- ---------------------------------------------------------------- as A: write own rows
set local role authenticated;
set local request.jwt.claims = '{"sub":"00000000-0000-0000-0000-00000000000a","role":"authenticated"}';

select lives_ok($$ insert into public.profiles (fisheries, home_place_id, age_13_plus) values ('{dungeness_commercial}', 'port-592', true) $$, 'A creates a profile');
select throws_ok($$ insert into public.profiles (user_id, age_13_plus) values ('00000000-0000-0000-0000-00000000000b', true) $$, '42501', null, 'A cannot create a profile for B');
select lives_ok($$ insert into public.saved_places (place_id, label) values ('port-592', 'Moss Landing') $$, 'A saves a place');
select lives_ok($$ insert into public.alert_subscriptions (place_id, triggers, channel) values ('port-592', '{status_change,new_toxin_result}', 'in_app') $$, 'A subscribes in-app');
select throws_ok($$ insert into public.alert_subscriptions (place_id, triggers, channel, sms_phone_e164) values ('port-592', '{status_change}', 'sms', '+14155550100') $$, '23514', null, 'SMS without a consent record is refused');
select lives_ok($$ insert into public.feedback (fact_id, place_id, useful) values ('port-592:status:dungeness_commercial', 'port-592', true) $$, 'A leaves feedback');
select lives_ok($$ insert into public.trip_logs (trip_date, place_id, outcome_band) values ('2026-11-08', 'port-592', 'fair') $$, 'A logs a trip');
select lives_ok($$ insert into public.consent_log (kind, granted, text_shown) values ('age_13_plus', true, 'I am 13 or older.') $$, 'A records consent');
select throws_ok($$ insert into public.alerts (user_id, place_id, fact_id, fact_version, trigger, severity, title, body)
                    values ('00000000-0000-0000-0000-00000000000a', 'port-592', 'f', 'v', 'status_change', 'red', 't', 'b') $$,
                 '42501', null, 'A cannot write inbox alerts (server only)');

-- ---------------------------------------------------------------- the server writes an alert for A
reset role;
insert into public.alerts (user_id, place_id, fact_id, fact_version, trigger, severity, title, body)
values ('00000000-0000-0000-0000-00000000000a', 'port-592', 'port-592:status:dungeness_commercial', 'v1', 'status_change', 'red', 'Dungeness delayed', 'Changed');
insert into public.notifications_sent (subscription_id, fact_id, fact_version, user_id, channel)
select id, 'port-592:status:dungeness_commercial', 'v1', user_id, 'in_app' from public.alert_subscriptions limit 1;
select throws_ok($$ insert into public.notifications_sent (subscription_id, fact_id, fact_version, user_id, channel)
                    select id, 'port-592:status:dungeness_commercial', 'v1', user_id, 'in_app' from public.alert_subscriptions limit 1 $$,
                 '23505', null, 'a second delivery of the same fact version is refused (idempotency key)');

-- ---------------------------------------------------------------- as B: sees none of A's rows
set local role authenticated;
set local request.jwt.claims = '{"sub":"00000000-0000-0000-0000-00000000000b","role":"authenticated"}';

select is((select count(*) from public.profiles)::int, 0, 'B sees no profiles of others');
select is((select count(*) from public.saved_places)::int, 0, 'B sees no saved places of others');
select is((select count(*) from public.alert_subscriptions)::int, 0, 'B sees no subscriptions of others');
select is((select count(*) from public.alerts)::int, 0, 'B sees no alerts of others');
select is((select count(*) from public.notifications_sent)::int, 0, 'B sees no deliveries of others');
select is((select count(*) from public.feedback)::int, 0, 'B sees no feedback of others');
select is((select count(*) from public.trip_logs)::int, 0, 'B sees no trip logs of others');
select is((select count(*) from public.consent_log)::int, 0, 'B sees no consent records of others');
select throws_ok($$ select count(*) from public.alert_fact_state $$, '42501', null, 'the fan-out state is server only');

-- B cannot change or delete A's rows (RLS filters them: zero rows affected)
update public.saved_places set label = 'hijacked';
delete from public.trip_logs;
reset role;
select is((select label from public.saved_places where user_id = '00000000-0000-0000-0000-00000000000a'), 'Moss Landing', 'B could not edit A''s saved place');
select is((select count(*) from public.trip_logs)::int, 1, 'B could not delete A''s trip log');

-- ---------------------------------------------------------------- as A: own inbox, mark read only
set local role authenticated;
set local request.jwt.claims = '{"sub":"00000000-0000-0000-0000-00000000000a","role":"authenticated"}';
select is((select count(*) from public.alerts)::int, 1, 'A sees their own alert');
select lives_ok($$ update public.alerts set read_at = now() $$, 'A marks an alert read');
select throws_ok($$ update public.alerts set title = 'edited' $$, '42501', null, 'A cannot edit an alert''s text');
select throws_ok($$ update public.consent_log set granted = false $$, '42501', null, 'the consent log is append-only');

-- ---------------------------------------------------------------- anon gets nothing
reset role;
set local role anon;
set local request.jwt.claims = '{"role":"anon"}';
select throws_ok($$ select count(*) from public.profiles $$, '42501', null, 'anon cannot read profiles');
select throws_ok($$ select public.delete_my_account() $$, '42501', null, 'anon cannot call delete_my_account');

-- ---------------------------------------------------------------- A deletes their account: everything goes
reset role;
set local role authenticated;
set local request.jwt.claims = '{"sub":"00000000-0000-0000-0000-00000000000a","role":"authenticated"}';
select lives_ok($$ select public.delete_my_account() $$, 'A deletes their account');
reset role;
select is((select count(*) from auth.users where id = '00000000-0000-0000-0000-00000000000a')::int, 0, 'A''s auth user is gone');
select is((select count(*) from public.alerts) + (select count(*) from public.saved_places) + (select count(*) from public.consent_log)
          + (select count(*) from public.notifications_sent) + (select count(*) from public.trip_logs), 0::bigint, 'every row of A''s cascaded away');

select * from finish();
rollback;

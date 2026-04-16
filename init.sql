-- Enable pg_hint_plan extension on the bao database
CREATE EXTENSION IF NOT EXISTS pg_hint_plan;

-- Set up generic schema structure if needed in future
CREATE SCHEMA IF NOT EXISTS public;

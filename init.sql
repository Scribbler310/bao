-- Enable pg_hint_plan extension on the bao database
CREATE EXTENSION IF NOT EXISTS pg_hint_plan;

-- Set up public schema
CREATE SCHEMA IF NOT EXISTS public;

-- Load the IMDB/JOB schema.
-- This file must exist inside the container at /imdb/schematext.sql.
\i /imdb/schematext.sql

-- Load IMDB CSV data.
-- These files must exist inside the container at /imdb/*.csv.
-- The docker-compose.yml mount should map ../imdb on your host to /imdb in the container.

COPY aka_name FROM '/imdb/aka_name.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY aka_title FROM '/imdb/aka_title.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY cast_info FROM '/imdb/cast_info.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY char_name FROM '/imdb/char_name.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY comp_cast_type FROM '/imdb/comp_cast_type.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY company_name FROM '/imdb/company_name.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY company_type FROM '/imdb/company_type.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY complete_cast FROM '/imdb/complete_cast.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY info_type FROM '/imdb/info_type.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY keyword FROM '/imdb/keyword.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY kind_type FROM '/imdb/kind_type.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY link_type FROM '/imdb/link_type.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY movie_companies FROM '/imdb/movie_companies.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY movie_info FROM '/imdb/movie_info.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY movie_info_idx FROM '/imdb/movie_info_idx.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY movie_keyword FROM '/imdb/movie_keyword.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY movie_link FROM '/imdb/movie_link.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY name FROM '/imdb/name.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY person_info FROM '/imdb/person_info.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY role_type FROM '/imdb/role_type.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');
COPY title FROM '/imdb/title.csv' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\');

create index company_id_movie_companies on movie_companies(company_id);
create index company_type_id_movie_companies on movie_companies(company_type_id);
create index info_type_id_movie_info_idx on movie_info_idx(info_type_id);
create index info_type_id_movie_info on movie_info(info_type_id);
create index info_type_id_person_info on person_info(info_type_id);
create index keyword_id_movie_keyword on movie_keyword(keyword_id);
create index kind_id_aka_title on aka_title(kind_id);
create index kind_id_title on title(kind_id);
create index linked_movie_id_movie_link on movie_link(linked_movie_id);
create index link_type_id_movie_link on movie_link(link_type_id);
create index movie_id_aka_title on aka_title(movie_id);
create index movie_id_cast_info on cast_info(movie_id);
create index movie_id_complete_cast on complete_cast(movie_id);
create index movie_id_movie_companies on movie_companies(movie_id);
create index movie_id_movie_info_idx on movie_info_idx(movie_id);
create index movie_id_movie_keyword on movie_keyword(movie_id);
create index movie_id_movie_link on movie_link(movie_id);
create index movie_id_movie_info on movie_info(movie_id);
create index person_id_aka_name on aka_name(person_id);
create index person_id_cast_info on cast_info(person_id);
create index person_id_person_info on person_info(person_id);
create index person_role_id_cast_info on cast_info(person_role_id);
create index role_id_cast_info on cast_info(role_id);

ANALYZE;

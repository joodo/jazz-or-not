/*
将需要的数据挪到新的 Schema 内
*/
CREATE SCHEMA done;
ALTER TABLE musicbrainz.tag SET SCHEMA done;
ALTER TABLE musicbrainz.release_group SET SCHEMA done;
ALTER TABLE musicbrainz.release_group_tag SET SCHEMA done;


/*
建立保存结果的表
*/
CREATE TABLE done.done
(
  mbid uuid,
  name text,
  tags text,
  cover bytea
);


/*
合并数据
*/
-- 如果一个 release_group 对应多个 tag，需要拼接 tag
CREATE AGGREGATE group_concat(anyelement)
(
  sfunc = array_append,
  stype = anyarray,
  initcond = '{}'
);

INSERT INTO done.done(mbid, name, tags)
SELECT
  done.release_group.gid,
  done.release_group.name,
  array_to_string( group_concat (done.tag.name), ',' )
FROM done.release_group
INNER JOIN done.release_group_tag
ON done.release_group.id=done.release_group_tag.release_group
INNER JOIN done.tag
ON done.release_group_tag.tag=done.tag.id
GROUP BY done.release_group.gid, done.release_group.name
ORDER BY random()
;

-- ALTER TABLE done.done ADD COLUMN id SERIAL PRIMARY KEY;

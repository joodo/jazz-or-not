/*
如果磁盘空间不足，可以在运行 process_data.sql 后运行这个，删除不用的数据，释放空间
*/
DROP SCHEMA musicbrainz CASCADE;

DROP TABLE done.release_group;
DROP TABLE done.release_group_tag;
DROP TABLE done.tag;

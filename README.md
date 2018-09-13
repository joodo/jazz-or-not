# Jazz or not 过程记录

判断一张专辑封面的“爵士味儿”

详细过程记录 [见此](https://zhuanlan.zhihu.com/p/44446435)

# 运行

## 从 MusicBrainz 下载专辑信息

从 [MusicBrainz 数据下载](http://ftp.musicbrainz.org/pub/musicbrainz/data/fullexport/) 处，根据 latest 提示进入最新的文件夹，下载 **mbdump.tar.bz2** （包含基本信息）和 **mbdump-derived.tar.bz2** （包含 Tags 风格信息）两个文件。

然后，根据 Github 上项目的 [说明](http://link.zhihu.com/?target=https%3A//github.com/metabrainz/musicbrainz-server/blob/master/INSTALL.md%23creating-the-database) 导入到数据库中。

## 整理数据

运行 `sql/process_data.sql` 整理数据。运行后可删除 `sql/drop_unused_data.sql` 来删除其余不需要的数据从而释放空间。

## 从 Cover Art Archive 获取

安装 [pip](https://pypi.org/project/pip/) 并运行

```
pip install -r python_requirements
```

安装依赖 python 包。运行

```
python crawler/crawler.py
```

开始爬取专辑封面。注意：爬取需要外网环境。

## 导出数据到 recordio 格式

运行

```
python export/exporter.py
```

将数据导出，将会生成 4 个文件：`train.recordio`、`test.recordio`、`verify.recordio` 和 `data.info`。

## 训练

运行

```
python train/train.py
```

即可开始训练。


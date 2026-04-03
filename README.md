# KG_Stock_SR_Django_Center

将原项目中 `taxwar`、`macroshock(GDP+PPI/CPI)`、`ppi` 相关页面、数据与后端逻辑集中到本目录，并使用 Django 提供服务。

## 目录
- `templates/`: 前端页面模板
- `data/`: 页面依赖数据
- `src/models/`: 模型定义
- `saved_models/` 和 `scalers/`: 预测权重与标准化器
- `stress/`: Django app（页面路由 + API）

## 启动
```bash
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
```

访问：
- `/taxwar`
- `/taxwar/compare`
- `/macroshock`
- `/ppi`

# 部署指南 — Gesture Agent

## 服务器要求

- Ubuntu 20.04+ / Debian 11+
- Python 3.9+
- Nginx
- 至少 1GB 内存

## 1. 安装系统依赖

```bash
sudo apt update
sudo apt install -y nginx python3 python3-venv python3-pip git
```

## 2. 部署代码

```bash
sudo mkdir -p /opt/gesture-agent
sudo chown $USER:$USER /opt/gesture-agent

# 方式一：git clone
git clone <your-repo-url> /opt/gesture-agent

# 方式二：本地打包上传
# 本地执行: tar czf gesture-agent.tar.gz --exclude='.venv' .
# 服务器执行: tar xzf gesture-agent.tar.gz -C /opt/gesture-agent
```

## 3. 创建虚拟环境并安装依赖

```bash
cd /opt/gesture-agent
python3 -m venv .venv
.venv/bin/pip install -e .
```

## 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 SILICONFLOW_API_KEY
nano .env
```

## 5. 配置 Nginx

```bash
sudo cp deploy/nginx/gesture-agent.conf /etc/nginx/sites-available/gesture-agent
sudo ln -sf /etc/nginx/sites-available/gesture-agent /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

## 6. 配置 systemd 服务

```bash
sudo cp deploy/systemd/gesture-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gesture-agent
sudo systemctl start gesture-agent
```

## 7. 设置文件权限

```bash
sudo chown -R www-data:www-data /opt/gesture-agent
```

## 8. 验证

```bash
# 检查服务状态
sudo systemctl status gesture-agent

# 检查日志
sudo journalctl -u gesture-agent -f

# 测试 API
curl http://localhost/api/health

# 浏览器访问
# http://<服务器IP>/
```

## 常用运维命令

```bash
# 重启服务
sudo systemctl restart gesture-agent

# 查看日志
sudo journalctl -u gesture-agent --since "10 min ago"

# 更新代码后重启
cd /opt/gesture-agent && git pull
sudo systemctl restart gesture-agent
```

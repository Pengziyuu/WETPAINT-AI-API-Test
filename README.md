# AI 評分服務 API

這是一個容器化的 AI 評分服務，提供安全的 API 接口，用於圖形識別和評分。

## 目錄
- [功能特點](#功能特點)
- [系統架構](#系統架構)
- [部署指南](#部署指南)
  - [環境需求](#環境需求)
  - [Docker 部署](#docker-部署)
  - [本機開發環境](#本機開發環境)
  - [部署方式比較](#部署方式比較)
- [外網訪問設定](#外網訪問設定)
- [API 說明](#api-說明)
  - [Swagger 文檔](#swagger-文檔)
  - [認證機制](#認證機制)
  - [API 規格](#api-規格)
  - [使用範例](#使用範例)
- [疑難排解](#疑難排解)
- [注意事項](#注意事項)

## 功能特點

- 容器化部署（支援 CUDA）
- 安全的 API 金鑰認證
- 基於 FastAPI 的高效框架
- 多模型圖像評分功能
- 支援五種基本圖形識別（圓形、十字、方形、三角形、菱形）
- 支援本地或雲端部署

## 系統架構

### 目錄結構

```
ai_service/
├── .env                   # 環境變數設定檔
├── Dockerfile             # Docker 容器設定檔
├── README.md              # 專案說明文件
├── api.py                 # API 服務主程式
├── autoScore.py           # AI 評分模型主程式
├── config.py              # 配置管理
├── draw_trainingdata.zip  # 壓縮的模型訓練資料 (已移至雲端)
├── requirements.txt       # Python 依賴套件列表
└── swagger.yaml           # OpenAPI 規範文件
```

### 模型結構
解壓後的 `draw_trainingdata` 目錄包含以下模型：

```
draw_trainingdata/
├── D/                # 菱形相關模型
├── O/                # 圓形相關模型
├── S/                # 方形相關模型
├── T/                # 三角形相關模型
├── X/                # 十字相關模型
└── five_pattern/     # 五種圖形識別模型
```

## 部署指南

### 環境需求

- NVIDIA GPU (推薦用於模型推理加速)
- Docker 環境
- Python 3.8 (相容於 Ubuntu 20.04 LTS)

### Docker 部署

1. **構建 Docker 映像**：
   ```bash
   docker build -t ai-service .
   ```

2. **GPU 模式運行**：
   ```bash
   docker run --gpus all -p 0.0.0.0:8000:8000 --env-file .env ai-service
   ```

3. **CPU 模式運行**：
   ```bash
   docker run -p 0.0.0.0:8000:8000 --env-file .env ai-service
   ```

### 本機開發環境

1. **安裝依賴**：
   ```bash
   pip install -r requirements.txt
   ```

2. **運行開發服務器**：
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

3. **解壓模型檔案**：
   ```bash
   unzip -O UTF-8 draw_trainingdata.zip
   ```

### 部署方式比較

| 特性 | Docker 部署 | 本機開發 |
|------|------------|---------|
| **環境隔離** | ✅ 完全隔離 | ❌ 依賴本機環境 |
| **跨平台兼容性** | ✅ 高 | ❌ 受限於本機設定 |
| **熱重載** | ❌ 需要重建鏡像 | ✅ 支援 `--reload` |
| **部署簡易度** | ✅ 一鍵部署 | ❌ 需要設置環境 |
| **GPU 支援** | ✅ 需使用 `--gpus` 參數 | ✅ 直接使用本機 GPU |
| **適用場景** | 生產環境 | 開發和調試 |

## 外網訪問設定

### 雲服務器部署

1. **防火牆設定**：
   - 確保雲服務器允許 8000 端口的入站流量
   - 建議限制訪問 IP 範圍

2. **域名設定**（可選）：
   - 設置 A 記錄指向服務器 IP
   - 例如：`api.yourdomain.com` -> `123.45.67.89`

3. **HTTPS 設定**（建議）：
   - 使用 Nginx 作為反向代理
   - 配置 SSL 證書

### 本地網路部署

1. **路由器設定**：
   - 設置端口轉發（8000端口）
   - 如有必要，配置動態 DNS 服務

## API 說明

### Swagger 文檔

本專案提供完整的 Swagger API 文檔，您可以通過以下方式訪問：

1. **Swagger UI**：
   - 訪問 `http://localhost:8000/docs`
   - 提供互動式的 API 測試界面
   - 支援直接在瀏覽器中測試 API 端點

2. **ReDoc**：
   - 訪問 `http://localhost:8000/redoc`
   - 提供更詳細的 API 文檔視圖
   - 適合閱讀和參考

3. **OpenAPI 規範文件**：
   - 位置：`swagger.yaml`
   - 符合 OpenAPI 3.0.0 規範
   - 可用於生成客戶端代碼或導入到其他 API 工具

### 認證機制

#### 設定 API 金鑰

1. 編輯 `.env` 文件，設定您的 API 金鑰：
   ```
   API_KEY=my_secret_key_123
   ```

2. 安全建議：
   - 使用至少 32 個隨機字符
   - 定期更換金鑰
   - 不要提交金鑰到版本控制系統

#### 使用 API 金鑰

所有請求都需要在 HTTP 標頭中包含 API 金鑰：
```
X-API-Key: your_api_key
```

### API 規格

#### 評分端點

- **URL**: `/score`
- **方法**: POST
- **認證**: 需要 API 金鑰

#### 請求格式
```json
{
    "CaseID": "string",      // 個案 ID
    "imagepath": "string",   // 圖片路徑或 URL
    "months": "int",         // 月齡
    "model": "string"        // 模型選擇
}
```

#### 回應格式
```json
{
    "CaseID": "string",      // 個案 ID
    "Score": {               // 包含各項評分的字典
        "Score": "float",    // 總分
        "O": "float",        // 圓形評分
        "X": "float",        // 十字評分
        "S": "float",        // 方形評分
        "T": "float",        // 三角形評分
        "D": "float"         // 菱形評分
    },
    "Probability": "float",  // 可信度
    "Timestamp": "string"    // 時間戳
}
```

### 使用範例

#### 使用 curl

```bash
curl -X POST "http://your-server-ip:8000/score" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "CaseID": "case123",
    "imagepath": "https://example.com/image.jpg",
    "months": 24,
    "model": "default"
  }'
```

#### 使用 Python

```python
import requests

url = "http://your-server-ip:8000/score"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your_api_key"
}

data = {
    "CaseID": "case123",
    "imagepath": "https://example.com/image.jpg",
    "months": 24,
    "model": "default"
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## 疑難排解

### 常見認證問題

1. **401 Unauthorized 錯誤**：
   - 確認 API 金鑰標頭名稱是否為 `X-API-Key`
   - 確認金鑰值與 `.env` 文件匹配

2. **圖片載入錯誤**：
   - 檢查 imagepath 是否可訪問
   - 確認支援的圖片格式

3. **模型載入錯誤**：
   - 確認 draw_trainingdata 目錄是否正確解壓
   - 檢查模型文件權限

## 注意事項

- GPU 加速可大幅提高處理效能
- 支援常見圖片格式（JPEG、PNG 等）
- 建議使用 HTTPS 保護 API 通信
- 定期更新系統和依賴以確保安全性
- 在生產環境中設置適當的日誌記錄和監控 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from datetime import datetime
from autoScore import Runall
import requests
import logging
import os
import sys

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置基礎路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'draw_trainingdata')

# 確保系統支援中文
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'zh_TW.UTF-8'
    os.environ['LC_ALL'] = 'zh_TW.UTF-8'

def check_model_files():
    """檢查模型文件是否存在"""
    try:
        # 檢查基礎目錄
        if not os.path.exists(MODEL_DIR):
            logger.error(f"模型目錄不存在: {MODEL_DIR}")
            return False
            
        logger.info("模型目錄檢查通過")
        return True
    except Exception as e:
        logger.error(f"檢查模型文件時發生錯誤: {str(e)}")
        return False

# 初始化 FastAPI 應用
app = FastAPI(
    title="AI 評分服務 API",
    description="提供圖像識別和評分的 API 服務",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，生產環境應設置具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義輸入模型
class AIRequest(BaseModel):
    CaseID: str
    imagepath: str  # 可以是URL或本地路徑
    months: int
    model: str

# 定義輸出模型
class AIResponse(BaseModel):
    CaseID: str
    Score: dict  # 接受字典類型，包含各項評分
    Probability: float
    Timestamp: str

# 根路徑
@app.get("/")
async def root():
    """API 根路徑，返回歡迎信息"""
    return {"message": "AI 評分服務 API", "version": "1.0.0"}

# 從 URL 下載圖片
def download_image_from_url(url: str) -> np.ndarray:
    """從 URL 下載圖片並轉換為 numpy 數組"""
    try:
        logger.info(f"開始下載圖片: {url}")
        # 設置請求頭
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 處理 Imgur 圖片特殊情況
        if 'imgur.com' in url:
            # 確保使用直接圖片URL
            if not url.startswith('https://i.imgur.com'):
                image_id = url.split('/')[-1]
                if '.' not in image_id:
                    url = f'https://i.imgur.com/{image_id}.jpg'
        
        # 下載圖片
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 將圖片數據轉換為 numpy 數組
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("圖片解析失敗")
            raise HTTPException(status_code=400, detail="無法解析圖片格式")
        
        logger.info("圖片下載和解析成功")
        return image
    except requests.exceptions.RequestException as e:
        logger.error(f"下載圖片失敗: {str(e)}")
        raise HTTPException(status_code=400, detail=f"下載圖片失敗: {str(e)}")
    except Exception as e:
        logger.error(f"處理圖片時發生未知錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"處理圖片時發生錯誤: {str(e)}")

# 評分 API 端點
@app.post("/score", response_model=AIResponse)
async def score_image(
    request: AIRequest
):
    """處理圖像評分請求"""
    try:
        logger.info(f"收到評分請求: CaseID={request.CaseID}, imagepath={request.imagepath}")
        
        # 檢查模型文件
        if not check_model_files():
            raise HTTPException(status_code=500, detail="模型文件缺失或路徑錯誤")
        
        # 判斷是URL還是本地路徑並載入圖片
        if request.imagepath.startswith(('http://', 'https://')):
            image = download_image_from_url(request.imagepath)
        else:
            # 讀取本地圖片
            logger.info(f"嘗試讀取本地圖片: {request.imagepath}")
            image = cv2.imread(request.imagepath)
            
        if image is None:
            logger.error("無法讀取圖片")
            raise HTTPException(status_code=400, detail="無法讀取圖片")

        # 執行 AI 評分
        logger.info("開始執行AI評分")
        result = Runall(image)
        logger.info(f"AI評分完成: {result}")
        
        # 生成時間戳
        timestamp = datetime.now().isoformat()

        # 構建回應
        response = AIResponse(
            CaseID=request.CaseID,
            Score=result,  # 直接使用完整的 result 字典
            Probability=1.0,
            Timestamp=timestamp
        )
        
        logger.info(f"評分完成: {response}")
        return response
        
    except Exception as e:
        logger.error(f"處理請求時發生錯誤: {str(e)}")
        # 添加目錄內容到錯誤信息以幫助診斷
        try:
            dir_content = os.listdir(MODEL_DIR)
            logger.error(f"模型目錄內容: {dir_content}")
            if os.path.exists(os.path.join(MODEL_DIR, 'D')):
                d_content = os.listdir(os.path.join(MODEL_DIR, 'D'))
                logger.error(f"D目錄內容: {d_content}")
        except Exception as dir_error:
            logger.error(f"無法讀取目錄: {str(dir_error)}")
        raise HTTPException(status_code=500, detail=str(e))

# 主入口點
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
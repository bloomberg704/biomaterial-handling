import cv2
import numpy as np

# step1 작업 결과 이어받기 가정
image_path = 'C:/Users/dkswo/OneDrive/바탕 화면/biomaterial/week2/apple_top_A.png'
img_array = np.fromfile(image_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
original_display = img.copy()

# 사과는 붉거나 초록이어서 R, G, B 값의 차이가 매우 큽니다.
# 반면 바닥의 그림자는 회색(무채색)이어서 R=G=B로 값의 차이가 거의 없습니다.
b, g, r = cv2.split(img)
max_rgb = np.maximum(np.maximum(b, g), r)
min_rgb = np.minimum(np.minimum(b, g), r)
chroma = cv2.subtract(max_rgb, min_rgb)

# RGB 편차(Chroma) 지도를 이용해 컴퓨터가 사진 상태에 맞는 최적의 컷트라인(Otsu)을 자동 계산합니다
ret, _ = cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 사과 꼭지나 밑동처럼 빛을 못 받아 거무죽죽해진 부분(채도가 5~20)마저 잘려나가는 것을 막기 위해,
# 컴퓨터가 찾은 컷트라인(ret)보다 조금 더 관대하게(-15) 기준을 낮춰줍니다. 최소 5은 유지하여 그림자는 자릅니다.
adjusted_ret = max(5, ret - 15)
_, thresh = cv2.threshold(chroma, adjusted_ret, 255, cv2.THRESH_BINARY)

# 사과 표면의 하얀색 빛 반사 (채도가 낮아 뚫리는 구멍) 등 내부를 촘촘히 닫기 연산으로 메움
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

# 테두리를 가볍게 다듬어 오돌토돌한 엣지를 평활화
kernel_open = np.ones((1, 1), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

# 4. 윤곽선 추출 (Contour) - CHAIN_APPROX_NONE 유지로 높은 해상도 스캐닝 보장
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 5. 윤곽선 시각화 - 쓸데없는 노이즈를 다 무시하고, '가장 면적이 큰 사과 윤곽선 1개만' 정확하게 그리기
if contours:
    apple_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(original_display, [apple_contour], -1, (0, 255, 0), 2)

cv2.imshow("Step 2: Threshold & Contour", original_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

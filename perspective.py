def find_big_box(image):
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    
    # image = FT.resize(image, (224,224))
    image = np.array(image)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    thresh = np.where(image>np.mean(image)//20*20, 255, 0).astype(np.uint8)
    thresh = cv2.erode(thresh, erode_kernel)

    _, labels = cv2.connectedComponents(thresh)

    thresh_list =list()
    thresh_idx =list()

    for (i, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        thresh_idx.append(numPixels)
        thresh_list.append(labelMask)
    if not thresh_idx:
        print("pass")
        return image_RGB
#     print(thresh_idx)
    img =thresh_idx.index(max(thresh_idx))
    img = thresh_list[img]
    row, col = img.shape
    arr = np.empty((0,2), int)

    for r in range(row):
        for c in range(col):
            if(img[r,c] > 0): 
                arr = np.append(arr, np.array([[c,r]]), axis=0)
    
    sm = arr.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
    diff = np.diff(arr, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

    topLeft = arr[np.argmin(sm)]         # x+y가 가장 값이 좌상단 좌표
    bottomRight = arr[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
    topRight = arr[np.argmin(diff)]     # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = arr[np.argmax(diff)]   # x-y가 가장 큰 값이 좌하단 좌표
    
    x1,y1 = topLeft
    x2, y2 = topRight
    x3, y3 = bottomRight
    x4, y4 = bottomLeft
    degree = cal_rad([x1,y1,x2,y2])
    # print(f"degree: {degree}")
    # if(degree >-4  and degree <4.5):
    if(degree >-4  and degree <0.01):
        result = image_RGB[min(y1,y2):max(y3,y4), min(x1,x4):max(x2,x3),:]
    else:
        result = find_perspective(topLeft, topRight, bottomRight , bottomLeft, image_RGB)
    
    return result

def find_perspective(topLeft, topRight, bottomRight , bottomLeft, image_RGB):
    pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산 ---③ 
    w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
    w2 = abs(topRight[0] - topLeft[0])          # 하당 좌우 좌표간의 거리
    h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
    h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
    width = int(max([w1, w2]))                       # 두 좌우 거리간의 최대값이 서류의 폭
    height = int(max([h1, h2]))                      # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0,0], [width-1,0], 
                        [width-1,height-1], [0,height-1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    # 원근 변환 적용
    result = cv2.warpPerspective(image_RGB, mtrx, (width, height))

    return result

def cal_rad(arr):
    rad = math.atan2(arr[3]-arr[1], arr[2]-arr[0])
    result = radTodegree(rad)
    return result
def radTodegree(rad):
    PI = math.pi
    deg = (rad*180)/PI
    return deg

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def zone_interest(image,v=100):
    height = image.shape[0]
    width = image.shape[1]
    zone = np.array([[(0,height),(0,0),(width,0),(width,height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly( mask, zone,255)
    maker = cv2.bitwise_and(image,mask)
    return maker

def display_line1(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
    return line_image       



# threshold tinh theo truc y 
def zone (lines, mi):
    a=[]
    for line in lines:
        if line[0][1] >= mi:
            a.append(line)
    return a
# determined firt point, point belong real way
# dis is distance of left way to right way
def learn_real_way(image,point_2d):
    width = image.shape[1]
    left_way=[]
    right_way=[]
    a,b=0,0
    check = 0
    for point in point_2d:
        if point[0] < int(width/2)and a==1:
            left_real_way = np.array([point[0],point[1]])
            a=1
            check = check + 1
        if point[0] >= int(width/2)and b==1:
            right_real_way = np.array([point[0],point[1]])
            b=1
            check = check + 1
        if check ==2:
            break
    if a*b==0:
        return 0
    return left_real_way, right_real_way
        
'''-----------------------------------------------------------------------------'''
# dis is distance of left way to right way
def learn_real_way(image,point_2d, dis=100):
    width = image.shape[1]
    left_way=[]
    right_way=[]
    check = 0
    left_real_way = np.array([point[0][0],point[0][1]])
    for point in point_2d:
        if  point[0] - left_real_way[0] >= dis:
            right_real_way = np.array([point[0],point[1]])
            check = 1      
           
        if  point[0] - left_real_way[0] <= -dis:
            right_real_way = left_real_way
            left_real_way = np.array([point[0],point[1]])
            check = 1
        if check ==1:
            break
    if check == 0:
        if left_real_way[0] <= width/2:
            right_real_way = left_real_way
            left_real_way = 0
        else:
            right_real_way =0  
        return left_real_way, right_real_way
    return left_real_way, right_real_way
    

def display_line(image, point_2d):
    line_image = np.zeros_like(image)
    if point_2d is not None:
        for point in point_2d:
            x1,y1,x2,y2 = point[0],point[1],point[2],point[3]
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),1)
    return line_image       




def check_lane( lines):
    if lines is not None:
        left_line=[]
        right_line = []
        mi = lines[0][0][0]
        ma = lines[0][0][0]
        for line in lines:
            if line[0][0] < mi:
                mi = line[0][0]
            if line[0][0] > ma:
                ma = line[0][0]
        tb = (mi+ma)/2
        for line1 in lines:
            if line1[0][0] < tb:
                left_line.append(line1[0])
            else:
                right_line.append(line1[0])
        
        return left_line,right_line

def cosin(u,v):
    u_x = u[0]
    u_y = u[1]
    v_x = v[0]
    v_y = v[1]
    cos_u_v = (u_x*v_x + u_y*v_y)/((math.sqrt(u_x**2+u_y**2))* (math.sqrt(v_x**2+v_y**2)))
    u_v = math.acos(cos_u_v) * 180 / np.pi
    return u_v

def tan_phi(a,b):
    return np.arctan((b[0]-a[0])/(a[1]-b[1]))*180/np.pi

def vec_a_b(point_2d):
    x = point_2d[2] - point_2d[0]
    y = point_2d[3] - point_2d[1]
    return np.array([x,y])

def distance (point):
    d = ((point[2]-point[0])**2 + (point[3]-point[1])**2)**0.5
    return d


# segment lines length more threshold
def segment(point, threshold = 10):
    k = 0
    l = k+1
    while l!=k:
        l=k
        for i in range(len(point)):
            #l=k
            if distance(point[i]) >= 2*threshold:
                x_tg = int(abs((point[i][2] + point[i][0])/2))
                y_tg = int(abs((point[i][3] + point[i][1])/2))
                a = [point[i][0], point[i][1], x_tg, y_tg]
                b = [ x_tg, y_tg, point[i][2], point[i][3]]
                del point[i]
                point.append(a)
                point.append(b)
                k = k+1
                break
    return point


# p is difference allow between left way and right ( for pixel )
# d is difference allow between vecto u and vecto v ( for radian )
#x is pixel / cm ( cm in way = x pixel in image)
def real_way (image, left_real_way, right_real_way, point_2d, p=20, h=0,x=30):
    left_way = []
    right_way = []
    a=[]
    check_left, check_right = 1,1
    heigh = image.shape[0]
    for point in point_2d :
        if point[1]<h:
            break
        else:
            if point[1] < left_real_way[1]:
                if abs( left_real_way[0] - point[0] ) <= p:#and abs(left_real_way[3] - point[1]) <= 50 :
                    left_real_way = point
                    left_way.append(point)
                    if heigh - point[1] > x and check_left ==1:
                       check_left=0
                       select_left = np.array([point[0],point[1]])
                      
            if point[1] < right_real_way[1]:               
                if abs( right_real_way[0] - point[0] ) <= p: #and abs(right_real_way[3] - point[1]) <= 200 :            
                    right_real_way = point
                    right_way.append(point)
                    if heigh - point[1] > x and check_right ==1:
                       check_right=0
                       select_right = np.array([point[0],point[1]])
                
    return left_way, right_way, select_left, select_right

def dis(p1, p2):
    return math.sqrt(sum((p1 - p2) ** 2))


def get_sub_lines(p1, p2, threshold = 10):
    if dis(p1, p2) >= 2*threshold:
        arr = []
        mid = (p1 + p2) / 2
        arr.append(get_sub_lines(p1, mid))
        arr.append(get_sub_lines(mid, p2))
        return np.array(arr).reshape(-1, 4)
    else:
        return np.array([np.append(p1, p2)])


# chuyen mang 3 chieu thanh mang 2 chieu               
def D3_D2 (lines):
    lines_2d = np.array(lines).reshape(-1, 4)
    list_arr = []
    for p in lines_2d:
        arr = get_sub_lines(p[:2], p[2:])
        list_arr.append(arr)
    a = np.concatenate(list_arr, axis=0).astype(np.int)
    L=len(a)
    b=a
    for i in range(L):
        if a[i][1]<a[i][3]:
            a[i] = np.array([a[i][2],a[i][3],a[i][0],a[i][1]])
    #a.sort(key=lambda x: x[1])
    indices = np.argsort(a[:,1])
    a = a[indices]
    for j in range(L):
        b[j] = a[L-1-j]
    return b


def fun( left_way, right_way, threshold ):
    for l in left_way:
        if l[3] - threshold <= 0:
            break
    for r in right_way:
        if r[3] - threshold <= 0:
            break
    return l,r


#check overlaping between left way and rght way, remove it
def overlaping(left_way, right_way, image, threshold = 50):
    high = image.shape[0]
    step = high//20
    for h in range(high, 0, - step):
        l,r = fun( left_way, right_way, h)
        if abs(r[0]-l[0]) <= threshold:
            l_index = left_way.index(l)
            r_index = right_way.index(r)
            del left_way[ l_index :]
            del right_way[ r_index :]
            break
    return left_way, right_way

def similar ( a,v,alpha):
    global frame_rate
    if alpha >= 0:
        # turn right
        return np.array([a[0]-(v/frame_rate)*math.sin(alpha*np.pi/180),a[1]+(v/frame_rate)*math.cos(alpha*np.pi/180)])
    else:
        #turn left
        return np.array([a[0]+(v/frame_rate)*math.sin(alpha*np.pi/180),a[1]+(v/frame_rate)*math.cos(alpha*np.pi/180)])

def guess(old_left_real_way, old_right_real_way, phi, v, time):
    if phi<=0:# turn left
        phi = abs(phi)
        left_point = np.array(old_left_real_way[0]+(v/time)*math.sin(phi*np.pi/180), old_left_real_way[1]+(v/time)*math.cos(phi*np.pi/180) )
        right_point = np.array(old_right_real_way[0]+(v/time)*math.sin(phi*np.pi/180), old_right_real_way[1]+(v/time)*math.cos(phi*np.pi/180) )    
    else: # turn right
        left_point = np.array(old_left_real_way[0]-(v/time)*math.sin(phi*np.pi/180), old_left_real_way[1]+(v/time)*math.cos(phi*np.pi/180) )
        right_point = np.array(old_right_real_way[0]-(v/time)*math.sin(phi*np.pi/180), old_right_real_way[1]+(v/time)*math.cos(phi*np.pi/180) )
    return left_point, right_point


def round_phi (phi):
    p = phi
    phi = abs( phi)
    if 0<=phi<=5:
        phi = 0
    elif 5<phi<=15:
        phi = 10
    elif 15<phi<=25:
        phi = 20
    elif 25<phi<=35:
        phi = 30
    elif 35<phi<=45:
        phi = 40
    elif 45<phi<=55:
        phi = 50
    elif 55<phi<=65:
        phi = 60
    elif 65<phi<=75:
        phi = 70
    else:
        phi = 80
    if p < 0:
        phi = -phi
    return phi
                        
cap = cv2.VideoCapture(0)

while(1):
        ret, frame = cap.read()
        cany = canny(frame)
        cv2.imshow('canny',cany)
        if not ret:
                break
        k = cv2.waitKey(1)

        if k%256 == 32:
        # SPACE pressed
                image = frame
                cany = canny(image)
                zone = zone_interest(cany)
                lines = cv2.HoughLinesP( zone,1, np.pi/180, 10, minLineLength=5, maxLineGap=5)
                point_2d = D3_D2 (lines)
                point_2d = point_2d.tolist() # convert type list array to type list

                if learn_real_way(image, point_2d)==0:
                    continue
                left_real_way, right_real_way = learn_real_way(image, point_2d)
                '''-----------------------------------------------------------------------------'''
                left_real_way, right_real_way = learn_real_way(image, point_2d)
                if left_real_way == 0:
                    '''turn left 10do
                        continue'''
                if right_real_way == 0:
                    '''turn right 10do
                        continue'''
                    
                
                '''-----------------------------------------------------------------------------'''
              
                left_way, right_way, select_left, select_right = real_way( image, left_real_way, right_real_way, point_2d )
                left_way, right_way = overlaping( left_way, right_way, image, 200)                
                way = left_way + right_way
                new_point_axit_x = (left_way[len(left_way)-1][2] + right_way[len(right_way)-1][2])//2
                new_point_axit_y = (left_way[len(left_way)-1][3] + right_way[len(right_way)-1][3])//2
                new_point = [new_point_axit_x, new_point_axit_y]
                car_in_image = [image.shape[1]//2, image.shape[0]]
                phi = round_phi(tan_phi(car_in_image, new_point))

                ''' xe chay '''
                
                break


while(1):
        start_time = time.time()
        ret, image = cap.read()
        #point_real_way = [ left_real_way[0]+right_real_way[0]//2, left_real_way[1]+right_real_way[1]//2]

        left_way, right_way, select_left, select_right = real_way( image, left_real_way, right_real_way, point_2d )
        left_way, right_way = overlaping( left_way, right_way, image, 200)

        
        way = left_way + right_way

        new_point_axit_x = (left_way[len(left_way)-1][2] + right_way[len(right_way)-1][2])//2
        new_point_axit_y = (left_way[len(left_way)-1][3] + right_way[len(right_way)-1][3])//2
        new_point = [new_point_axit_x, new_point_axit_y]
        car_in_image = [image.shape[1]//2, image.shape[0]]
        phi = round_phi(tan_phi(car_in_image, new_point))
        print('deviation = ',phi)
        
        
        ''' xe chay '''

        time = time.time()-start_time
        left_real_way, right_real_way = guess( select_left, select_right, phi, v, time)
        lines_image = display_line(image, way)
        cv2.imshow('a',lines_image)
        
        if cv2.waitKey(1)==ord('q'):
                break;

cv2.destroyAllWindows()
cap.release()            





        



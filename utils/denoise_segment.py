import numpy as np
import cv2
def me_median(a):
    return np.mean(a, axis = 0)

def connected_CC(stat1,stat2):
    x_new = min(stat1[0],stat2[0])
    y_new = min(stat1[1],stat2[1])
    w_new = max(stat1[0]+stat1[2],stat2[0]+stat2[2])-x_new
    h_new = max(stat1[1]+stat1[3],stat2[1]+stat2[3])-y_new
    area_new = w_new*h_new
    return [x_new,y_new,w_new,h_new,area_new]
    
def distance(stat1, stat2):
    if stat1[0] > stat2[0]:
        stat1, stat2 = stat2, stat1
    return stat2[0] - stat1[0] - stat1[2]

def merge_CC(stats_final_, thresh_hold):
    #[x y w h area]
    # kiem tra CC chong cheo nhau truoc
    stats_list = stats_final_.copy()
    i = 0
    while(i<len(stats_list)-1):
        if stats_list[i][0]<=stats_list[i+1][0] and  stats_list[i][0]+stats_list[i][2]>=stats_list[i+1][0]:
                stats_list[i] =  connected_CC(stats_list[i],stats_list[i+1])    
                stats_list = np.delete(stats_list,i+1,axis=0)
        else:i=i+1
        # co chong cheo xay ra
    me_median_width = me_median(stats_list[:,2])
    i = 0
    while i < len(stats_list):
        # kiem tra noi neu < median_width
        if stats_list[i][2]< int(0.61*me_median_width) and distance(stats_list[i], stats_list[i+1]) < thresh_hold:
            if i==0:# noi voi cai tiep theo
                stats_list[i+1] = connected_CC(stats_list[i],stats_list[i+1])
                stats_list = np.delete(stats_list,i,axis=0)
                continue
            elif i==len(stats_list)-1 and distance(stats_list[i], stats_list[i-1]) < thresh_hold: # noi voi cai truoc do
                stats_list[i-1] = connected_CC(stats_list[i-1],stats_list[i])
                stats_list = np.delete(stats_list,i,axis=0)
                continue
            else: #kiem tra khoang cac den 2 cai gan nhat
                if (stats_list[i][0]-stats_list[i-1][0]-stats_list[i-1][2]\
                  < stats_list[i+1][0]-stats_list[i][0]-stats_list[i][2]) \
                  and (stats_list[i][0]-stats_list[i-1][0]-stats_list[i-1][2] < thresh_hold):
                    if(stats_list[i][2]*0.6>stats_list[i][0]-stats_list[i-1][0]-stats_list[i-1][2]):
                        stats_list[i-1] = connected_CC(stats_list[i-1],stats_list[i])
                        stats_list = np.delete(stats_list,i,axis=0)
                        continue
                    
                elif (stats_list[i+1][0]-stats_list[i][0]-stats_list[i][2]) < thresh_hold:
                    stats_list[i+1] = connected_CC(stats_list[i],stats_list[i+1])
                    stats_list = np.delete(stats_list,i,axis=0)
                i += 1
                continue
            
        i=i+1    
    return stats_list

def Final_Binary_convert(img,connectivity, thresh_hold_distance, breakdown = False):
    # address_area = img
    address_area =cv2.medianBlur(img, 3)   
     
    # return address_area, []
    processed_area = cv2.adaptiveThreshold(address_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           151, 20)
    # if breakdown:
    #     return processed_area, []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed_area = cv2.morphologyEx(processed_area, op=cv2.MORPH_OPEN, kernel=kernel, anchor=(-1, -1), iterations=-1)
    if breakdown:
        return processed_area, []
    #CC
    num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(processed_area, connectivity, cv2.CV_32S)
    mask = np.ones_like(labels)

    labels_in_mask = list(np.arange(1,num_labels))

    areas = [s[4] for s in stats]

    sorted_idx = np.argsort(areas)

    areas_final = [areas[s] for s in sorted_idx][:-1]
    
    for lidx, cc in zip(sorted_idx, areas_final):
        if cc <= 20:
            mask[labels == lidx] = 0
            labels_in_mask.remove(lidx)
    
    x=stats[:,0][1:]
    y=stats[:,1][1:]
    height,width = img.shape
    
    idx = np.arange(0,len(x)+1)[1:]
    for i,x_ in enumerate(x):
        if y[i]>=0.86500*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:
                labels_in_mask.remove(i+1)
            continue
        # xoa nhung vạch nhỏ sát viền trên
        if y[i]<=(1-0.96)*height and stats[i+1][3]<=(1-0.96)*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:
                labels_in_mask.remove(i+1)
            continue
        #xóa những vạch nhỏ sát mép phải
        if x[i]>=(1-0.996)*width and stats[i+1][2]<=(1-0.996)*width:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:
                labels_in_mask.remove(i+1)
            continue
        # xoa những vạch kẻ tương đối lớn
        if stats[i+1][2]>=0.6*width or  stats[i+1][2] > 3*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:
                labels_in_mask.remove(i+1)
            continue
        # xoa nhung net thang dung(ap dung cho 1 line)
        if stats[i+1][3]>=0.96*height :#and stats[i+1][3]<0.5*height 
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue 

    arr_ret = np.array([stats[s] for s in labels_in_mask])
    arr_temp = np.argsort(arr_ret[:,0])
    stats_final = np.array([arr_ret[s] for s in arr_temp])
    stats_all = stats_final 
    stats_all = merge_CC(stats_final, thresh_hold_distance)     
    
    
    processed_area = processed_area*mask

    processed_area = 255 - processed_area
    return processed_area,stats_all

def recognize_character_bbox(img, connectivity=8, thresh_hold_distance = 5):
    process_img = img
    if img.ndim == 3:
        process_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, stat = Final_Binary_convert(process_img, connectivity=connectivity, thresh_hold_distance= thresh_hold_distance)
    return stat.shape[0], stat
def recognize_character_img(img, connectivity=8, thresh_hold_distance = 5):
    if img.ndim == 3:
        process_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, stat = Final_Binary_convert(process_img, connectivity=connectivity, thresh_hold_distance= thresh_hold_distance)
    res = []
    for bbox in stat:
        bbox = np.squeeze(bbox)
        x, y, w, h,_ = bbox
        res.append(img[y:y+h, x:x+w])
    return len(res), res
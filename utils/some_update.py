import numpy as np
import cv2
def me_median(a):
    m = np.sort(a,axis=0)
    if(len(m)%2==1):
        return m[int(len(m)/2)]
    else: return (m[int(len(m)/2)]+m[int(len(m)/2)-1])/2
def connected_CC(stat1,stat2):
    x_new = min(stat1[0],stat2[0])
    y_new = min(stat1[1],stat2[1])
    w_new = max(stat1[0]+stat1[2],stat2[0]+stat2[2])-x_new
    h_new = max(stat1[1]+stat1[3],stat2[1]+stat2[3])-y_new
    area_new = stat1[4]+stat2[4]
    return [x_new,y_new,w_new,h_new,area_new]
    

def merge_CC(stats_final_):
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
    while(i<len(stats_list)):
        # kiem tra noi neu < median_width
        if(stats_list[i][2]< int(0.61*me_median_width)):
            if(i==0):# noi voi cai tiep theo
                stats_list[i+1] = connected_CC(stats_list[i],stats_list[i+1])
                stats_list = np.delete(stats_list,i,axis=0)
                continue
            elif i==len(stats_list)-1: # noi voi cai truoc do
                stats_list[i-1] = connected_CC(stats_list[i-1],stats_list[i])
                stats_list = np.delete(stats_list,i,axis=0)
                continue
            else: #kiem tra khoang cac den 2 cai gan nhat
                if(stats_list[i][0]-stats_list[i-1][0]-stats_list[i-1][2]
                  <stats_list[i+1][0]-stats_list[i][0]-stats_list[i][2]):
                    
                    if(stats_list[i][2]*0.6>stats_list[i][0]-stats_list[i-1][0]-stats_list[i-1][2]):
                        stats_list[i-1] = connected_CC(stats_list[i-1],stats_list[i])
                        stats_list = np.delete(stats_list,i,axis=0)
                        continue
                elif (stats_list[i][2]*0.6>stats_list[i+1][0]-stats_list[i][0]-stats_list[i][2]):
                    stats_list[i+1] = connected_CC(stats_list[i],stats_list[i+1])
                    stats_list = np.delete(stats_list,i,axis=0)
                    continue

            
        i=i+1    
    return stats_list

def Final_Binary_convert(img,connectivity=4):
    address_area =cv2.medianBlur(img,3)    

    processed_area = cv2.adaptiveThreshold(address_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           151, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_area = cv2.morphologyEx(processed_area, op=cv2.MORPH_OPEN, kernel=kernel, anchor=(-1, -1), iterations=2)
    
    #CC
    num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(processed_area, connectivity, cv2.CV_32S)
    
    mask = np.ones_like(labels)

    labels_in_mask = list(np.arange(1,num_labels))

    areas = [s[4] for s in stats]

    sorted_idx = np.argsort(areas)

    areas_final = [areas[s] for s in sorted_idx][:-1]
    
    for lidx, cc in zip(sorted_idx, areas_final):
        if cc <= 125:
            mask[labels == lidx] = 0
            labels_in_mask.remove(lidx)
    
    x=stats[:,0][1:]
    y=stats[:,1][1:]
    height,width = img.shape
    
    idx = np.arange(0,len(x)+1)[1:]
    for i,x_ in enumerate(x):
        if y[i]>=0.86500*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        # xoa nhung vạch nhỏ sát viền trên
        if y[i]<=(1-0.96)*height and stats[i+1][3]<=(1-0.96)*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        #xóa những vạch nhỏ sát mép phải
        if x[i]>=(1-0.996)*width and stats[i+1][2]<=(1-0.996)*width:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        # xoa những vạch kẻ tương đối lớn
        if stats[i+1][2]>=0.6*width or  stats[i+1][2] > 3*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        # xoa nhung net thang dung(ap dung cho 1 line)
        if stats[i+1][3]>=0.96*height :#and stats[i+1][3]<0.5*height 
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue 

    arr_ret = np.array([stats[s] for s in labels_in_mask])
    arr_temp = np.argsort(arr_ret[:,0])
    stats_final = np.array([arr_ret[s] for s in arr_temp])
    stats_all = merge_CC(stats_final)      
    
    processed_area = processed_area*mask

    processed_area = 255 - processed_area
    return processed_area,stats_all
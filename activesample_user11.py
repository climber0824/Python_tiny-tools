"""
add entry_center and create a timer for routine running

"""

import os
import cv2
import glob
import json
import numpy as np
import time
import math
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from line_profiler import LineProfiler

from util import privacy, create_folder
from config import ActiveConfig as Config


class ActiveSample():
    def __init__(self, config):
        # Config
        self.config = config

        # Background sub
        self.knnbg = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))

        # Segmentation
        self.segmentation = cv2.imread(self.config.seg_path)
        self.segmentation = cv2.resize(self.segmentation, (self.config.img_w,self.config.img_h))
        
        # Pre-process segmentation image
        self.segmask = self.segmentation.copy()
        for i in range(self.config.img_h):
            for j in range(self.config.img_w):
                if sum(self.segmentation[i][j]) == 0:
                    self.segmask[i][j] = [0, 0, 0]
                else:
                    self.segmask[i][j] = [1, 1, 1]

        # Pre-process entry-segmentation image
        self.entry_mask = self.segmentation.copy()
        for i in range(self.config.img_h):
            for j in range(self.config.img_w):
                if sum(self.segmentation[i][j]) == 300:                   
                    self.entry_mask[i][j] = [100, 100, 100]
                else:                   
                    self.entry_mask[i][j] = [0, 0, 0]

        # privacy flag
        self.privacy_processing = False

        # active sample box result
        self.activesampleresult = []

        # visualization motion flag
        self.visual_flag = False
        
        # save image index
        self.save_img = True
        self.index = 0
        self.save_index_FP = 0
        self.save_index_FN = 0
        self.need_ref_FN = False
        self.need_ref_FP = False
        self.save_FN_img = True   #for deleting duplicate FN

        # optical threshold and BG threshold
        self.human_threshold = 0.3  # region thres for FN motion box
        self.human_threshold_FP = 0.2  # region thres for FP motion box
        self.opt_threshold = 0.001  # region thres for FN optical box
        self.iou_threshold = 0.4  # iou thres for FN motion box
        self.motion_box_th = 1000  # response thres for motion box candidate 
        self.convert_opt_to_binary_th = 50  # response thres for optical box candidate
        self.overlap_threshold = 0.6  #for deleting duplicate FN
        self.entry_dist_thr = 70  # the threshold of  motion_boxs and entry distances 
        
        # function segmentation
        self.activesampledoor = False
        self.function_seg_img = cv2.imread(self.config.function_seg)
        self.function_seg_img = self.active_sample_door(self.function_seg_img)

        # zone for detecting persons:
        self.zone_one = [400, 0, 800, 300]
        self.zone_two = [0, 0, 400, 300]
        self.zone_three = [0, 300, 400, 600]
        self.zone_four = [400, 300, 800, 600]

    def knn(self, frame):
        fgmask = self.knnbg.apply(frame)

        return fgmask


    def findfg(self, fgbgp, frame):
        all_bbox = []
        contours, hierarchy = cv2.findContours(fgbgp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_check = False
        for c in contours:
            if cv2.contourArea(c) < self.motion_box_th:
                continue
            motion_check = True
            (x, y, w, h) = cv2.boundingRect(c)
            all_bbox.append([x, y, x + w, y + h])

        if motion_check: motion_trigger = True
        else: motion_trigger = False

        return frame, motion_trigger, all_bbox


    def iou(self, bbox1, bbox2):
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0

        intersection_area = abs(x_right - x_left) * abs(y_bottom - y_top)
        bb1_area = abs(bbox1[0] - bbox1[2]) * abs(bbox1[1] - bbox1[3])
        bb2_area = abs(bbox2[0] - bbox2[2]) * abs(bbox2[1] - bbox2[3])
        ioue = intersection_area / float(bb1_area + bb2_area - intersection_area)

        return ioue


    def entry_center(self, entry_mask):
        gray = cv2.cvtColor(self.entry_mask, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        binaryIMG = cv2.Canny(blurred, 20, 160)
        cnts, hierarchy = cv2.findContours(binaryIMG.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_cX = []
        total_cY = []
        for c in cnts:
            M = cv2.moments(c)            
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                total_cX.append(cX)
                total_cY.append(cY)

        return total_cX, total_cY


    def false_negative(self, vis, pred_box, motion_box, thr, mapf, human_threshold, FN_box, number, store_img, buff, save_FN_img):
        FN_cand = []
        box_too_large_thres = 40000

        # draw all preds on vis
        for people in range(len(pred_box)):
            box = pred_box[people]
            cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(vis, "pred", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        for motion_b in motion_box:
            # filter out too large box
            if (motion_b[2] - motion_b[0]) * (motion_b[3] - motion_b[1]) > box_too_large_thres:
                continue

            #filter out non-floor region:
            central_point = (motion_b[0] + (motion_b[2]- motion_b[0]) // 2, motion_b[1] + (motion_b[3] - motion_b[1]) // 2)
            #print("central_point1", central_point)
            
            if  self.segmentation[central_point[1]][central_point[0]][0] != 0 or \
                self.segmentation[central_point[1]][central_point[0]][1] != 100 or \
                self.segmentation[central_point[1]][central_point[0]][2] != 0:
                continue

            # motion region
            region1 = thr[int(motion_b[1]):int(motion_b[3]), int(motion_b[0]):int(motion_b[2])] / 255
            motion_region = np.where(region1 == 1)
            # optical region
            region2 = mapf[int(motion_b[1]):int(motion_b[3]), int(motion_b[0]):int(motion_b[2])] / 255
            flow_region = np.where(region2 == 1)
            
            motion_ratio = len(motion_region[0]) / (region1.shape[0] * region1.shape[1]) 
            flow_ratio = len(flow_region[0]) / (region2.shape[0] * region2.shape[1])

            if motion_ratio > human_threshold and flow_ratio > self.opt_threshold:             
                iou_check = False
                
                for people in range(len(pred_box)):
                    box = pred_box[people]
                    iou_result = self.iou(motion_b, [int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                    if iou_result > self.iou_threshold:
                        iou_check = True
                
                if not iou_check:
                    FN_cand.append([int(motion_b[0]), int(motion_b[1]), int(motion_b[2]), int(motion_b[3])])
                    cv2.rectangle(vis, (int(motion_b[0]), int(motion_b[1])), (int(motion_b[2]), int(motion_b[3])), (255, 0, 0), 2)
                    cv2.circle(vis, (int(central_point[0]), int(central_point[1])), 1, (255, 0, 0), -1)
                    cv2.putText(vis, "FN", (int(motion_b[0]), int(motion_b[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis, "FN ({:.3f}, {:.3f})".format(flow_ratio, motion_ratio), (int(motion_b[0]), int(motion_b[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    #continue
                    cv2.rectangle(vis, (int(motion_b[0]), int(motion_b[1])), (int(motion_b[2]), int(motion_b[3])), (255, 0, 255), 2)
                    cv2.circle(vis, (int(central_point[0]), int(central_point[1])), 1, (255, 0, 255), -1)
                    cv2.putText(vis, "fn", (int(motion_b[0]), int(motion_b[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)


        FN_box.append(FN_cand)
        if self.save_FN_img:
            if len(FN_cand) > 0:
                print('save FN')
                cv2.imwrite(os.path.join(self.config.img_path, "FN", str(self.save_index_FN).zfill(6) + ".png"), store_img) 
                #cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(self.save_index_FN).zfill(6) + ".png"), vis)

                """ determine the person in which zone"""
                #print("central_point2", central_point)
                if ( central_point[0] > self.zone_one[0] ) and ( central_point[0] < self.zone_one[2] ) and ( central_point[1] > self.zone_one[1] ) and (central_point[1] < self.zone_one[3] ):
                    print('zone one')
                    cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(self.save_index_FN).zfill(6) + "zone1.png"), vis)
                    cv2.imwrite(os.path.join(self.config.img_path, "zone_one", str(self.save_index_FN).zfill(6) + ".png"), store_img)

                elif ( central_point[0] > self.zone_two[0] ) and ( central_point[0] < self.zone_two[2] ) and ( central_point[1] > self.zone_two[1] ) and (central_point[1] < self.zone_two[3] ):
                    print('zone two')
                    cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(self.save_index_FN).zfill(6) + "zone2.png"), vis)
                    cv2.imwrite(os.path.join(self.config.img_path, "zone_two", str(self.save_index_FN).zfill(6) + ".png"), store_img) 

                elif ( central_point[0] > self.zone_three[0] ) and ( central_point[0] < self.zone_three[2] ) and ( central_point[1] > self.zone_three[1] ) and (central_point[1] < self.zone_three[3] ):
                    print('zone three')
                    cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(self.save_index_FN).zfill(6) + "zone3.png"), vis)
                    cv2.imwrite(os.path.join(self.config.img_path, "zone_three", str(self.save_index_FN).zfill(6) + ".png"), store_img) 

                elif ( central_point[0] > self.zone_four[0] ) and ( central_point[0] < self.zone_four[2] ) and ( central_point[1] > self.zone_four[1] ) and (central_point[1] < self.zone_four[3] ):
                    print('zone four')
                    cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(self.save_index_FN).zfill(6) + "zone4.png"), vis)
                    cv2.imwrite(os.path.join(self.config.img_path, "zone_four", str(self.save_index_FN).zfill(6) + ".png"), store_img) 
                
                self.save_index_FN += 1

                """ determine the distance of motion_bbox and entry_center """
                entry_center = self.entry_center(self.entry_mask)
                for entry_num in range(len(entry_center[0])):
                    dist = math.sqrt( (central_point[0] - entry_center[0][entry_num])**2 + (central_point[1] - entry_center[1][entry_num])**2 )                   
                    
                    if dist < self.entry_dist_thr:
                        print("save entry")                        
                        cv2.imwrite(os.path.join(self.config.img_path, "entry", str(self.save_index_FN).zfill(6) + ".png"), store_img)
                        cv2.rectangle(vis, (int(motion_b[0]), int(motion_b[1])), (int(motion_b[2]), int(motion_b[3])), (255, 0, 255), 2)
                        cv2.circle(vis, (int(central_point[0]), int(central_point[1])), 1, (255, 0, 255), -1)
                        cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(self.save_index_FN).zfill(6) + "entry.png"), vis)
                        self.save_index_FN += 1
                        break 

                #if self.need_ref_FN:
                #    if buff[0] is not None:
                #        cv2.imwrite(os.path.join(self.config.ref_path, "FN", "ref1-" + str(self.save_index_FN).zfill(6) + ".png"), buff[0])
                #        cv2.imwrite(os.path.join(self.config.ref_path, "FN", "ref2-" + str(self.save_index_FN).zfill(6) + ".png"), buff[1])
                #    else:
                #        cv2.imwrite(os.path.join(self.config.ref_path, "FN", "ref1-" + str(self.save_index_FN).zfill(6) + ".png"), buff[1])

                #self.save_index_FN += 1

        return FN_box

    def active_sample_door(self, doorseg):
        img = cv2.resize(doorseg, (800, 600))
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i, j, 0] == 100 and img[i, j, 1] == 100 and img[i, j, 2] == 100:
                    img[i, j, :] = [255, 255, 255]
                else:
                    img[i, j, :] = [0, 0, 0]

        return img

    def visualization(self, vis, thr, mapf):
        cv2.imshow("motion",thr)
        cv2.imshow("vis_result",vis)
        cv2.imshow("mapf", mapf)
        cv2.waitKey(100)

    @profile
    def run(self, videolist, predlist):
        prediction = glob.glob(os.path.join(predlist, '*.json'))
        prediction.sort()

        for path in prediction:
            # prediction file
            video_path = os.path.join(videolist, os.path.basename(path).replace(".json", ".mp4"))  # avi or mp4
            pred_file = json.load(open(path)) 

            print(path)
            print(video_path)
            
            #if (path[-13:-11] not in ["1", "2", "3"]):
            #    continue

            # read video
            cap = cv2.VideoCapture(video_path)

            #  FN cand
        
            FN_box = []
            
            self.save_FN_img = True  #add: self.save_FN_img = True: for deleting duplicate FN

            # buff for ref images
            buff = [None, None]
            
            # Start active sample
            self.index = 0 # for save images
            frame_index = 0 # for read video

            while(cap.isOpened()):
                # Read video image
                ret, img = cap.read()
                
                if ret:
                    img = cv2.resize(img, (self.config.img_w, self.config.img_h))
                    if self.privacy_processing:
                        img = privacy(img)
                    store_img = img.copy()
                    
                    if self.index == 0 and frame_index == 0:
                        frame1 = img.copy()
                        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                        hsv = np.zeros_like(frame1)
                        hsv[..., 1] = 255
                    else:                        
                        # segmentation pre-processing                   
                        img = img * self.segmask
                        
                        # active sample door
                        if self.activesampledoor:
                            raise NotImplementedError
                            rgb = rgb * (self.function_seg_img // 255)

                        # BG sub
                        thr = self.knn(img)
                        thr_fp = cv2.dilate(thr, self.kernel) 
                        vis, trigger, motion_box = self.findfg(thr, img)

                        # optical flow                        
                        OF_start = time.time()
                        next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        #flow = cv2.calcOpticalFlowPyrLK(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)                       
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv[...,0] = ang * 180 / np.pi / 2
                        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        # optical update
                        prvs = next
                         

                        mapf = np.sum(rgb, axis=2)
                        mapf = np.array(np.where(mapf > self.convert_opt_to_binary_th, 255, 0), dtype=np.uint8)
                        

                        # False Positive box detection
                        FN_box = self.false_negative(vis, pred_file['bbox'][self.index][frame_index], motion_box, thr, mapf, self.human_threshold, FN_box, self.save_index_FN, store_img, buff, self.save_FN_img)
                        
                        while [] in FN_box:
                            FN_box.remove([])
                        
                        for num_of_bbox in range(len(FN_box)):   
                             if len(FN_box) > 1:
                            
                                frame_iou = self.iou(FN_box[num_of_bbox][0], FN_box[num_of_bbox-1][0])
                                

                                if frame_iou > self.overlap_threshold:                                    
                                    self.save_FN_img = False
                                else:
                                    self.save_FN_img = True
                       
                        
                        
                        # append result
                        self.activesampleresult.append([FN_box])

                        if self.visual_flag:
                            self.visualization(vis, thr, mapf)
                    
                    # update ref image
                    buff[0], buff[1] = buff[1], store_img
                    # update video frame index
                    frame_index = (frame_index + 1) % 5
                        
                    if frame_index == 0:
                        self.index += 1
                else: 
                    # close cap
                    cap.release()
            

if __name__ == "__main__":

    username = 'user726'     # remember to change .avi or .mp4
    #user_list = ['user01', 'user02', 'user03', 'user04', 'user05', 'user06', 'user07', 'user08', 'user09', 'user10', 'user11',\
    #                'user12', 'user13', 'user14', 'user15']
    print("start", username, datetime.datetime.now())

    #while True:
    now = datetime.datetime.now()
    hour = now.strftime("%H")
    hour = int(hour)
    time_min = now.strftime("%M")
    time_min = int(time_min)
    time_sec = now.strftime("%S")
    time_sec = int(time_sec)
    #config.setup_info(username, yesterday)
    #config2.setup_info(username, previous_day)
    #activesample = ActiveSample(config)
    #activesample2 = ActiveSample(config2)
    #print(hour, time_min, time_sec)
    
    #if hour == 1 and time_min == 5:
    config = Config()
    now = datetime.datetime.now()
    ddelay = datetime.timedelta(days=1)
    #yesterday = now - ddelay
    #yesterday = yesterday.strftime("%Y-%m-%d")
    yesterday = '2021-06-28'
    config.setup_info(username, yesterday)
    activesample = ActiveSample(config)
    print("start3", username, datetime.datetime.now())
    start = time.time()
    create_folder(config.img_path, config.vis_path, config.ref_path)
    activesample.run('../' + username + '/' + yesterday, '../' + username + '/' + yesterday + '_prediction')
    print("total time:", time.time() - start)
            
        
    """
    if hour == 8 and time_min == 45:
        config = Config()
        now = datetime.datetime.now()
        ddelay = datetime.timedelta(days=3)
        yesterday = now - ddelay
        yesterday = yesterday.strftime("%Y-%m-%d")
        config.setup_info(username, yesterday)
        activesample = ActiveSample(config)
        print("start3", username, datetime.datetime.now())
        start = time.time()
        create_folder(config.img_path, config.vis_path, config.ref_path)
        activesample.run('../' + username + '/' + yesterday, '../' + username + '/' + yesterday + '_prediction')
        print("total time:", time.time() - start)
    """
    """
    #timer
    #print("start sched", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sched = BlockingScheduler()
    sched.add_job(activesample.run, 'cron', day_of_week='0-6', hour=00, minute=30, args = ['../' + username + '/' + date, '../' + username + '/' + date + '_prediction'])
    print(username, date, "start sched_", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sched.start()
    print("sched done", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    """

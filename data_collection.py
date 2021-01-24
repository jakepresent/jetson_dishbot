import jetson.inference
import jetson.utils
import cv2
import numpy as np
import timeit
from datetime import datetime
import os
from collections import deque

cwd = os.getcwd()

face_width = 640
face_height = 480

sink_width = 1280
sink_height = 720
sink_area = sink_width*sink_height
sink_motion_contour_threshold = sink_area*0.002
sink_background_contour_threshold = sink_area*0.03

face_cam = cv2.VideoCapture('/dev/video1')
sink_cam = cv2.VideoCapture('/dev/video0')

face_cam.set(cv2.CAP_PROP_FRAME_WIDTH, face_width)
face_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, face_height)
face_cam.set(cv2.CAP_PROP_FPS, 30)

sink_cam.set(cv2.CAP_PROP_FRAME_WIDTH, sink_width)
sink_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, sink_height)
sink_cam.set(cv2.CAP_PROP_FPS, 30)
sink_cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
sink_cam.set(cv2.CAP_PROP_FOCUS,0)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

face_frames_window = 30
face_frames_threshold = 0.3
face_save_period = 8

face_frames = [0]*face_frames_window
face_frames_avg = 0

class WeightedAverageBuffer:
        
    def __init__(self, maxl):
        self.maxlen = maxl
        self.buffer = deque(maxlen=maxl)
        self.shape = None

    def apply(self, frame):
        self.shape = frame.shape
        self.buffer.append(frame)
        
    def get_frame(self):
        mean_frame = np.zeros(self.shape, dtype='float32')
        i = 0
        for item in self.buffer:
            i += 4
            mean_frame += item*i
        mean_frame /= (i*(i + 1))/8.0
        return mean_frame.astype('uint8')
    
weighted_buffer = WeightedAverageBuffer(30)

def process_img(cam_name):
    if cam_name == 'face':
        _,frame = face_cam.read()
        width = face_width
        height = face_height
    elif cam_name == 'sink':
        _,frame = sink_cam.read()
        width = sink_width
        height = sink_height
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img = jetson.utils.cudaFromNumpy(img)
    detections = net.Detect(img, width, height)
    jetson.utils.cudaDeviceSynchronize()
    return (frame, img, detections)
    
def shift_detection_filter(face_detections):
    global face_frames, face_frames_avg
    face_found = False
    for detection in face_detections:
        if detection.ClassID == 1:
            face_found = True
            break
    face_frames = [face_found] + face_frames[:-1]
    face_frames_avg = sum(face_frames)/face_frames_window
    face_detected = face_frames_avg > face_frames_threshold
    return face_detected
    
def init_sink_background():
    _, sink_frame = sink_cam.read()
    back_sink_frame = sink_frame
    for i in range(10):
        _, sink_frame = sink_cam.read()
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        back_sink_frame = cv2.addWeighted(sink_frame, alpha, back_sink_frame, beta, 0.0)
    save_img('frame','SinkBackground',back_sink_frame)
    return back_sink_frame
    
def compare_background(frame, back_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    back_frame = cv2.cvtColor(back_frame, cv2.COLOR_BGR2GRAY)
    back_frame = cv2.GaussianBlur(back_frame, (21, 21), 0)
    diff_frame = cv2.absdiff(back_frame, gray)
    thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    diff = False
    max_area = 0
    for contour in cnts:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
        if area > sink_background_contour_threshold:
            diff = True
            #(x, y, w, h) = cv2.boundingRect(contour)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #save_img('frame','Sink',frame)
            #save_img('frame','Sink',thresh_frame)
    return (diff,max_area/sink_area)
            
def detect_motion(frame, prev_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
    diff_frame = cv2.absdiff(prev_frame, gray)
    thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    diff = False
    max_area = 0
    for contour in cnts:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
        if area > sink_motion_contour_threshold:
            diff = True
            #(x, y, w, h) = cv2.boundingRect(contour)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return (diff,max_area/sink_area)
            
def save_img(imtype,name,img):
    if name == 'Face':
        width = face_width
        height = face_height
    elif name == 'Sink' or name == 'SinkBackground':
        width = sink_width
        height = sink_height
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    if imtype == 'frame':
        cv2.imwrite(cwd+'/Images/'+dt_string+'_'+str(width)+'x'+str(height)+'_'+name+'Frame'+'.png', img)    
    elif imtype == 'img':
        jetson.utils.saveImageRGBA(cwd+'/Images/'+dt_string+'_'+str(width)+'x'+str(height)+'_'+name+'Img'+'.png', img, width, height)    
    
def main():
    state = 'idle'
    prev_state = state
    face_frame = []
    sink_frame = []
    prev_sink_frame = []
    back_sink_frame = []
    
    # Timer variables
    face_save_timer = 0
    face_save_timer_max = 10
    face_save_timer_reset = True
    
    face_no_detection_timer = timeit.default_timer()
    face_no_detection_timer_max = 6
    face_no_detection_timer_reset = True
    
    sink_no_motion_timer = timeit.default_timer()
    sink_no_motion_timer_max = 10
    sink_no_motion_timer_reset = True
    
    # Init
    for x in range(3):
        _, sink_frame = sink_cam.read()
        prev_sink_frame = sink_frame
    back_sink_frame = init_sink_background()
    
    while True:
        (face_frame, face_img, face_detections) = process_img('face')
        face_detected = shift_detection_filter(face_detections)                
        _, sink_frame = sink_cam.read()
        (motion_detected,motion_area) = detect_motion(sink_frame, prev_sink_frame)
        (change_detected,change_area) = compare_background(sink_frame, back_sink_frame)
        prev_sink_frame = sink_frame
        
        if motion_detected:
            sink_no_motion_timer = timeit.default_timer()
        sink_no_motion_timer_elapsed = timeit.default_timer()-sink_no_motion_timer
        
        if face_detected:
            face_no_detection_timer = timeit.default_timer()
        else:
            face_save_timer = timeit.default_timer()
        face_save_timer_elapsed = timeit.default_timer()-face_save_timer
        face_no_detection_timer_elapsed = timeit.default_timer()-face_no_detection_timer
        
        if face_no_detection_timer_elapsed > face_no_detection_timer_max and sink_no_motion_timer_elapsed > sink_no_motion_timer_max:
            if change_detected:
                back_sink_frame = init_sink_background()
                
        if face_save_timer_elapsed > face_save_timer_max:
            face_save_timer = timeit.default_timer()
            save_img('frame','Face',face_frame)
        
        print('Face Detected:',face_detected,' ',round(face_frames_avg,2),' ||',' Motion:',motion_detected,' Max:',round(motion_area*100,2),' Change:',change_detected,' Max:',round(change_area*100,2),' Face Save:',round(face_save_timer_elapsed,2))     
        
        prev_state = state
        
if __name__ == "__main__":
    main()
    

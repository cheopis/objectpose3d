import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from ultralytics import YOLO   
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

frame_shape = [480, 640]
IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_CX = 724.2716497016585/2
IMG_CY = 368.6598636244455/2

def run_mp(input_stream1, input_stream2, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]
    
    dji_height = 0.43 #m
    dji_width = 0.67 #m
    f_leght = 728.4652506156009

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    #create body keypoints detector objects.
    model = YOLO("best_01c.pt")
    conf = 0.25
    iou = 0.45

    #containers for detected keypoints for each camera. These are filled at each frame.
    #This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    n = 0

    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: break

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = model.predict(frame0, conf=conf, iou=iou, show=True)
        results1 = model.predict(frame1, conf=conf, iou=iou, show=True)

        #reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        #check for keypoints detection
        frame0_keypoints = []
        for result0 in results0:
            if result0.boxes:
                for i, box in enumerate(result0.boxes):
                    box = box.cpu()
                    x_min = np.int32(box.xyxy)[0][0]
                    x_max = np.int32(box.xyxy)[0][2]
                    y_min = np.int32(box.xyxy)[0][1]
                    y_max = np.int32(box.xyxy)[0][3]
                    
                    pxl_x = int(round(x_max + x_min)/2)
                    pxl_y = int(round(y_max + y_min)/2)
                    
                    cv.circle(frame0,(pxl_x, pxl_y), 3, (0,0,255), -1) #add keypoint detection points into figure
                    cv.rectangle(frame0, 
                          (x_min, y_min), 
                          (x_max, y_max), 
                          (0, 0, 255), 2)
                          
                    depth0 = (f_leght*dji_height)/abs(x_max - x_min)
                    depth1 = (f_leght*dji_width)/abs(y_max - y_min)
                    depth = 0.4*depth0 + 0.6*depth1
                          
                    kpts = [(pxl_x - IMG_CX)*depth/f_leght, 
                            (pxl_y - IMG_CY)*depth/f_leght,
                            depth]
                    frame0_keypoints.append(kpts)
            else:
                kpts = [-1, -1, -1]
                frame0_keypoints.append(kpts)

        #this will keep keypoints of this frame in memory
        kpts_cam0.append(frame0_keypoints)

        frame1_keypoints = []
        for result1 in results1:
            if result1.boxes:
                for i, box in enumerate(result1.boxes):
                    box = box.cpu()
                    x_min = np.int32(box.xyxy)[0][0]
                    x_max = np.int32(box.xyxy)[0][2]
                    y_min = np.int32(box.xyxy)[0][1]
                    y_max = np.int32(box.xyxy)[0][3]
                    
                    pxl_x = int(round(x_max + x_min)/2)
                    pxl_y = int(round(y_max + y_min)/2)
                    
                    cv.circle(frame1,(pxl_x, pxl_y), 3, (0,0,255), -1)
                    cv.rectangle(frame1, 
                          (x_min, y_min), 
                          (x_max, y_max), 
                          (0, 0, 255), 2)
                    
                    depth0 = (f_leght*dji_height)/abs(x_max - x_min)
                    depth1 = (f_leght*dji_width)/abs(y_max - y_min)
                    depth = 0.4*depth0 + 0.6*depth1
                          
                    kpts = [(pxl_x - IMG_CX)*depth/f_leght, 
                            (pxl_y - IMG_CY)*depth/f_leght,
                            depth]
                    frame1_keypoints.append(kpts)
            else:
                kpts = [-1, -1, -1]
                frame1_keypoints.append(kpts)

        #update keypoints container
        kpts_cam1.append(frame1_keypoints)

        #calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 and uv2[0] == -1:
                _p3d = [-1, -1, -1]
            elif uv1[0] == -1:
                _p3d = uv2#[0, (0.4*depth11 + 0.6*depth12), 0]111111111111111
            elif uv2[0] == -1:
                _p3d = uv1#[0, (0.4*depth01 + 0.6*depth02), 0]
            else:
                _p3d = [(x + y)/2 for x, y in zip(uv1, uv2)]
                #_p3d = [0, ((0.4*depth01 + 0.6*depth02)+(0.4*depth11 + 0.6*depth12))/2, 0]
                #_p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            print(_p3d)
            frame_p3ds.append(_p3d)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((1, 3))
        kpts_3d.append(frame_p3ds)

        # uncomment these if you want to see the full keypoints detections
        # mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #
        # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        
        cv.imwrite(f'/home/ariele/media/cam0/{str(n).zfill(4)}.jpg',frame0)
        cv.imwrite(f'/home/ariele/media/cam1/{str(n).zfill(4)}.jpg',frame1)
        
        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)
        n += 1
        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':

    #this will load the sample videos if no camera ID is given
    # input_stream1 = 'media/cam0_test.mp4'
    # input_stream2 = 'media/cam1_test.mp4'
    input_stream1 = 'media/video_down.mp4'
    input_stream2 = 'media/video_upper.mp4'

    #put camera id as command line arguements
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    #get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    #this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_cam_d.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam_u.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d_r.dat', kpts_3d)

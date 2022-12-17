import itertools
import os

from .bag_file_parser import BagFileParser
import pandas as pd
import numpy as np

import time as timetime
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

from random import sample

class InsertionExperimentBag( BagFileParser ):
    DEFAULT_TOPICS_OF_INTEREST = {
        "camera": [ "/".join([ns, topic, cam]) for ns, topic, cam in 
                      itertools.product(["/camera"],
                                         ["left", "right"], 
                                         ["image_raw", "image_rect_color", "camera_info"])
        ],
        "robot": [ "/".join([ns, topic, axis]) for ns, topic, axis in 
                      itertools.product(["/stage/axis"], 
                                        ["command", "position", "state/moving", "state/on"], 
                                        ["linear_stage", "x", "y", "z"])
        ],
        "needle": [ "/needle/sensor/raw", "/needle/sensor/processed",
                    "/needle/state/current_shape", "/needle/state/kappac","/needle/state/winit", 
                    "/needle/state/skin_entry", "/stage/state/needle_pose", "/needle/state/curvatures"
        ],
    }

    def __init__( self, bagdir: str, bagfile: str = None, yamlfile: str = None, topics: list = None ):
        super().__init__( bagdir, bagfile=bagfile, yamlfile=yamlfile )

        self.topics_of_interest = topics if topics is not None else InsertionExperimentBag.DEFAULT_TOPICS_OF_INTEREST

        # data containers
        self.camera_data = []
        self.robot_data  = []
        self.needle_data = []
        self.bagdir = bagdir
    # __init__

    def parse_data(self, camera: bool = False, robot: bool = False, str_side: str = "left", isGetTimestamps: bool = True):
        """ Parse the data  """
        num_experiment = 8

        # parse camera data
        if camera:
            str_side = str_side
            isGetTimestamps = isGetTimestamps



            bridge = CvBridge()

            #get the needle messages as a generator (to not overload RAM)
            needle_topics = list(filter(lambda t: f"/{str_side}/image_raw" in t, self.topics_of_interest['camera']))
            # print(needle_topics)
            bag_rows = self.get_messages(topic_name=needle_topics, generator_count=len(needle_topics))

            if isGetTimestamps:
                if str_side=="left":
                    ##### Find needed insertion depths timestamp #####
                    df_insertion_depths=pd.read_csv(str(self.bagdir)+'_axes_positions'+'.csv').filter(['ts', 'ls'])
                    # print(df_insertion_depths)
                    max_insertion_depth = -df_insertion_depths['ls'].loc[len(df_insertion_depths)-1]
                    num_samples = int((max_insertion_depth-30)//5)
                    print(num_samples)

                    list_sampled_depths = sample( range(30,int(max_insertion_depth)+1) , num_samples)
                    # print(list_sampled_depths)

                    df_sampled_depths = pd.DataFrame(columns= ['ts', 'ls'])
                    for depth in list_sampled_depths:
                        df_sampled_depths = pd.concat([df_sampled_depths, df_insertion_depths.iloc[ (df_insertion_depths['ls']+depth).abs().argsort()[:1] ]], ignore_index=True)

                    df_sampled_depths = df_sampled_depths.sort_values(by=['ts'])

                    list_ts = list(df_sampled_depths['ts'])
                    list_ls = list(df_sampled_depths['ls'])
                    print(list_ts)
                    df = pd.DataFrame( {'ts': list_ts, 'ls': list_ls})
                    df.to_csv( self.bagdir + f'_timestamps_ls'+'.csv', index=False)



                timestamps_camera= []
                # # iterate through the generator to find ts of images closest to ts of saved positions
                for i, rows in enumerate(bag_rows):
                    try:
                        # parse the set of rows
                        for ts,topic, msg in rows:
                            timestamps_camera.append(ts)
                            print(ts, topic)
                            if topic==f'/camera/{str_side}/image_raw':
                                images_raw = {
                                    topic.replace(f'/camera/{str_side}/image_raw', str_side) : (ts, msg)
                                }
                                # make sure this set has all 4 messages
                                if len(images_raw) != len(needle_topics):
                                    continue
                                # print(len(v) for v in images_raw.values())
                        

                    except:
                        print('*********EXCEPTION*************')
                        df = pd.DataFrame(np.array(timestamps_camera), columns=['ts'])
                        df.to_csv( self.bagdir + f'_timestamps_{str_side}'+'.csv', index=False)
                        print("saved timestamps")
                        break
                

            if not isGetTimestamps:

                ### Find the closest ts in images for sampled positions
                list_ts_ls = list(pd.read_csv(self.bagdir + f'_timestamps_ls'+'.csv')['ts'])
                list_ls = list(pd.read_csv(self.bagdir + f'_timestamps_ls'+'.csv')['ls'])
                list_ts_img = list(pd.read_csv(self.bagdir + f'_timestamps_{str_side}'+'.csv')['ts'])

                list_sampled_ts_img = []
                for ts in list_ts_ls:
                    list_sampled_ts_img.append( list_ts_img[min(range(len(list_ts_img)), key=lambda i: abs(list_ts_img[i]-ts))] )

                print(list_sampled_ts_img)
                    


                path=f"./{str_side}_imgs/"
                path_ref=f"./ref_{str_side}_imgs/"
                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(path_ref):
                    os.makedirs(path_ref)
                
                # # iterate through the generator to save images
                for i, rows in enumerate(bag_rows):
                    # parse the set of rows
                    
                    try:
                        for ts,topic, msg in rows:
                            print(ts, topic)
                            if topic==f'/camera/{str_side}/image_raw':
                                images_raw = {
                                    topic.replace(f'/camera/{str_side}/image_raw', str_side) : (ts, msg)
                                }
                                if len(images_raw) != len(needle_topics):
                                    continue
                        
                        #Save reference image first
                        if i==0:
                            try:
                                cv2_img = bridge.imgmsg_to_cv2(images_raw[list(images_raw.keys())[0]][1], "bgr8")
                            except CvBridgeError as e:
                                print(e)
                            else:
                                print("saving reference", str_side)
                                cv2.imwrite(os.path.join(path_ref , str(self.bagdir) + '.png'), cv2_img)
                                print("saved reference", str_side)

                        for side,img in images_raw.items():
                            if ts==list_sampled_ts_img[0]:
                                
                                try:
                                    cv2_img = bridge.imgmsg_to_cv2(img[1], "bgr8")
                                except CvBridgeError as e:
                                    print(e)
                                else:
                                    ### Save your OpenCV2 image as a png
                                    # time = img[1].header.stamp
                                    
                                    cv2.imwrite(os.path.join(path , str(self.bagdir) + '_'+str(int(-list_ls[0]))+'.png'), cv2_img)
                                    # timetime.sleep(1)
                                list_ls.pop(0)
                                list_sampled_ts_img.pop(0)

                    except:
                        print('*********EXCEPTION*************')
                        print("saved images")
                        break



        # if: camera

        # parse robot data
        if robot:
            # get the position messages as a generator (to not overload RAM)
            robot_position_topics = list(filter(lambda t: "/position/" in t, self.topics_of_interest['robot']))
            bag_rows = self.get_messages(topic_name=robot_position_topics, generator_count=len(robot_position_topics))

            # iterate through the generator
            for i, rows in enumerate(bag_rows):
                # parse the set of rows
                try:
                    robot_positions = {topic.replace('/stage/axis/position/', '') : (ts, msg.data) for ts, topic, msg in rows}
                # timestamp = min([ts for ts, *_ in robot_positions.values()])
                

                    # make sure this set has all 4 messages
                    if len(robot_positions) != len(robot_position_topics):
                        continue
                    # print(  robot_positions['x'][0]     )
                    # append to robot data
                    tupl=( robot_positions['x'][0], #timestamp
                          robot_positions['x'][1],
                          robot_positions['y'][1],
                          robot_positions['z'][1],
                          robot_positions['linear_stage'][1] )
                    self.robot_data.append(
                        tupl
                    )
                    print(tupl)

                except:
                    print("exception caught")
                    df = pd.DataFrame(np.array(self.robot_data), columns=['ts', 'x', 'y', 'z', 'ls'])
                    df.to_csv(str(self.bagdir)+ '_axes_positions'+'.csv', index=False)
                    print("saved axes positions")
                    break

            #  for
        # if: robot
            


# class: InsertionExperimentBag

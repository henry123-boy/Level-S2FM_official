from collections import deque, OrderedDict
import easydict
import argparse
import os 
import numpy as np 
import copy 
import cv2
from scipy.spatial.transform import Rotation
from PIL import Image


import os
import json
import time
import logging as log
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from viewer import PipelineView

class PipelineController:
    def __init__(self, filesdict, width, height, dest, render=True):
        self.view = PipelineView(filesdict, 
            width=width,
            height=height,
            dest=dest,
            on_toggle_capture=self.on_toggle_capture,
            on_window_close=self.on_window_close,
            on_toggle_fitview = self.on_toggle_fitview,
            on_toggle_two_view = self.on_toogle_two_view,
            on_capture = self.on_capture,
            on_clear = self.on_clear,
            on_toggle_mesh = self.on_toggle_mesh,
            )
        self.render = render
        if self.render:
            threading.Thread(target=self.update_thread).start()
        else:
            threading.Thread(target=self.update_view(all=True)).start()

        gui.Application.instance.run()

    def update_view(self, all=False):
        """Updates view with new data. May be called from any thread.

        Args:
            frame_elements (dict): Display elements (point cloud and images)
                from the new frame to be shown.
        """
        gui.Application.instance.post_to_main_thread(
            self.view.window,
            lambda: self.view.update(all=all))
    
    def update_thread(self):
        N = len(self.view.filesdict)
        for i in range(N):
            # self.update_view(self.view.filesdict[i])
            time.sleep(0.5)
            # gui.Application.instance.post_to_main_thread(
                # self.view.window, self.view.update({}))
            self.update_view()

        rendered_images = []        
        for i in range(1,len(self.view.filesdict)):
            rendered_images.append(
                Image.open(f'{self.view.dest}/{i:06d}.jpg'))

        rendered_images[1].save(f'{self.view.dest}.gif', save_all=True, append_images=rendered_images[1:], optimize=True, duration=100, loop=0)    
        
        # self.view.window.close()
    def on_toggle_mesh(self, state):
        gui.Application.instance.post_to_main_thread(
            self.view.window,
            lambda : self.view.toggle_mesh(state))
    def on_clear(self):
        gui.Application.instance.post_to_main_thread(
            self.view.window,
            self.view.clear_geometry)
    def on_capture(self):
        gui.Application.instance.post_to_main_thread(
            self.view.window,
            self.view.capture
        )
    def on_toogle_two_view(self):
        gui.Application.instance.post_to_main_thread(
            self.view.window,
            self.view.toggle_twoview,

        )
    def on_toggle_capture(self):
        self.update_view()
        
    def on_toggle_fitview(self):
        
        gui.Application.instance.post_to_main_thread(
            self.view.window,
            lambda: self.view.toggle_fitview(True)
        )
        #reset the toggle button


    def on_window_close(self):
        """Callback when the user closes the application window."""
        # self.pipeline_model.flag_exit = True
        # with self.pipeline_model.cv_capture:
            # self.pipeline_model.cv_capture.notify_all()
        return True  # OK to close window
